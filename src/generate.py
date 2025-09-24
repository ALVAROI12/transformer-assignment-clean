"""
Text generation script for trained GPT-style transformer.

Implements various sampling strategies used in modern language models:
1. Greedy sampling (deterministic)
2. Temperature sampling (controls randomness)
3. Top-k sampling (Radford et al., 2019 - GPT-2)
4. Top-p/nucleus sampling (Holtzman et al., 2019)

Based on:
- "Language Models are Unsupervised Multitask Learners" (GPT-2)
- "The Curious Case of Neural Text Degeneration" (nucleus sampling)
- Andrej Karpathy's nanoGPT generation techniques
- Hugging Face transformers generation utilities
"""

import torch
import torch.nn.functional as F
from model import GPTTransformer
import argparse
import os
import pickle

def top_k_logits(logits, k):
    """
    Filter a distribution of logits using top-k filtering.
    
    Args:
        logits: Tensor of shape (vocab_size,)
        k: Number of top tokens to keep
        
    Returns:
        Filtered logits with non-top-k tokens set to -inf
    """
    # Ensure k doesn't exceed vocabulary size
    k = min(k, logits.size(-1))
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[-1]] = -float('inf')
    return out

def top_p_logits(logits, p):
    """
    Filter a distribution of logits using nucleus (top-p) filtering.
    
    Nucleus filtering removes tokens with cumulative probability above the threshold p.
    This is more adaptive than top-k as it adjusts the number of tokens based on
    the probability distribution.
    
    Args:
        logits: Tensor of shape (vocab_size,)
        p: Cumulative probability threshold
        
    Returns:
        Filtered logits
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token (the most likely one)
    sorted_indices_to_remove[0] = False
    
    # Convert back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('inf')
    return logits

def sample(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Sample from a probability distribution over tokens.
    
    Args:
        logits: Tensor of shape (vocab_size,) - raw logits from model
        temperature: Float > 0, controls randomness. 1.0 = no change, < 1.0 = less random, > 1.0 = more random
        top_k: Integer, only sample from top k tokens
        top_p: Float in (0,1), nucleus sampling threshold
        
    Returns:
        Sampled token index
    """
    # Apply temperature scaling
    logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    # Apply top-p filtering
    if top_p is not None:
        logits = top_p_logits(logits, top_p)
    
    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, device='cuda'):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained GPT model
        idx: Tensor of shape (B, T) with context tokens
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Nucleus sampling parameter  
        device: Device to run on
        
    Returns:
        Generated sequence including original context
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # If sequence is longer than model's context, crop to fit
        idx_cond = idx if idx.size(1) <= model.get_block_size() else idx[:, -model.get_block_size():]
        
        # Forward pass to get logits
        logits, _ = model(idx_cond)
        
        # Get logits for the last token in sequence
        logits = logits[:, -1, :]  # Shape: (B, vocab_size)
        
        # Sample next token for each sequence in batch
        idx_next = torch.zeros(idx.size(0), 1, dtype=torch.long, device=device)
        
        for i in range(idx.size(0)):
            idx_next[i, 0] = sample(logits[i], temperature=temperature, top_k=top_k, top_p=top_p)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def load_model(ckpt_path='ckpt.pt'):
    """
    Load trained model and associated data.
    
    Returns:
        model: Loaded GPT model
        stoi: String to integer mapping
        itos: Integer to string mapping  
        device: Device model is on
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract configuration and mappings
    model_args = checkpoint['model_args']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Create model
    model = GPTTransformer(
        vocab_size=model_args['vocab_size'],
        d_model=model_args['d_model'],
        n_heads=model_args['n_heads'],
        n_layers=model_args['n_layers'],
        dropout=0.0,  # No dropout during inference
        max_seq_len=model_args['max_seq_len']
    )
    
    # Load weights
    state_dict = checkpoint['model']
    # Remove 'module.' prefix if it exists (from DataParallel)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {ckpt_path}")
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Running on {device}")
    
    return model, stoi, itos, device

def encode(text, stoi):
    """Convert string to list of integers."""
    return [stoi[c] for c in text]

def decode(tokens, itos):
    """Convert list of integers to string."""
    return ''.join([itos[i] for i in tokens])

def main():
    parser = argparse.ArgumentParser(description='Generate text with trained GPT model')
    parser.add_argument('--checkpoint', type=str, default='ckpt.pt', help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='ROMEO:', help='Text prompt for generation')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k filtering (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling threshold (0 = disabled)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint {args.checkpoint} not found. Please train a model first.")
        return
    
    model, stoi, itos, device = load_model(args.checkpoint)
    
    # Prepare sampling parameters
    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p > 0 else None
    
    print(f"\nGenerating {args.num_samples} samples with:")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")
    print("=" * 80)
    
    # Generate samples
    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1} ---")
        
        # Encode prompt
        start_ids = encode(args.prompt, stoi)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        # Generate
        with torch.no_grad():
            y = generate(model, x, args.max_new_tokens, 
                        temperature=args.temperature, top_k=top_k, top_p=top_p, device=device)
            
            # Decode and print
            generated_text = decode(y[0].tolist(), itos)
            print(generated_text)
            print("-" * 50)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("Enter prompts to generate text (or 'quit' to exit)")
    print("=" * 80)
    
    while True:
        try:
            user_prompt = input("\nPrompt: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_prompt:
                user_prompt = "ROMEO:"
            
            # Generate
            start_ids = encode(user_prompt, stoi)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            
            with torch.no_grad():
                y = generate(model, x, args.max_new_tokens,
                           temperature=args.temperature, top_k=top_k, top_p=top_p, device=device)
                
                generated_text = decode(y[0].tolist(), itos)
                print("\nGenerated:")
                print("-" * 40)
                print(generated_text)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\nGoodbye!")

def demo_different_strategies():
    """
    Demonstrate different sampling strategies with the same prompt.
    This function shows how different hyperparameters affect generation.
    """
    print("Loading model for sampling strategy demonstration...")
    model, stoi, itos, device = load_model()
    
    prompt = "ROMEO:"
    start_ids = encode(prompt, stoi)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    strategies = [
        ("Greedy (temperature=0.1)", {"temperature": 0.1, "top_k": None, "top_p": None}),
        ("Low temperature", {"temperature": 0.5, "top_k": None, "top_p": None}),
        ("Medium temperature", {"temperature": 0.8, "top_k": None, "top_p": None}),
        ("High temperature", {"temperature": 1.2, "top_k": None, "top_p": None}),
        ("Top-k sampling", {"temperature": 0.8, "top_k": 20, "top_p": None}),
        ("Nucleus sampling", {"temperature": 0.8, "top_k": None, "top_p": 0.9}),
        ("Combined (k+p)", {"temperature": 0.8, "top_k": 40, "top_p": 0.9}),
    ]
    
    print(f"Prompt: '{prompt}'")
    print("=" * 80)
    
    for name, params in strategies:
        print(f"\n{name}:")
        print("-" * 40)
        
        with torch.no_grad():
            y = generate(model, x, 150, device=device, **params)
            generated_text = decode(y[0].tolist(), itos)
            print(generated_text)

if __name__ == "__main__":
    # Run main generation script
    main()
    
    # Uncomment to run sampling strategy demo
    # demo_different_strategies()