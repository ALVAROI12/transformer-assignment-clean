"""
Training script for GPT-style transformer on character-level Shakespeare data.

Based on training practices from:
1. "Attention Is All You Need" (original transformer training)
2. "Language Models are Unsupervised Multitask Learners" (GPT-2 training)
3. Andrej Karpathy's makemore and nanoGPT repositories
4. PyTorch official examples

Key training techniques used:
- AdamW optimizer (better than Adam for transformers)
- Cosine learning rate decay with warmup
- Gradient clipping (prevents exploding gradients)
- Mixed precision training (faster on modern GPUs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import time
import pickle
from tqdm import tqdm
import logging
from model import GPTTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CharDataset(Dataset):
    """
    Character-level dataset for autoregressive language modeling.
    
    Each sample returns a sequence of characters and the next character
    as the target (teacher forcing for training).
    """
    
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Grab a chunk of data and return input and target sequences
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # input
        y = torch.tensor(chunk[1:], dtype=torch.long)   # target (shifted by 1)
        return x, y

def get_data():
    """
    Download and prepare the Tiny Shakespeare dataset.
    Returns train/val splits and character mappings.
    """
    # Download dataset if not exists
    if not os.path.exists('input.txt'):
        logger.info("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open('input.txt', 'w') as f:
            f.write(requests.get(url).text)
    
    # Read and prepare text
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    logger.info(f"Dataset length: {len(text):,} characters")
    
    # Get all unique characters and create mappings
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    logger.info(f"Vocabulary size: {vocab_size} unique characters")
    logger.info(f"Vocabulary: {''.join(chars)}")
    
    # Create character-to-integer and integer-to-character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the entire dataset
    data = [stoi[c] for c in text]
    
    # Train/validation split (90%/10%)
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    logger.info(f"Train dataset: {len(train_data):,} tokens")
    logger.info(f"Val dataset: {len(val_data):,} tokens")
    
    return train_data, val_data, stoi, itos, vocab_size

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """
    Learning rate scheduler with warmup and cosine decay.
    This is the standard approach used in most modern transformer training.
    """
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio)) # cosine decay
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters=100):
    """
    Estimate loss on train and validation sets.
    
    This function switches model to eval mode, estimates loss on both
    train and validation sets, then switches back to train mode.
    """
    out = {}
    model.eval()
    
    for split, data_loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(data_loader):
            if k >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_compile):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

# Training configuration - based on successful GPT training runs
config = {
    # Model hyperparameters
    'vocab_size': None,  # Will be set after loading data
    'd_model': 384,      # Slightly larger than assignment suggestion
    'n_heads': 6,        # 384/6 = 64 dim per head (good choice)
    'n_layers': 6,       # Deeper model for better performance
    'dropout': 0.2,      # Higher dropout for small dataset
    'max_seq_len': 256,  # Longer context than minimum requirement
    
    # Training hyperparameters  
    'batch_size': 64,         # Good for 8GB GPU
    'learning_rate': 3e-4,    # Standard for transformers
    'max_iters': 5000,        # Reasonable for this dataset size
    'weight_decay': 1e-1,     # L2 regularization
    'beta1': 0.9,             # Adam beta1
    'beta2': 0.95,            # Adam beta2 (slightly lower than default)
    'grad_clip': 1.0,         # Gradient clipping value
    
    # Learning rate schedule
    'decay_lr': True,         # Whether to decay learning rate
    'warmup_iters': 100,      # Warmup steps
    'lr_decay_iters': 5000,   # Steps to decay over
    'min_lr': 3e-5,           # Minimum learning rate (10% of max)
    
    # Evaluation and logging
    'eval_interval': 250,     # How often to evaluate
    'log_interval': 10,       # How often to log
    'eval_iters': 100,        # Number of iterations for loss estimation
    
    # System
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'compile': True,          # Use torch.compile for better performance
}

def train():
    """
    Main training function.
    """
    global device, use_compile
    
    # Set device and compilation
    device = config['device']
    use_compile = config['compile'] and device == 'cuda'
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Using torch.compile: {use_compile}")
    
    # Load and prepare data
    train_data, val_data, stoi, itos, vocab_size = get_data()
    config['vocab_size'] = vocab_size
    
    # Create datasets and data loaders
    train_dataset = CharDataset(train_data, config['max_seq_len'])
    val_dataset = CharDataset(val_data, config['max_seq_len'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Keep 0 for simplicity
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    model = GPTTransformer(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    )
    model.to(device)
    
    # Compile model for better performance (PyTorch 2.0+)
    if use_compile:
        logger.info("Compiling model...")
        model = torch.compile(model)
    
    # Create optimizer (AdamW is better than Adam for transformers)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        weight_decay=config['weight_decay']
    )
    
    # Initialize GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    
    # Training loop
    train_losses = []
    val_losses = []
    learning_rates = []
    iter_num = 0
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    start_time = time.time()
    
    # Training loop
    train_iter = iter(train_loader)
    for iter_num in range(config['max_iters']):
        
        # Determine and set learning rate
        lr = get_lr(iter_num, config['learning_rate'], config['warmup_iters'], 
                   config['lr_decay_iters'], config['min_lr']) if config['decay_lr'] else config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        learning_rates.append(lr)
        
        # Evaluate model periodically
        if iter_num % config['eval_interval'] == 0:
            losses = estimate_loss(model, train_loader, val_loader, config['eval_iters'])
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            
            logger.info(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'stoi': stoi,
                    'itos': itos,
                }
                logger.info(f"Saving checkpoint to ckpt.pt")
                torch.save(checkpoint, 'ckpt.pt')
        
        # Get next batch
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
        
        X, Y = X.to(device), Y.to(device)
        
        # Forward pass with mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            logits, loss = model(X, Y)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Clip gradients
        if config['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Log training progress
        if iter_num % config['log_interval'] == 0:
            logger.info(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:.2e}")
    
    # Final evaluation
    losses = estimate_loss(model, train_loader, val_loader, config['eval_iters'])
    logger.info(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Save final checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': config,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
        'stoi': stoi,
        'itos': itos,
    }
    torch.save(checkpoint, 'ckpt_final.pt')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    eval_steps = np.arange(0, len(train_losses)) * config['eval_interval']
    plt.plot(eval_steps, train_losses, label='train')
    plt.plot(eval_steps, val_losses, label='val')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates)
    plt.xlabel('iterations')
    plt.ylabel('learning rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return model, stoi, itos

if __name__ == "__main__":
    # Train the model
    model, stoi, itos = train()
    logger.info("Training finished! Model saved to ckpt.pt")