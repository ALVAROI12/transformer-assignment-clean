"""
Transformer implementation based on:
1. "Attention Is All You Need" (Vaswani et al., 2017)
2. GPT architecture (Radford et al., 2018)
3. Andrej Karpathy's minGPT implementation
4. PyTorch official transformer tutorial

Key references:
- Original paper: https://arxiv.org/abs/1706.03762
- GPT paper: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- minGPT: https://github.com/karpathy/minGPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from "Attention Is All You Need"
    
    This implementation follows GPT-2 style efficiency improvements:
    - Combined QKV projections for better memory usage
    - Proper causal masking for autoregressive generation
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Combined linear transformation for Q, K, V (GPT-2 style)
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Dropout layers as in original transformer
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()  # batch_size, seq_len, d_model
        
        # Calculate Q, K, V in batch (more efficient than separate operations)
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        # Reshape for multi-head attention: (B, T, C) -> (B, nh, T, hs)
        # where nh = n_heads, hs = head_size = C // nh
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Scaled dot-product attention (Equation 1 from paper)
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask for autoregressive generation
        if mask is not None:
            att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = torch.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network from original transformer paper.
    Uses the standard architecture: Linear -> ReLU -> Linear
    
    The paper uses d_ff = 4 * d_model, but we make it configurable.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2 (Equation 2 from paper)
        x = self.c_fc(x)
        x = F.relu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need"
    
    Uses sine and cosine functions of different frequencies:
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even positions, cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward layers.
    Uses pre-layer normalization (LayerNorm before attention/FFN) which is
    more stable for training than post-layer norm from original paper.
    
    This follows the GPT-2 architecture.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)  # Pre-attention layer norm
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)  # Pre-FFN layer norm
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        # Pre-layer norm (GPT-2 style) - more stable than post-layer norm
        x = x + self.attn(self.ln_1(x), mask)  # residual connection
        x = x + self.ffn(self.ln_2(x))         # residual connection
        return x

class GPTTransformer(nn.Module):
    """
    GPT-style decoder-only transformer for autoregressive language modeling.
    
    Based on:
    - Original GPT architecture (Radford et al., 2018)
    - Improvements from GPT-2 (Radford et al., 2019)
    - Clean implementation patterns from minGPT (Karpathy)
    """
    
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, 
                 d_ff=None, max_seq_len=1024, dropout=0.1):
        super().__init__()
        
        # Default d_ff to 4 * d_model as in original paper
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.wte = nn.Embedding(vocab_size, d_model)  # token embeddings
        self.wpe = nn.Embedding(max_seq_len, d_model)  # learned positional embeddings
        
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights following GPT-2 initialization
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")
        
    def _init_weights(self, module):
        """
        Initialize weights following GPT-2 initialization scheme:
        - Normal initialization with std=0.02 for most layers
        - Special initialization for residual projections
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_block_size(self):
        return self.max_seq_len
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.max_seq_len, f"Cannot forward sequence of length {t}, max is {self.max_seq_len}"
        
        # Create position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        
        # Token embeddings and positional embeddings
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, d_model)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, d_model)
        x = self.drop(tok_emb + pos_emb)
        
        # Create causal mask (lower triangular matrix)
        mask = torch.tril(torch.ones(t, t, device=device)).view(1, 1, t, t)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), 
                                 ignore_index=-1)
        
        return logits, loss

# Test the model
if __name__ == "__main__":
    # Test configuration
    vocab_size = 50257  # GPT-2 vocabulary size
    model = GPTTransformer(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_seq_len=128
    )
    
    # Test input
    batch_size, seq_len = 2, 64
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, loss = model(idx)
    print(f"Input shape: {idx.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model ready for training!")