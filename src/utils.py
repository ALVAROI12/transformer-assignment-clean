"""
Utility functions for transformer training and evaluation.

Contains helper functions commonly used in NLP research:
1. Model parameter counting and memory estimation
2. Learning rate scheduling utilities  
3. Text preprocessing and tokenization helpers
4. Evaluation metrics (perplexity, etc.)
5. Visualization utilities for attention patterns
6. Model analysis and debugging tools

Based on common practices from:
- PyTorch official examples and tutorials
- Hugging Face transformers utilities
- Research codebases (nanoGPT, minGPT, etc.)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import time
import psutil
import os

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Returns both total and trainable parameters, which is standard
    practice in ML papers for reporting model size.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'total_million': total_params / 1e6,
        'trainable_million': trainable_params / 1e6
    }

def estimate_model_size(model: nn.Module, input_shape: Tuple[int, ...] = None) -> Dict[str, float]:
    """
    Estimate model memory usage.
    
    This is useful for determining if a model will fit in GPU memory
    before starting training.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, seq_len)
        
    Returns:
        Memory usage estimates in MB
    """
    param_count = count_parameters(model)
    
    # Parameter memory (assume float32 = 4 bytes)
    param_memory_mb = param_count['total'] * 4 / (1024 ** 2)
    
    # Gradient memory (same as parameters for standard training)
    grad_memory_mb = param_memory_mb
    
    # Optimizer memory (AdamW stores 2 momentum terms per parameter)
    optimizer_memory_mb = param_memory_mb * 2
    
    # Activation memory (rough estimate if input shape provided)
    activation_memory_mb = 0
    if input_shape is not None:
        batch_size, seq_len = input_shape
        # Very rough estimate: assume activations scale with model size and sequence length
        activation_memory_mb = (param_count['total'] * seq_len * batch_size * 4) / (1024 ** 2) * 0.1
    
    total_memory_mb = param_memory_mb + grad_memory_mb + optimizer_memory_mb + activation_memory_mb
    
    return {
        'parameters_mb': param_memory_mb,
        'gradients_mb': grad_memory_mb,
        'optimizer_mb': optimizer_memory_mb,
        'activations_mb': activation_memory_mb,
        'total_estimated_mb': total_memory_mb,
        'total_estimated_gb': total_memory_mb / 1024
    }

def get_system_info() -> Dict[str, str]:
    """
    Get system information for reproducibility and debugging.
    
    Returns system specs that are useful to include in experiment logs.
    """
    info = {
        'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        'pytorch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_available': True,
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
        })
    else:
        info['cuda_available'] = False
    
    return info

def print_model_info(model: nn.Module, input_shape: Tuple[int, ...] = None):
    """
    Print comprehensive model information.
    
    This is a common utility function used in research codebases
    to understand model complexity and memory requirements.
    """
    param_info = count_parameters(model)
    memory_info = estimate_model_size(model, input_shape)
    system_info = get_system_info()
    
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    
    print(f"Total parameters: {param_info['total']:,} ({param_info['total_million']:.2f}M)")
    print(f"Trainable parameters: {param_info['trainable']:,} ({param_info['trainable_million']:.2f}M)")
    
    print("\nESTIMATED MEMORY USAGE:")
    print(f"Parameters: {memory_info['parameters_mb']:.1f} MB")
    print(f"Gradients: {memory_info['gradients_mb']:.1f} MB") 
    print(f"Optimizer: {memory_info['optimizer_mb']:.1f} MB")
    if input_shape:
        print(f"Activations: {memory_info['activations_mb']:.1f} MB")
    print(f"Total estimated: {memory_info['total_estimated_gb']:.2f} GB")
    
    print("\nSYSTEM INFO:")
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    print("=" * 60)

def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Perplexity is a standard metric in language modeling that represents
    the average number of choices the model has for each token.
    Lower perplexity = better model.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()

def create_learning_rate_schedule(
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int
) -> List[float]:
    """
    Create learning rate schedule with warmup and cosine decay.
    
    This is the standard LR schedule used in most transformer training.
    
    Args:
        max_lr: Peak learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        
    Returns:
        List of learning rates for each step
    """
    lrs = []
    
    for step in range(max_steps):
        if step < warmup_steps:
            # Linear warmup
            lr = max_lr * step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        lrs.append(lr)
    
    return lrs

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    learning_rates: List[float] = None,
    save_path: str = 'training_curves.png'
):
    """
    Plot training curves with professional styling.
    
    Creates publication-ready plots commonly used in ML papers.
    """
    fig_width = 15 if learning_rates else 10
    fig, axes = plt.subplots(1, 3 if learning_rates else 2, figsize=(fig_width, 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # Loss curves
    epochs = range(len(train_losses))
    axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2, alpha=0.8)
    axes[0].plot(epochs, val_losses, label='Validation Loss', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Evaluation Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity
    train_perplexity = [calculate_perplexity(loss) for loss in train_losses]
    val_perplexity = [calculate_perplexity(loss) for loss in val_losses]
    
    axes[1].plot(epochs, train_perplexity, label='Train Perplexity', linewidth=2, alpha=0.8)
    axes[1].plot(epochs, val_perplexity, label='Val Perplexity', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Evaluation Steps')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Perplexity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # Learning rate schedule
    if learning_rates:
        steps = range(len(learning_rates))
        axes[2].plot(steps, learning_rates, linewidth=2, alpha=0.8, color='orange')
        axes[2].set_xlabel('Training Steps')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to {save_path}")

def save_training_log(
    config: Dict,
    train_losses: List[float],
    val_losses: List[float],
    learning_rates: List[float] = None,
    save_path: str = 'training_log.txt'
):
    """
    Save detailed training log for reproducibility.
    
    This creates a comprehensive log file that's essential for
    research reproducibility and experiment tracking.
    """
    system_info = get_system_info()
    
    with open(save_path, 'w') as f:
        f.write("TRANSFORMER TRAINING LOG\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SYSTEM INFORMATION:\n")
        for key, value in system_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("MODEL CONFIGURATION:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write(f"Final train loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final val loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final train perplexity: {calculate_perplexity(train_losses[-1]):.2f}\n")
        f.write(f"Final val perplexity: {calculate_perplexity(val_losses[-1]):.2f}\n")
        f.write(f"Best val loss: {min(val_losses):.4f}\n")
        
        if learning_rates:
            f.write(f"Max learning rate: {max(learning_rates):.2e}\n")
            f.write(f"Final learning rate: {learning_rates[-1]:.2e}\n")
        
        f.write(f"\nTotal training evaluations: {len(train_losses)}\n")
        
    print(f"Training log saved to {save_path}")

def benchmark_model(model: nn.Module, input_shape: Tuple[int, int], device: str = 'cuda', num_runs: int = 100):
    """
    Benchmark model inference speed.
    
    This is useful for understanding model performance characteristics,
    especially when comparing different architectures.
    """
    model.eval()
    batch_size, seq_len = input_shape
    
    # Warm up GPU
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    print(f"Inference benchmark ({num_runs} runs):")
    print(f"Average time per batch: {avg_time*1000:.2f} ms")
    print(f"Tokens per second: {tokens_per_second:.0f}")
    print(f"Batches per second: {1/avg_time:.1f}")
    
    return {
        'avg_time_ms': avg_time * 1000,
        'tokens_per_second': tokens_per_second,
        'batches_per_second': 1/avg_time
    }

def check_gradient_flow(model: nn.Module) -> Dict[str, float]:
    """
    Check gradient flow through the model.
    
    This debugging utility helps identify vanishing/exploding gradient problems
    that are common in deep networks.
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    
    return {
        'layer_names': layers,
        'average_gradients': ave_grads,
        'max_gradients': max_grads,
        'gradient_norm': sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    }

def setup_logging(log_file: str = 'training.log'):
    """
    Set up logging for training.
    
    Creates both console and file logging that's commonly used
    in research code for experiment tracking.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Example usage functions for common operations
def quick_model_test():
    """
    Quick test to verify model implementation works.
    """
    from model import GPTTransformer
    
    print("Testing model implementation...")
    
    model = GPTTransformer(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=64
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    x = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(x, x)  # Use x as both input and target for test
    
    print(f"✅ Model test passed!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    print_model_info(model, (batch_size, seq_len))

if __name__ == "__main__":
    # Run quick model test
    quick_model_test()