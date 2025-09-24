# Transformer Implementation from Scratch

**Assignment 1: Understanding & Implementing Transformers**  
*CPE-4953-001, EE-5453-005: Embedded Systems for AI*

## Overview

This project implements a GPT-style transformer model from scratch using PyTorch, trains it on the Tiny Shakespeare dataset, and performs comprehensive system analysis. The implementation follows the original "Attention Is All You Need" paper while incorporating modern training practices.

## Results Summary

- **Model Size**: 10.8M parameters
- **Training Time**: 11.3 minutes on RTX 4060 Laptop GPU
- **Final Training Loss**: 1.0564
- **Final Validation Loss**: 1.2575
- **Text Quality**: Generates coherent Shakespeare-like dialogue

## Project Structure

```
transformer-assignment/
├── src/
│   ├── model.py          # Transformer architecture implementation
│   ├── train.py          # Training script with modern techniques
│   ├── generate.py       # Text generation with multiple sampling strategies
│   └── utils.py          # Helper functions and analysis tools
├── ckpt.pt              # Trained model checkpoint
├── training_curves.png  # Training loss visualization
├── input.txt           # Tiny Shakespeare dataset
├── report.pdf          # Complete assignment report
└── README.md           # This file
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ALVAROI12/transformer-assignment-clean.git
cd transformer-assignment-clean

# Create virtual environment
python3 -m venv transformer_env
source transformer_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model (required - no pre-trained model included)
cd src
python train.py
```

## Important Note
The trained model checkpoints (ckpt.pt, ckpt_final.pt) are excluded due to GitHub size limits. The model must be trained from scratch using the provided training script.

### Training

```bash
cd src
python train.py
```

The training script will:
- Download Tiny Shakespeare dataset automatically
- Train for 5000 iterations (~10-15 minutes on GPU)
- Save best model to `ckpt.pt`
- Generate training curves plot

### Text Generation

```bash
# Generate with default settings
python generate.py --prompt "ROMEO:"

# Try different sampling strategies
python generate.py --prompt "JULIET:" --temperature 0.7 --top_k 20
python generate.py --prompt "To be or not to be" --temperature 0.9 --top_p 0.8
```

## Model Architecture

### Specifications
- **Layers**: 6 transformer blocks
- **Hidden Size**: 384 dimensions
- **Attention Heads**: 6 heads (64 dim each)
- **Sequence Length**: 256 tokens
- **Vocabulary**: 65 unique characters
- **Parameters**: 10.8 million

### Key Components
1. **Multi-Head Self-Attention**: Scaled dot-product attention with causal masking
2. **Position-wise Feed-Forward**: Two-layer MLP with ReLU activation
3. **Positional Encoding**: Learned positional embeddings
4. **Layer Normalization**: Pre-layer norm for training stability
5. **Residual Connections**: Skip connections for gradient flow

## Training Features

### Modern Training Techniques
- **AdamW Optimizer**: Better weight decay than Adam
- **Learning Rate Scheduling**: Warmup + cosine decay
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Faster training on modern GPUs
- **Automatic Checkpointing**: Saves best model during training

### Performance Metrics
- Training converged smoothly from loss ~4.2 to ~1.06
- Validation loss remained stable, indicating good generalization
- Generated text shows proper Shakespeare dialogue structure
- Model learned character names, vocabulary, and conversational patterns

## Sample Generated Text

**Prompt: "ROMEO:"**
```
ROMEO:
The gods that do resolve me not and leave.
ROMEO:
I will not be long in blood of the mother.
ROMEO:
And be not a thousand too; when I have made thee,
Nor I will be changed for thee with thy head,
```

**Prompt: "JULIET:"**
```
JULIET:
I do beseech you, my lord.
JULIET:
The lords of York, sir, sir, and there I may be
The heavens and fled these garden friends?
```

## System Analysis

### Hardware Configuration
- **CPU**: 16 logical cores (8 physical)
- **GPU**: NVIDIA RTX 4060 Laptop GPU (7.6GB VRAM)
- **Memory**: 14.9 GB RAM
- **CUDA**: Version 12.1

### Performance Analysis
- **Workload Type**: Compute-bound (not memory-bound)
- **GPU Utilization**: 85-90% during training
- **Memory Usage**: ~6-7GB out of 7.6GB available
- **Arithmetic Intensity**: 96 FLOPs/byte for attention layers

### Scaling Potential
- Current system can handle 4× larger models (43M parameters)
- Training efficiency could be improved with gradient checkpointing
- Mixed precision training already enabled for optimal performance

## Key Learnings

1. **Theoretical Understanding**: Deep dive into attention mechanisms, positional encoding, and transformer architecture
2. **Implementation Skills**: Building complex neural networks from scratch using PyTorch
3. **Training Best Practices**: Modern techniques for stable and efficient training
4. **System Optimization**: Understanding compute vs memory bottlenecks
5. **Text Generation**: Various sampling strategies (temperature, top-k, nucleus)

## Files Description

- **model.py**: Complete transformer implementation following GPT architecture
- **train.py**: Training loop with modern techniques (AdamW, LR scheduling, mixed precision)
- **generate.py**: Text generation with multiple sampling strategies
- **utils.py**: Helper functions for analysis, benchmarking, and visualization

## References

1. Vaswani et al., "Attention Is All You Need" (2017)
2. Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)
3. Andrej Karpathy's nanoGPT implementation
4. PyTorch official transformer tutorials

## Author

**Alvaro Ibarra**  
*Course: CPE-4953-001, EE-5453-005*  
*Assignment 1: Understanding & Implementing Transformers*

---

*This implementation demonstrates successful training of a transformer model from scratch, achieving high-quality text generation while maintaining efficient resource utilization on consumer hardware.*
