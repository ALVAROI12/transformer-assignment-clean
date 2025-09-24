from utils import get_system_info, print_model_info
from model import GPTTransformer
import torch

print('=== SYSTEM SPECIFICATIONS ===')
info = get_system_info()
for key, value in info.items():
    print(f'{key}: {value}')

print('\n=== MODEL ANALYSIS ===')
model = GPTTransformer(vocab_size=65, d_model=384, n_heads=6, n_layers=6, max_seq_len=256)
print_model_info(model, (64, 256))

print('\n=== PERFORMANCE ANALYSIS ===')
# Quick benchmark
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
from utils import benchmark_model
benchmark_model(model, (64, 256), device, num_runs=50)
