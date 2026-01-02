import time
import sys

print("Testing import speed...", flush=True)

start = time.time()
import torch
print(f"torch: {time.time() - start:.1f}s", flush=True)

start = time.time()
from transformers import AutoTokenizer
print(f"transformers: {time.time() - start:.1f}s", flush=True)

start = time.time()
import datasets
print(f"datasets: {time.time() - start:.1f}s", flush=True)

print(f"Total: Done!", flush=True)
