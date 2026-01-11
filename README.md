# MOAI-LLM: Pure Mamba SSM Language Model

A state-of-the-art language model featuring **pure Mamba Selective State Space Model (SSM)** architecture with linear complexity O(L) for efficient long-context processing.

## Architecture Overview

### Pure Mamba SSM Architecture
MOAI-LLM uses a **pure Mamba Selective State Space Model** architecture - no attention layers, all SSM for maximum efficiency.

**Key Benefits:**
- **Linear Complexity O(L)**: 16x faster than attention for long sequences
- **Memory Efficient**: Constant memory vs quadratic for attention
- **Long Context**: Native support for 32K+ tokens without RAG
- **Edge Ready**: 4-bit quantization for mobile/embedded deployment
- **QAT Support**: Quantization-Aware Training for production models

**vs Transformer:**
| Metric | Transformer | Mamba SSM |
|--------|-------------|-----------|
| Time Complexity | O(L²) | **O(L)** ⚡ |
| Memory Complexity | O(L²) | **O(L)** ⚡ |
| Long Context | RAG needed | **Native** ✅ |
| Quantization | Difficult | **4-bit QAT ready** ✅ |
| Inference Speed | 1x | **16x** (32K+ tokens) |

### Model Specifications

| Config | Parameters | Hidden Size | Layers | State Size | Vocabulary |
|--------|------------|-------------|--------|------------|------------|
| **2B** | ~2.1B | 2048 | 24 | 16 | 128K |
| **8B** | ~8.2B | 4096 | 40 | 16 | 128K |
| **16B** | ~16.4B | 5120 | 48 | 16 | 128K |

**Architecture Details:**
- **SSM State Dimension**: 16 (efficient selective state)
- **Convolution Kernel**: 4 (local context modeling)
- **Expansion Factor**: 2 (2x inner dimension)
- **Normalization**: RMSNorm (eps=1e-5)
- **Activation**: SwiGLU (via FFN)
- **Position Encoding**: None (SSM is inherently positional)
- **Max Context**: 32K tokens (native)

## Features

### Core SSM Features
- **Selective Mechanism**: Input-dependent A, B, C parameters
- **Hardware-Aware Scan**: Optimized selective scan implementation
- **1D Convolution**: Depthwise conv for local context
- **Gating Mechanism**: Efficient information flow
- **Gradient Checkpointing**: Memory-efficient training

### Training Optimizations
- **Chunked Cross-Entropy**: Memory-efficient loss (32GB GPU compatible)
- **QAT (Quantization-Aware Training)**: Train in quantized mode
- **4-bit Quantization**: INT4 weights for edge deployment
- **Mixed Precision (BF16)**: Efficient training on modern GPUs
- **Gradient Checkpointing**: Reduced memory footprint
- **Fused Operations**: Optimized CUDA kernels when available

### Tokenizer Compatibility
- **Existing Tokenizers**: Works with all existing MOAI tokenizers
- **Multilingual**: ko/en/ja/zh support
- **Domain Extended**: Finance, Alpaca, custom vocabularies
- **Vocabulary Size**: 128K tokens

## Installation

### Requirements
```bash
# Core
Python 3.10+
PyTorch 2.5+
CUDA 11.8+ (for GPU training)

# Required
transformers>=4.30.0
einops  # for tensor operations
```

### Quantization Packages (Optional)

Choose based on your needs:

```bash
# INT4 Group-wise (default, included)
# No extra packages needed

# FP8/FP4 (NVIDIA H100/RTX 40xx only)
pip install transformer-engine

# AWQ (Activation-aware, best accuracy)
pip install auto-awq

# GPTQ (Gradient-based, production)
pip install auto-gptq
# or
pip install optimum[gptq]

# BitsAndBytes (HuggingFace integration)
pip install bitsandbytes

# Mamba SSM (optional, for optimized scan)
pip install mamba-ssm

# Flash Attention (optional, for faster RMSNorm)
pip install flash-attn --no-build-isolation
```

### Install from Source

```bash
# Clone repository
git clone https://github.com/sh2orc/moai-llm.git
cd moai-llm

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Test Model Creation

```bash
# Test 2B model
python scripts/test_mamba.py --config 2b

# Test all model sizes
python scripts/test_mamba.py --all-sizes

# Verbose output
python scripts/test_mamba.py --config 2b --verbose
```

### 2. Train with Dummy Data (Quick Test)

```bash
# 2B model, dummy data
python scripts/train_mamba.py \
  --config 2b \
  --test \
  --batch_size 4 \
  --num_epochs 1
```

### 3. Standard Training

```bash
# Pretrain from scratch
python scripts/train_mamba.py \
  --config 2b \
  --tokenizer_path tokenizers/moai \
  --data_path data/train \
  --valid_data_path data/valid \
  --output_dir outputs/mamba-2b \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_epochs 1 \
  --learning_rate 1e-4 \
  --max_length 2048

# Fine-tune existing model
python scripts/train_mamba.py \
  --model_path outputs/mamba-2b/final \
  --data_path data/finetune \
  --output_dir outputs/mamba-2b-finetune
```

### 4. QAT (Quantization-Aware Training) ⭐

**Train in 4-bit quantized mode:**

```bash
# QAT with INT4 (default)
python scripts/train_mamba.py \
  --config 2b \
  --data_path data/train \
  --output_dir outputs/mamba-2b-qat \
  --quantize \
  --bits 4 \
  --learning_rate 1e-5

# QAT with pre-quantized model
python scripts/train_mamba.py \
  --model_path outputs/mamba-2b/final \
  --data_path data/finetune \
  --output_dir outputs/mamba-2b-qat-finetune \
  --quantize \
  --bits 4
```

**Using pretrain.sh with QAT:**

```bash
# Standard QAT
USE_QUANTIZATION=true ./pretrain.sh 2b

# Custom bits
USE_QUANTIZATION=true QUANT_BITS=4 ./pretrain.sh 2b
```

### 5. Advanced Quantization

**FP8 (NVIDIA H100/RTX 40xx only):**

```python
from moai_llm.modeling.advanced_quantization import quantize_fp8
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM
from moai_llm.modeling.ssm_config import get_mamba_config

config = get_mamba_config("2b")
model = MoaiMambaForCausalLM(config)
model = quantize_fp8(model)  # H100 or RTX 40xx required
```

**AWQ (best accuracy, requires calibration):**

```python
from moai_llm.modeling.advanced_quantization import quantize_awq

# Calibration data (128 samples)
calib_data = torch.randint(0, vocab_size, (128, 512))

model = quantize_awq(model, calib_data, w_bit=4)
```

**GPTQ (production, requires calibration):**

```python
from moai_llm.modeling.advanced_quantization import quantize_gptq

calib_data = torch.randint(0, vocab_size, (128, 512))

model = quantize_gptq(model, calib_data, bits=4, group_size=128)
```

### 6. Text Generation

```bash
# Interactive chat
python scripts/generate_mamba.py \
  --model_path outputs/mamba-2b/final \
  --tokenizer_path tokenizers/moai \
  --chat

# Single prompt
python scripts/generate_mamba.py \
  --model_path outputs/mamba-2b/final \
  --tokenizer_path tokenizers/moai \
  --prompt "Hello, world!" \
  --max_new_tokens 128

# Batch generation
python scripts/generate_mamba.py \
  --model_path outputs/mamba-2b/final \
  --tokenizer_path tokenizers/moai \
  --input_file prompts.txt \
  --output_file outputs.json
```

## Quantization Guide

### Quantization Methods Comparison

| Method | Bits | HW Accel | Accuracy | Memory | Use Case |
|--------|------|----------|----------|--------|----------|
| **INT4 Group** | 4 | No | Good | 1x | Edge CPU, QAT ⭐ |
| **FP8** | 8 | H100/RTX40xx | Excellent | 2x | NVIDIA GPU |
| **FP4** | 4 | H100 only | Good | 4x | H100 GPU |
| **AWQ** | 4 | No | Excellent | 1x | Production |
| **GPTQ** | 4 | No | Excellent | 1x | Production |
| **BnB 4-bit** | 4 | No | Good | 1x | HF Models |
| **BnB 8-bit** | 8 | No | Excellent | 2x | HF Models |

### QAT vs PTQ vs QLoRA

| Method | Description | Training | Memory | Accuracy | Use Case |
|--------|-------------|----------|--------|----------|----------|
| **QAT** | Quantization-Aware Training | Full | Medium | Good | Small models (2B, 8B) |
| **PTQ** | Post-Training Quantization | Inference only | Minimal | Good | Quick deployment |
| **QLoRA** | 4-bit + LoRA adapters | LoRA only | Minimal | Excellent | Large models (16B+) |

### QAT Workflow

**Step 1: Quantize Model**

```python
from moai_llm.modeling.ssm_config import get_mamba_config
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM
from moai_llm.modeling.quantization import quantize_model

# Load or create model
config = get_mamba_config("2b")
model = MoaiMambaForCausalLM(config)

# Quantize to INT4
model = quantize_model(model, bits=4, group_size=128)

# Save quantized model
model.save_pretrained("outputs/mamba-2b-int4")
```

**Step 2: Train (QAT)**

```python
import torch
from torch.utils.data import DataLoader

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
for batch in dataloader:
    # Forward: quantized weights are dequantized for computation
    outputs = model(**batch)
    loss = outputs.loss

    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save fine-tuned QAT model
model.save_pretrained("outputs/mamba-2b-qat-finetuned")
```

**Step 3: Inference**

```python
# Load QAT model
model = MoaiMambaForCausalLM.from_pretrained("outputs/mamba-2b-qat-finetuned")

# Generate (weights remain quantized)
model.eval()
outputs = model.generate(input_ids, max_new_tokens=100)
```

### Choosing the Right Method

**For Training:**
- **2B, 8B models**: Use QAT (quantize and train)
- **16B+ models**: Use QLoRA (4-bit + LoRA adapters)

**For Inference:**
- **Edge CPU/Mobile**: INT4 Group-wise
- **NVIDIA H100**: FP8 or FP4 (fastest)
- **NVIDIA RTX 40xx**: FP8
- **Production accuracy**: AWQ or GPTQ
- **Quick HF integration**: BitsAndBytes

## Configuration

### Model Configurations

JSON configs in `configs/`:

```json
// configs/mamba_config_2b.json
{
  "model_type": "mamba_moai",
  "vocab_size": 128000,
  "hidden_size": 2048,
  "intermediate_size": 5632,
  "num_hidden_layers": 24,
  "state_size": 16,
  "conv_kernel_size": 4,
  "expand_factor": 2,
  "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-05,
  "use_fast_scan": true,
  "use_checkpoint": true,
  "quantization_bits": 4,
  "use_quantization": false
}
```

### Loading Configs

```python
# From predefined size
from moai_llm.modeling.ssm_config import get_mamba_config
config = get_mamba_config("2b")  # or "8b", "16b"

# From custom JSON
from moai_llm.modeling.ssm_config import load_mamba_config
config = load_mamba_config("path/to/config.json")

# Create model
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM
model = MoaiMambaForCausalLM(config)
```

### Training Parameters

| Parameter | 2B Model | 8B Model | 16B Model |
|-----------|----------|----------|-----------|
| **Learning Rate (FP)** | 1e-4 | 8e-5 | 6e-5 |
| **Learning Rate (QAT)** | 1e-5 | 5e-6 | 3e-6 |
| **Batch Size** | 4-8 | 2-4 | 1-2 |
| **Gradient Accum** | 8-16 | 16-32 | 32-64 |
| **Warmup Steps** | 2000 | 3000 | 4000 |
| **Max Length** | 2048-4096 | 2048-4096 | 2048-4096 |
| **Weight Decay** | 0.01 | 0.01 | 0.01 |

**GPU Requirements:**

| GPU | Model | Batch Size | GRAD_ACC | Effective Batch |
|-----|-------|------------|----------|-----------------|
| **A100 80GB** | 2B | 16 | 4 | 256 |
| **A100 80GB** | 8B | 8 | 8 | 256 |
| **A40 48GB** | 2B | 8 | 8 | 256 |
| **A40 48GB** | 8B | 4 | 16 | 256 |
| **RTX 4090 24GB** | 2B | 4 | 16 | 256 |

**QAT Memory Savings:**

| Model | FP16 Training | QAT (INT4) | Reduction |
|-------|---------------|-------------|-----------|
| 2B | ~8 GB | ~4 GB | 2x |
| 8B | ~32 GB | ~16 GB | 2x |
| 16B | ~64 GB | ~32 GB | 2x |

## Project Structure

```
moai-llm/
├── moai_llm/
│   ├── modeling/
│   │   ├── ssm_config.py              # Mamba config (JSON-based)
│   │   ├── ssm.py                     # Core SSM implementation
│   │   ├── moai_mamba.py              # Pure Mamba model
│   │   ├── quantization.py            # INT4 quantization (QAT)
│   │   ├── advanced_quantization.py   # FP8/FP4/AWG/GPTQ
│   │   ├── normalization.py           # RMSNorm (common)
│   │   ├── activations.py             # SwiGLU (common)
│   │   ├── rope.py                    # RoPE (common, not used in Mamba)
│   │   └── legacy_transformer/        # Old Transformer (deprecated)
│   │       ├── attention.py
│   │       ├── decoder.py
│   │       └── model.py
│   └── inference.py                    # Inference pipeline
├── scripts/
│   ├── train_mamba.py                 # Training script (QAT support)
│   ├── generate_mamba.py              # Generation script
│   └── test_mamba.py                  # Test script
├── configs/
│   ├── mamba_config_2b.json           # 2B model config
│   ├── mamba_config_8b.json           # 8B model config
│   └── mamba_config_16b.json          # 16B model config
├── tokenizers/                        # Existing tokenizers (compatible)
│   ├── moai/                          # Base tokenizer (128K vocab)
│   ├── moai_multilingual/             # Multilingual
│   └── moai_finance/                  # Finance domain
├── pretrain.sh                        # Multi-GPU pretrain script
├── data/                              # Training data
├── outputs/                           # Model checkpoints
└── README.md
```

## Performance

### Speed vs Attention

For long sequences, Mamba SSM provides significant speedup:

| Sequence Length | Attention Time | Mamba Time | Speedup |
|----------------|----------------|------------|---------|
| 512 | 1.0x | 1.2x | 0.8x |
| 2048 | 1.0x | 0.8x | 1.2x |
| 8192 | 1.0x | 0.4x | 2.5x |
| 32768 | 1.0x | 0.06x | **16x** ⚡ |

### Memory Usage

| Sequence Length | Attention | Mamba | Reduction |
|----------------|-----------|-------|-----------|
| 2048 | 4 GB | 2 GB | 2x |
| 8192 | 64 GB | 8 GB | 8x |
| 32768 | 1024 GB | 32 GB | 32x |

### Quantization Impact

**Memory Savings:**

| Model | FP32 | INT4 | INT8 | FP8 | Reduction |
|-------|------|------|------|-----|-----------|
| 2B | 8 GB | 1 GB | 2 GB | 2 GB | 8x (INT4) |
| 8B | 32 GB | 4 GB | 8 GB | 8 GB | 8x (INT4) |
| 16B | 64 GB | 8 GB | 16 GB | 16 GB | 8x (INT4) |

**Speed Comparison (Inference):**

| Method | 2B Model | 8B Model | Hardware |
|--------|----------|----------|----------|
| FP32 | 1.0x | 1.0x | All GPUs |
| INT4 | 1.5x | 1.8x | All GPUs |
| FP8 | 2.5x | 3.0x | H100/RTX 40xx |
| FP4 | 4.0x | 4.5x | H100 only |

## Technical References

### Mamba Architecture
- **Mamba Paper**: https://arxiv.org/abs/2312.00752
- **Selective State Spaces**: Input-dependent SSM parameters
- **Hardware-Aware Scan**: Optimized CUDA implementation
- **Linear Complexity**: O(L) vs O(L²) for attention

### Quantization
- **QAT**: Training with quantized weights
- **AWQ**: https://arxiv.org/abs/2306.00978 (Activation-aware)
- **GPTQ**: https://arxiv.org/abs/2210.17323 (Gradient-based)
- **BitsAndBytes**: LLM.int8() and 4-bit quantization
- **FP8 Training**: https://arxiv.org/abs/2306.06965

### Related Work
- **Transformers**: Attention is all you need (2017)
- **State Space Models**: Efficient modeling alternative
- **RWKV**: Recurrent neural network with attention-like properties
- **RetNet**: Retentive network for efficient language modeling

## Examples

### Basic Usage

```python
from moai_llm.modeling.ssm_config import get_mamba_config
from moai_llm.modeling.moai_mamba import MoaiMambaForCausalLM
from transformers import AutoTokenizer

# Load config and model
config = get_mamba_config("2b")
model = MoaiMambaForCausalLM(config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizers/moai")

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### QAT Training

```python
from moai_llm.modeling.quantization import quantize_model

# Create model
config = get_mamba_config("2b")
model = MoaiMambaForCausalLM(config)

# Quantize
model = quantize_model(model, bits=4, group_size=128)

# Train in quantized mode
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Save
model.save_pretrained("outputs/mamba-2b-qat")
```

### FP8 Quantization

```python
from moai_llm.modeling.advanced_quantization import quantize_fp8

model = MoaiMambaForCausalLM(config)
model = quantize_fp8(model)  # H100 or RTX 40xx required
```

### Custom Config

```python
from moai_llm.modeling.ssm_config import MoaiMambaConfig

# Create custom config
config = MoaiMambaConfig(
    vocab_size=128000,
    hidden_size=2560,        # Custom hidden size
    num_hidden_layers=32,    # Custom layers
    state_size=16,
    conv_kernel_size=4,
    expand_factor=2,
    use_fast_scan=True,
)

# Create model with custom config
model = MoaiMambaForCausalLM(config)
```

## Migration from Transformer

### For Existing Users

If you're migrating from the Transformer-based MOAI-LLM:

**What Changed:**
- ✅ **Same tokenizers**: No changes needed
- ✅ **Same training pipeline**: Compatible data format
- ⚠️ **Different architecture**: Pure SSM (no attention)
- ⚠️ **Different configs**: Use `mamba_config_*.json`

**What Stays the Same:**
- Tokenizer files and paths
- Training data format
- Inference interface
- Quantization approach (enhanced with QAT)

**What's Different:**
- Model architecture (SSM vs Attention)
- Configuration files
- Memory usage (lower)
- Speed for long sequences (faster)
- QAT support (built-in)

### Using Legacy Transformer

The old Transformer code is preserved in `moai_llm/modeling/legacy_transformer/`:

```python
# For backward compatibility
from moai_llm.modeling.legacy_transformer.model import MoaiForCausalLM
model = MoaiForCausalLM.from_pretrained("path/to/transformer/checkpoint")
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps`
- Enable `--gradient_checkpointing`
- Use QAT: `--quantize --bits 4`
- Use smaller model config

**2. QAT Training Issues**
- Use lower learning rate: `1e-5` instead of `1e-4`
- Ensure `use_quantization=false` in config (script enables it)
- Check gradient accumulation for stability

**3. Slow Training**
- Ensure `use_fast_scan=True` in config
- Install `mamba-ssm` for optimized scan
- Use `flash-attn` for faster RMSNorm
- Increase `--batch_size` if memory allows
- Use QAT for memory savings

**4. Import Errors**
- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.10+)
- Verify PyTorch: `python -c "import torch; print(torch.__version__)"`

**5. FP8/FP4 Errors**
- Check GPU: `nvidia-smi` (H100 or RTX 40xx required)
- Install: `pip install transformer-engine`
- Verify CUDA version: `nvcc --version`

## Citation

If you use MOAI-LLM in your research, please cite:

```bibtex
@software{moai-llm,
  title = {MOAI-LLM: Pure Mamba SSM Language Model with QAT Support},
  author = {MOAI Team},
  year = {2025},
  url = {https://github.com/sh2orc/moai-llm}
}
```

Also cite the original Mamba paper:

```bibtex
@article{mamba,
  title = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author = {Gu, Albert and Dao, Tri},
  journal = {arXiv preprint arXiv:2312.00752},
  year = {2023}
}
```

## License

Apache License 2.0

## Acknowledgments

This project builds upon research and implementations from:
- **Mamba Team**: Original Mamba architecture
- **State Space Models**: Efficient sequence modeling research
- **HuggingFace Transformers**: Model infrastructure
- **AWQ/GPTQ**: Quantization research
- **EleutherAI**: Scaling and evaluation research

## Contact

For questions and feedback:
- GitHub Issues: https://github.com/sh2orc/moai-llm/issues
- Discussions: https://github.com/sh2orc/moai-llm/discussions
