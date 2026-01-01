# MOAI-LLM: A 3B Parameter Language Model

A state-of-the-art 3B parameter language model based on Qwen3 architecture, featuring cutting-edge optimization techniques from 2024-2025 research.

## Features

### Architecture Innovations
- **Grouped Query Attention (GQA)**: 28 query heads, 4 key/value heads for efficient inference
- **PyTorch SDPA + Flash Attention**: Memory-efficient O(n) attention (no flash-attn install required!)
- **SwiGLU Activation**: Inlined forward pass for memory efficiency
- **Fused RMSNorm**: Uses flash-attn's CUDA kernel when available (~30-50% faster)
- **RoPE with YaRN**: Context extension up to 128K+ tokens
- **QK-Norm**: Training stability (Qwen3 feature)

### Training Optimizations
- **Fused AdamW Optimizer**: `adamw_torch_fused` for 2x faster optimizer step
- **Chunked Cross-Entropy**: Memory-efficient loss for 32GB GPUs (chunk_size=2048)
- **Multi-Objective Loss**: Combined cross-entropy, focal loss, and label smoothing
- **Warmup-Stable-Decay (WSD)**: Advanced learning rate schedule
- **Hierarchical Sequence Packing**: 90%+ GPU utilization
- **Mixed Precision (BF16)**: Efficient training on modern GPUs
- **Gradient Checkpointing**: Reduced memory footprint with `use_reentrant=False`

### Performance Optimizations (New!)
- **No dtype conversion**: logits stay in bf16 throughout forward pass
- **Fused RMSNorm**: CUDA kernel via flash-attn (falls back to PyTorch if unavailable)
- **Optimized reshapes**: `reshape()` instead of `contiguous().view()`
- **NCCL tuning**: Auto-detect GPU type for optimal P2P settings, extended timeouts
- **CUDA optimizations**: TF32, cuDNN v8 API, async kernel execution

### Memory Optimizations (New!)
- **Shared Memory Tokenization**: `TOKENIZERS_PARALLELISM=true` (Rust multi-threading)
- **Memory Mapping**: `IN_MEMORY_MAX_SIZE=0` forces disk-based Arrow files
- **Dataset Cache**: `load_from_cache_file=True` skips re-tokenization
- **Rust JSON**: `orjson` for 10-50x faster JSON parsing

### Tokenizer
- **HuggingFace Tokenizers (Rust)**: 10-50x faster than SentencePiece
- **Multilingual Support**: ko/en/ja/zh (64K base → 122K extended)
- **Domain Extended**: Finance vocabulary (BCCard, Alpaca-Korean)
- **Speed Modes**: `--fast`, `--turbo`, `--ultrafast` for large-scale training
- **Extensible**: Incrementally add domain-specific vocabulary

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | ~3B |
| Hidden Size | 3840 |
| Layers | 28 |
| Attention Heads (Q) | 28 |
| KV Heads | 4 |
| Vocabulary Size | **122K** (multilingual + finance) |
| Max Context Length | 8K → 128K+ (with YaRN) |
| FFN Intermediate Size | 10240 |
| Activation | SwiGLU |
| Normalization | RMSNorm (Pre-LN) |

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.5+
- CUDA 11.8+ (for GPU training)
- GPU Options:
  - **8× A40 48GB**: 2B model with vocab=92k (batch=12, effective=384) ⭐ Recommended
  - **4× A40 48GB**: 2B model with vocab=92k (batch=12, effective=192)
  - **4× RTX 5090 32GB**: 2B model with vocab=92k (batch=4, effective=64)
  - **4× RTX 4090 24GB**: 2B model with vocab=64k (batch=2, effective=32)
  - **8× A100 80GB**: Full 3B model (batch=24, effective=768)

### Install from Source

```bash
# Clone repository
git clone https://github.com/sh2orc/moai-llm.git
cd moai-llm

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# (Optional) Install Flash Attention 3
pip install flash-attn --no-build-isolation
```

## Quick Start

**10-Minute Test** to verify everything works:

```bash
# 1. Tokenizer (2 minutes) - Fast multilingual
python train_tokenizer.py \
    --multilingual ko en \
    --vocab_size 32000 \
    --max_samples_per_lang 5000 \
    --turbo \
    --output_dir tokenizers/test

# 2. Pretrain (3 minutes, 100 steps)
python train.py --mode pretrain --dataset wikitext --dataset_config wikitext-2-raw-v1 \
    --output_dir outputs/test --max_steps 100

# 3. SFT (2 minutes, 50 steps)
python train.py --mode sft --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/test/final_model --output_dir outputs/test-sft --max_steps 50

# 4. Chat (instant)
python chat.py --model_path outputs/test-sft/final_model

# (Optional) Test tokenizer
python test_tokenizer.py --tokenizer_path tokenizers/moai
python test_tokenizer.py --compare  # Compare all tokenizers
```

**Production Training with pretrain.sh** (Recommended):

```bash
# Multi-GPU pretraining with optimized settings
# Automatically configures batch size based on GPU memory

# 8× A40 48GB (recommended)
NUM_GPUS=8 GPU_MEMORY=48 ./pretrain.sh

# 4× RTX 5090 32GB
NUM_GPUS=4 GPU_MEMORY=32 ./pretrain.sh

# Custom gradient accumulation
NUM_GPUS=8 GPU_MEMORY=48 GRADIENT_ACCUMULATION_STEPS=8 ./pretrain.sh
```

**pretrain.sh Features**:
- Auto GPU detection (NCCL P2P for datacenter, disabled for consumer GPUs)
- Optimized CUDA settings (TF32, cuDNN v8, async kernels)
- Memory optimization (`PYTORCH_CUDA_ALLOC_CONF`)
- Sequential dataset processing with checkpoints
- Configurable via environment variables

**Production Training** (Manual - Using Pre-built Tokenizer):

```bash
# Tokenizer already built: tokenizers/moai (122K vocab)
# - Base: multilingual 64K (ko/en/ja/zh)
# - Extended: +alpaca-korean, +finance (BCCard)

# Step 1: Test tokenizer
python test_tokenizer.py --tokenizer_path tokenizers/moai

# Step 2: Pretrain
python train.py --mode pretrain \
    --dataset wikimedia/wikipedia --dataset_config 20231101.ko \
    --tokenizer_path tokenizers/moai \
    --output_dir outputs/pretrain \
    --bf16 --gradient_checkpointing

# Step 3: SFT
python train.py --mode sft \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/pretrain/final_model \
    --output_dir outputs/sft

# Step 4: Chat
python chat.py --model_path outputs/sft/final_model
```

**Build Your Own Tokenizer** (Optional):

```bash
# Step 1: Multilingual base (64K)
python train_tokenizer.py \
    --multilingual ko en ja zh \
    --vocab_size 64000 \
    --max_samples_per_lang 150000 \
    --turbo \
    --output_dir tokenizers/ \
    --model_prefix moai_multilingual

# Step 2: Extend with domain data (+16K → 80K)
python train_tokenizer.py \
    --base_tokenizer tokenizers/moai_multilingual \
    --dataset unoooo/alpaca-korean \
    --vocab_size 80000 \
    --turbo \
    --output_dir tokenizers/ \
    --model_prefix moai_alpaca

# Step 3: Add finance vocab (+16K → 96K)
python train_tokenizer.py \
    --base_tokenizer tokenizers/moai_alpaca \
    --dataset BCCard/BCCard-Finance-Kor-QnA \
    --vocab_size 96000 \
    --turbo \
    --output_dir tokenizers/ \
    --model_prefix moai_finance
```

### Tokenizer Speed Modes

| Mode | Algorithm | Speed | Use Case |
|------|-----------|-------|----------|
| Default | BPE | 1x | Small datasets |
| `--fast` | BPE optimized | 10x | Medium datasets |
| `--turbo` | BPE aggressive | 20x | Large datasets (recommended) |
| `--ultrafast` | Unigram | 50x | Maximum speed |

**For detailed guides, see:**
- `QUICKSTART.md` - Copy-paste ready commands
- `USER_GUIDE.md` - Complete training guide
- `DATASETS.md` - Dataset selection guide

## Training Recipes

### Recipe 1: Basic (Budget)
- **Tokens**: 20-30B
- **Hardware**: 1-2× A100 80GB
- **Duration**: 10-15 days
- **Cost**: ~$3-5K
- **Use case**: Proof of concept, experimentation

### Recipe 2: Intermediate (Production) ⭐
- **Tokens**: 60-100B
- **Hardware**: 8× A100 80GB
- **Duration**: 5-7 days
- **Cost**: ~$10-20K
- **Use case**: Competitive performance, recommended

### Recipe 3: Advanced (SOTA)
- **Tokens**: 100-300B
- **Hardware**: 16-32× H100
- **Duration**: 2-4 days
- **Cost**: ~$30-60K
- **Use case**: State-of-the-art results

### Recipe 4: Ultra-Efficient
- **Tokens**: 30B (highly curated)
- **Hardware**: 4× RTX 4090
- **Duration**: 15-20 days
- **Cost**: ~$2-3K
- **Use case**: Maximum cost efficiency

## Configuration

### Model Configuration (`configs/model_config.json`)

```json
{
  "vocab_size": 122100,
  "hidden_size": 3840,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0,
  "use_qk_norm": true
}
```

### Training Configuration (`configs/training_config.yaml`)

Key parameters:
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Gradient accumulation
- `learning_rate`: Peak learning rate (3e-4 recommended)
- `warmup_steps`: Warmup steps (2000 recommended)
- `bf16`: Mixed precision training (true recommended)
- `gradient_checkpointing`: Memory optimization

### Batch Size & Gradient Accumulation Guide

**Effective Batch Size Formula**:
```
Effective Batch = BATCH_SIZE × NUM_GPUS × GRADIENT_ACCUMULATION_STEPS
```

**Recommended Effective Batch Size by Model Size**:

| Model Size | Recommended Effective Batch | Notes |
|------------|----------------------------|-------|
| 1-3B | 256 ~ 512 | Stable training |
| 7B+ | 512 ~ 1024 | Large-scale training |
| Research | 128 ~ 256 | Fast iteration |

**Example Configurations (2B model, vocab=92k)**:

| GPU | BATCH_SIZE | GRAD_ACC | Effective Batch | Command |
|-----|------------|----------|-----------------|---------|
| **8× A100 80GB** | 24 | 4 | 768 | `NUM_GPUS=8 GPU_MEMORY=80 ./pretrain.sh` |
| **8× A40 48GB** | 12 | 4 | 384 | `NUM_GPUS=8 GPU_MEMORY=48 ./pretrain.sh` ⭐ |
| **4× A100 80GB** | 24 | 4 | 384 | `GPU_MEMORY=80 ./pretrain.sh` |
| **4× A40 48GB** | 12 | 8 | 384 | `GPU_MEMORY=48 ./pretrain.sh` |
| **4× RTX 5090 32GB** | 4 | 24 | 384 | `GPU_MEMORY=32 ./pretrain.sh` |

⚠️ **GPU Memory Limits**:
- 32GB: batch=4 (practical maximum with DDP overhead)
- 48GB: batch=12 (3× throughput vs 32GB)
- 80GB: batch=24 (6× throughput vs 32GB)

**Environment Variables**:
- `NUM_GPUS`: Number of GPUs (default: 4)
- `GPU_MEMORY`: GPU memory in GB (auto-selects batch size)
- `GRADIENT_ACCUMULATION_STEPS`: Override default accumulation

**Memory Usage (2B model, vocab=92k, bf16, seq=1024, DDP 4 GPUs)**:

| Component | Memory | Notes |
|-----------|--------|-------|
| Model weights | ~4 GB | bf16 |
| Optimizer (Fused AdamW) | ~8 GB | FP32 optimizer states |
| Gradients | ~4 GB | bf16 |
| DDP buffers & overhead | ~6-7 GB | Multi-GPU sync |
| **Fixed total** | **~22-23 GB** | |
| Activations (batch=4) | ~6-8 GB | With gradient checkpointing |
| Logits + Loss | ~2-3 GB | Direct bf16 cross-entropy |
| **Working total (batch=4)** | **~30-34 GB** | ⚠️ Near 32GB limit |

**Tips**:
- For **2B model + vocab=92k** on 32GB GPU: use `BATCH_SIZE=4` (practical limit)
- **PyTorch SDPA** provides Flash Attention-like efficiency without flash-attn installation
- **Fused AdamW** is faster than 8-bit Adam (prefer for 48GB+ GPUs)
- **Fused RMSNorm** (flash-attn): ~30-50% faster normalization
- DDP overhead is significant (~6-7GB) - single GPU may allow larger batches
- Direct bf16 cross-entropy: no dtype conversion overhead
- Always enable `--gradient_checkpointing` for memory savings

### Learning Rate Guide

**Recommended Learning Rates by Training Type**:

| Training Type | Learning Rate | Notes |
|--------------|---------------|-------|
| **Pretrain (from scratch)** | **1e-4 ~ 3e-4** | Standard for new models |
| Continued Pretrain | 5e-5 ~ 1e-4 | Lower than initial pretrain |
| Fine-tuning (full) | 1e-5 ~ 5e-5 | Careful not to forget |
| LoRA/QLoRA | 1e-4 ~ 2e-4 | Can be higher due to fewer params |
| SFT | 2e-5 ~ 1e-4 | Depends on dataset size |

**⚠️ Common Mistake**: Using `1e-6` for pretrain is **100x too low** - training will barely progress!

**Learning Rate by Model Size**:

| Model Size | Pretrain LR | Fine-tune LR |
|------------|-------------|--------------|
| 1-3B | 2e-4 ~ 3e-4 | 1e-5 ~ 5e-5 |
| 7B | 1e-4 ~ 2e-4 | 5e-6 ~ 2e-5 |
| 13B+ | 5e-5 ~ 1e-4 | 1e-6 ~ 1e-5 |

**Warmup Steps**:
- Typically 1-5% of total training steps
- 2000 steps is a good default for most cases

### Loss Configuration

```yaml
loss:
  type: "multi_objective"
  params:
    ce_weight: 0.6        # Cross-entropy weight
    focal_weight: 0.3     # Focal loss weight
    smooth_weight: 0.1    # Label smoothing weight
    focal_gamma: 2.0      # Focal loss gamma
    smoothing: 0.1        # Smoothing factor
```

## Advanced Features

### 1. Optimized Cross-Entropy Loss

MOAI-LLM uses **chunked cross-entropy** for memory efficiency on 32GB GPUs:

```python
# Chunked cross-entropy (memory-efficient, 32GB GPU compatible)
from moai_llm.losses import chunked_cross_entropy_loss
loss = chunked_cross_entropy_loss(
    shift_logits,
    shift_labels,
    chunk_size=2048,  # Balanced speed/memory
    ignore_index=-100,
)
```

**Memory Comparison (batch=4, seq=1024, vocab=92k)**:

| Method | Peak Memory | Speed | GPU Compatibility |
|--------|-------------|-------|-------------------|
| Standard F.cross_entropy | ~1.4 GB | Fastest | 48GB+ recommended |
| **Chunked (chunk=2048)** | **~0.3 GB** | **Fast** | **32GB compatible** ⭐ |
| Chunked (chunk=1024) | ~0.15 GB | Slower | 24GB compatible |

**Chunk Size Guide**:

| GPU Memory | Recommended chunk_size | Notes |
|------------|------------------------|-------|
| 24GB | 1024 | Conservative |
| 32GB | 2048 | Balanced ⭐ |
| 48GB+ | 4096 or direct CE | Maximum speed |

### 2. Memory-Efficient Data Processing

**Problem**: Large datasets (1M+ samples) can exhaust 300GB+ RAM during tokenization.

**Solution**: MOAI-LLM uses multiple strategies:

```python
# 1. Memory mapping (no full RAM load)
import datasets
datasets.config.IN_MEMORY_MAX_SIZE = 0  # Force disk-based Arrow

# 2. Cache reuse (skip re-tokenization)
dataset.map(..., load_from_cache_file=True)

# 3. Tokenizer-level parallelism (shared memory, not process duplication)
export TOKENIZERS_PARALLELISM=true  # Rust multi-threading
--num_proc 1  # Single process (no RAM duplication)
```

**RAM Usage Comparison (1.7M samples)**:

| Method | RAM Usage | Speed |
|--------|-----------|-------|
| `num_proc=8` + no cache | ~400 GB ❌ | Fast |
| `num_proc=4` + cache | ~100 GB | Medium |
| **`num_proc=1` + TOKENIZERS_PARALLELISM=true** | **~30 GB** ✅ | **Fast** |

### 3. NCCL Configuration for Multi-GPU

**Problem**: DDP training can hang with default NCCL timeouts.

**Solution**: Extended timeouts in `pretrain.sh`:

```bash
# NCCL timeout settings (handles slow gradient sync)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30 min (default: 480s)
export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_DEBUG=OFF  # Disable for performance

# GPU-specific P2P settings
# Datacenter GPUs (A40, A100, H100): P2P enabled
# Consumer GPUs (RTX): P2P disabled
```

### 4. Rust-Based Performance Packages

MOAI-LLM uses Rust packages for 10-50x faster operations:

| Package | Use | Speed vs Python |
|---------|-----|-----------------|
| `tokenizers` | Tokenization | ~10x faster |
| `orjson` | JSON parsing | 10-50x faster |
| `safetensors` | Model I/O | ~5x faster |
| `regex` | Pattern matching | ~2x faster |

```python
# Automatic fallback to Python if Rust packages unavailable
try:
    import orjson as json
except ImportError:
    import json
```

### 5. Context Extension with YaRN

```python
from moai_llm.config import MoaiConfig

config = MoaiConfig(
    max_position_embeddings=131072,  # 128K context
    rope_scaling={
        "type": "yarn",
        "factor": 4.0,  # Extension factor (32K → 128K)
        "original_max_position_embeddings": 32768,
        "alpha": 1.0,
        "beta": 32.0,
    }
)
```

### 6. Sequence Packing

```python
from moai_llm.data import HierarchicalBalancePacker

packer = HierarchicalBalancePacker(
    max_seq_length=8192,
    num_bins=8,
)

packed_sequences = packer.pack(sequences)
# Achieves 90%+ GPU utilization
```

### 7. Custom Loss Functions

```python
from moai_llm.losses import create_loss_function

# Focal loss for hard examples
loss_fn = create_loss_function({
    "type": "focal",
    "params": {"gamma": 2.0}
})

# Multi-objective
loss_fn = create_loss_function({
    "type": "multi_objective",
    "params": {
        "ce_weight": 0.6,
        "focal_weight": 0.3,
        "smooth_weight": 0.1
    }
})
```

## Architecture Details

### Attention Mechanism (GQA + SDPA)
- **Query Heads**: 28 (one per layer)
- **Key/Value Heads**: 4 (shared across 7 query heads each)
- **Head Dimension**: 137 (3840 / 28)
- **KV Cache Reduction**: 7× smaller than Multi-Head Attention
- **Memory-Efficient Attention**: O(N) memory complexity vs O(N²)

**Attention Backend Priority**:

| Priority | Backend | Requirements | Memory |
|----------|---------|--------------|--------|
| 1 | Flash Attention 2/3 | `flash-attn` package | O(N) |
| 2 | **PyTorch SDPA** | PyTorch 2.0+ (built-in) | O(N) |
| 3 | Standard Attention | Fallback | O(N²) |

**SDPA Benefits** (no extra installation required!):
- **Memory-efficient attention**: O(N) instead of O(N²)
- Enables **batch_size=4** on 32GB GPUs (vs batch_size=2 without)
- Automatically uses optimal kernel (Flash/Efficient/Math)
- Works on all GPUs including consumer RTX cards

### Normalization (Fused RMSNorm)
- **Type**: RMSNorm (Root Mean Square Layer Normalization)
- **Backend Priority**:
  1. **Fused CUDA kernel** (flash-attn): ~30-50% faster
  2. **PyTorch fallback**: Works on all devices
- **Optimization**: No float32 conversion (stays in bf16)
- **Auto-detection**: Uses fused kernel when flash-attn is installed

```python
# Automatically uses best backend
from moai_llm.modeling.normalization import MoaiRMSNorm
norm = MoaiRMSNorm(hidden_size=3840)
# Prints: "backend=fused" or "backend=pytorch"
```

### Feed-Forward Network (SwiGLU)
- **Structure**: Gate-Up-Down projection (inlined forward)
- **Intermediate Size**: 10240 (~2.67× hidden size)
- **Activation**: SwiGLU (Swish + GLU)
- **Optimization**: Single-line forward to minimize memory allocation
- **Parameters per layer**: ~147M

### Position Encoding (RoPE + YaRN)
- **Base Frequency**: 1,000,000 (Qwen3 standard)
- **Scaling**: YaRN for efficient context extension
- **Context Length**: 32K (base) → 128K+ (extended)
- **Training Efficiency**: <0.1% of original pretraining data for extension

## Benchmarks

Performance on common benchmarks (after full pre-training):

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU | TBD | 5-shot |
| HellaSwag | TBD | 10-shot |
| ARC-Challenge | TBD | 25-shot |
| TruthfulQA | TBD | 0-shot |
| GSM8K | TBD | 8-shot CoT |

*Benchmarks will be updated after training completion*

## Project Structure

```
moai-llm/
├── moai_llm/                   # Main package
│   ├── config.py              # Model configuration
│   ├── modeling/              # Model implementation
│   │   ├── attention.py       # GQA + SDPA/Flash Attention
│   │   ├── activations.py     # SwiGLU (inlined forward)
│   │   ├── normalization.py   # Fused RMSNorm (flash-attn backend)
│   │   ├── rope.py            # RoPE + YaRN
│   │   ├── transformer.py     # Decoder layer
│   │   └── model.py           # Full model (optimized loss)
│   ├── losses.py              # Loss functions (chunked CE available)
│   ├── data/                  # Data utilities
│   └── tokenizer/             # Tokenizer utilities
├── train_tokenizer.py          # Tokenizer training script
├── test_tokenizer.py           # Tokenizer testing & comparison
├── train.py                    # Unified training (pretrain + SFT)
├── pretrain.sh                 # ⭐ Multi-GPU pretrain script (recommended)
├── chat.py                     # Interactive chat interface
├── test_inference.py           # Inference testing
├── check_dataset.py            # Dataset info tool
├── configs/                    # Configuration files
│   ├── model_config.json      # Model config
│   ├── model_config_2b.json   # 2B model config
│   └── training_config.yaml   # Training config
├── examples/                   # Example use cases
│   └── bccard_example.md      # BCCard dataset example
├── QUICKSTART.md              # Quick start guide
├── USER_GUIDE.md              # Complete training guide
├── DATASETS.md                # Dataset guide
├── ARCHITECTURE.md            # Architecture details
└── README.md                  # This file
```

## Technical References

### Architecture
- Qwen3: https://arxiv.org/abs/2505.09388
- Qwen2.5: https://arxiv.org/abs/2412.15115
- GQA: https://arxiv.org/abs/2305.13245
- Flash Attention 3: https://github.com/togethercomputer/flash-attention-3

### Position Encoding
- RoPE: https://arxiv.org/abs/2104.09864
- YaRN: https://arxiv.org/abs/2309.00071
- LongRoPE: https://arxiv.org/abs/2402.13753

### Activation & Normalization
- GLU Variants: https://arxiv.org/abs/2002.05202
- RMSNorm: https://arxiv.org/abs/1910.07467

### Training Techniques
- Warmup-Stable-Decay: https://arxiv.org/abs/2410.05192
- Focal Loss: https://arxiv.org/abs/1708.02002
- Label Smoothing: https://arxiv.org/abs/1512.00567
- Hierarchical Packing: https://arxiv.org/abs/2503.07680

### Tokenization
- HuggingFace Tokenizers: https://github.com/huggingface/tokenizers
- SentencePiece: https://github.com/google/sentencepiece
- BPE-Dropout: https://arxiv.org/abs/1910.13267

## Citation

If you use MOAI-LLM in your research, please cite:

```bibtex
@software{moai-llm,
  title = {MOAI-LLM: A 3B Parameter Language Model with State-of-the-Art Optimizations},
  author = {MOAI Team},
  year = {2025},
  url = {https://github.com/sh2orc/moai-llm}
}
```

## License

Apache License 2.0

## Acknowledgments

This project builds upon research and implementations from:
- Qwen Team (Alibaba Cloud)
- HuggingFace Transformers
- Dao-AILab (Flash Attention)
- Google (SentencePiece)
- EleutherAI (scaling research)

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions and feedback:
- GitHub Issues: https://github.com/sh2orc/moai-llm/issues

