# MOAI-LLM: A 3B Parameter Language Model

A state-of-the-art 3B parameter language model based on Qwen3 architecture, featuring cutting-edge optimization techniques from 2024-2025 research.

## Features

### Architecture Innovations
- **Grouped Query Attention (GQA)**: 28 query heads, 4 key/value heads for efficient inference
- **Flash Attention 3**: Memory-efficient attention with INT8 quantization support
- **SwiGLU Activation**: Superior performance compared to standard FFN
- **RMSNorm**: Efficient pre-normalization for stable training
- **RoPE with YaRN**: Context extension up to 128K+ tokens
- **QK-Norm**: Training stability (Qwen3 feature)

### Training Optimizations
- **Multi-Objective Loss**: Combined cross-entropy, focal loss, and label smoothing
- **Warmup-Stable-Decay (WSD)**: Advanced learning rate schedule
- **Hierarchical Sequence Packing**: 90%+ GPU utilization
- **Mixed Precision (BF16)**: Efficient training on modern GPUs
- **Gradient Checkpointing**: Reduced memory footprint

### Tokenizer
- **SentencePiece BPE**: 128K vocabulary optimized for multilingual text
- **CJK Optimization**: Special focus on Korean, Chinese, Japanese
- **BPE-Dropout**: Regularization for robust tokenization

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | ~3B |
| Hidden Size | 3840 |
| Layers | 28 |
| Attention Heads (Q) | 28 |
| KV Heads | 4 |
| Vocabulary Size | 128K |
| Max Context Length | 8K → 128K+ (with YaRN) |
| FFN Intermediate Size | 10240 |
| Activation | SwiGLU |
| Normalization | RMSNorm (Pre-LN) |

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.5+
- CUDA 11.8+ (for GPU training)
- 80GB+ GPU memory (recommended: 8× A100 80GB)

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
# 1. Tokenizer (2 minutes)
python train_tokenizer.py --dataset wikipedia --dataset_config 20220301.ko \
    --vocab_size 32000 --max_samples 10000 --output_dir tokenizers/test

# 2. Pretrain (3 minutes, 100 steps)
python train.py --mode pretrain --dataset wikitext --dataset_config wikitext-2-raw-v1 \
    --output_dir outputs/test --max_steps 100

# 3. SFT (2 minutes, 50 steps)
python train.py --mode sft --dataset BCCard/BCCard-Finance-Kor-QnA \
    --pretrained_model outputs/test/final_model --output_dir outputs/test-sft --max_steps 50

# 4. Chat (instant)
python chat.py --model_path outputs/test-sft/final_model
```

**Production Training**:

```bash
# Full workflow (tokenizer → pretrain → SFT)
python train_tokenizer.py --dataset wikipedia --dataset_config 20220301.ko --output_dir tokenizers/ko
python train.py --mode pretrain --dataset wikipedia --dataset_config 20220301.ko --output_dir outputs/pretrain --bf16
python train.py --mode sft --dataset BCCard/BCCard-Finance-Kor-QnA --pretrained_model outputs/pretrain/final_model --output_dir outputs/sft
python chat.py --model_path outputs/sft/final_model
```

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
  "vocab_size": 128000,
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

### 1. Context Extension with YaRN

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

### 2. Sequence Packing

```python
from moai_llm.data import HierarchicalBalancePacker

packer = HierarchicalBalancePacker(
    max_seq_length=8192,
    num_bins=8,
)

packed_sequences = packer.pack(sequences)
# Achieves 90%+ GPU utilization
```

### 3. Custom Loss Functions

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

### Attention Mechanism (GQA)
- **Query Heads**: 28 (one per layer)
- **Key/Value Heads**: 4 (shared across 7 query heads each)
- **Head Dimension**: 137 (3840 / 28)
- **KV Cache Reduction**: 7× smaller than Multi-Head Attention
- **Flash Attention**: O(N) memory complexity vs O(N²)

### Feed-Forward Network (SwiGLU)
- **Structure**: Gate-Up-Down projection
- **Intermediate Size**: 10240 (~2.67× hidden size)
- **Activation**: SwiGLU (Swish + GLU)
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
│   │   ├── attention.py       # GQA + Flash Attention
│   │   ├── activations.py     # SwiGLU
│   │   ├── normalization.py   # RMSNorm
│   │   ├── rope.py            # RoPE + YaRN
│   │   ├── transformer.py     # Decoder layer
│   │   └── model.py           # Full model
│   ├── losses.py              # Loss functions
│   ├── data/                  # Data utilities
│   └── tokenizer/             # Tokenizer utilities
├── train_tokenizer.py          # Tokenizer training script
├── train.py                    # Unified training (pretrain + SFT)
├── chat.py                     # Interactive chat interface
├── test_inference.py           # Inference testing
├── check_dataset.py            # Dataset info tool
├── configs/                    # Configuration files
│   ├── model_config.json      # Model config
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

---

**Built with ❤️ for the open-source AI community**
