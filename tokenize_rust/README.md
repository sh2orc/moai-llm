# MOAI Ultra-Fast Rust Tokenizer

**Performance**: 40,000-60,000 samples/sec (20-30x faster than Python)

## Prerequisites

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
rustc --version
cargo --version
```

## Build

```bash
cd tokenize_rust
./build.sh
```

This creates: `target/release/moai-tokenizer`

## Usage

### Option 1: Python Wrapper (Recommended)

```bash
# From moai-llm directory
python tokenize_rust_wrapper.py \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --tokenizer_path tokenizers/moai \
    --max_samples 10000  # Test with 10K samples
```

### Option 2: Direct Rust Binary

```bash
# First convert dataset to Parquet manually
# Then run:
./target/release/moai-tokenizer \
    --input data.parquet \
    --output tokenized.parquet \
    --tokenizer ../tokenizers/moai/tokenizer.json \
    --max-length 0  # 0 = no truncation (packing mode)
```

## Integration with tokenize_datasets.py

The Rust tokenizer is automatically used by `tokenize_datasets.py` if available.

## Performance

| Dataset Size | Python (datasets.map) | Rust Tokenizer | Speedup |
|--------------|----------------------|----------------|---------|
| 100K samples | ~60 sec | ~2 sec | 30x |
| 1M samples | ~10 min | ~20 sec | 30x |
| 7.5M samples | ~75 min | ~3 min | 25x |

## Troubleshooting

### Build errors

```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

### Missing dependencies

```bash
# Ubuntu/Debian
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
xcode-select --install
```
