#!/bin/bash
# Build MOAI Rust Tokenizer

set -e

echo "🔨 Building MOAI Rust Tokenizer..."
echo "========================================"

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not installed!"
    echo ""
    echo "Install Rust:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo ""
    exit 1
fi

echo "✓ Rust: $(rustc --version)"
echo "✓ Cargo: $(cargo --version)"
echo ""

# Build
echo "📦 Building release binary..."
cargo build --release

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Build successful!"
    echo "   Binary: target/release/moai-tokenizer"
    echo "========================================"
    echo ""
    echo "Test it:"
    echo "  python ../tokenize_rust_wrapper.py \\"
    echo "    --dataset wikitext \\"
    echo "    --dataset_config wikitext-2-raw-v1 \\"
    echo "    --tokenizer_path ../tokenizers/moai \\"
    echo "    --max_samples 1000"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi
