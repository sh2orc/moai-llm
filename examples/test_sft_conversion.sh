#!/bin/bash
# SFT 데이터 변환 테스트 스크립트

set -e

echo "=========================================="
echo "SFT Data Conversion Test"
echo "=========================================="
echo ""

# 출력 디렉토리 생성
mkdir -p data/sft_test

echo "1. Testing Alpaca format..."
python scripts/prepare_sft_data.py \
    --input_file examples/sample_sft_alpaca.json \
    --format alpaca \
    --template qwen \
    --output_file data/sft_test/alpaca_qwen.txt

echo "✓ Alpaca format converted"
echo ""

echo "2. Testing ShareGPT format..."
python scripts/prepare_sft_data.py \
    --input_file examples/sample_sft_sharegpt.json \
    --format sharegpt \
    --template qwen \
    --output_file data/sft_test/sharegpt_qwen.txt

echo "✓ ShareGPT format converted"
echo ""

echo "3. Testing Chat format..."
python scripts/prepare_sft_data.py \
    --input_file examples/sample_sft_chat.json \
    --format chat \
    --template qwen \
    --output_file data/sft_test/chat_qwen.txt

echo "✓ Chat format converted"
echo ""

echo "4. Testing auto-detect format..."
python scripts/prepare_sft_data.py \
    --input_file examples/sample_sft_alpaca.json \
    --format auto \
    --template qwen \
    --output_file data/sft_test/auto_detect.txt

echo "✓ Auto-detect worked"
echo ""

echo "5. Testing combined datasets..."
python scripts/prepare_sft_data.py \
    --input_files \
        examples/sample_sft_alpaca.json \
        examples/sample_sft_sharegpt.json \
        examples/sample_sft_chat.json \
    --format auto \
    --template qwen \
    --shuffle \
    --output_file data/sft_test/combined_all.txt

echo "✓ Combined datasets created"
echo ""

echo "6. Testing different templates..."

# Llama template
python scripts/prepare_sft_data.py \
    --input_file examples/sample_sft_alpaca.json \
    --template llama \
    --output_file data/sft_test/alpaca_llama.txt

# ChatML template
python scripts/prepare_sft_data.py \
    --input_file examples/sample_sft_alpaca.json \
    --template chatml \
    --output_file data/sft_test/alpaca_chatml.txt

# Simple template
python scripts/prepare_sft_data.py \
    --input_file examples/sample_sft_alpaca.json \
    --template simple \
    --output_file data/sft_test/alpaca_simple.txt

echo "✓ Multiple templates created"
echo ""

echo "=========================================="
echo "Results:"
echo "=========================================="
ls -lh data/sft_test/
echo ""

echo "=========================================="
echo "Sample Output (Qwen template):"
echo "=========================================="
head -n 20 data/sft_test/alpaca_qwen.txt
echo ""

echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo ""
echo "Check the files in data/sft_test/ to see the results."
