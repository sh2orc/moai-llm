#!/bin/bash
# MOAI-LLM Pretrain Script (Multi-Dataset)
# 
# 데이터셋:
# - sh2orc/bccard-maywell-jojo0217-markai-lcw99-kendamarron-microsoft (instruction/output)
# - BCCard/BCAI-Finance-Kor-1862K (instruction/output)
# - HAERAE-HUB/KOREAN-WEBTEXT (text)
#
# Usage: ./pretrain.sh [config_size]
# Example: ./pretrain.sh 2b

set -e

# ============================================================================
# Configuration
# ============================================================================

CONFIG_SIZE=${1:-2b}
NUM_GPUS=${NUM_GPUS:-4}

# Model config based on size
case $CONFIG_SIZE in
    2b)
        MODEL_CONFIG="configs/model_config_2b.json"
        BATCH_SIZE=16  # vocab_size=128000이라 logits가 크므로 배치 줄임
        ;;
    5b)
        MODEL_CONFIG="configs/model_config.json"
        BATCH_SIZE=4
        ;;
    *)
        echo "Unknown config size: $CONFIG_SIZE (use 2b or 5b)"
        exit 1
        ;;
esac

# Common settings
TOKENIZER_PATH="tokenizers/moai"
GRADIENT_ACCUMULATION_STEPS=6  # 배치 줄인만큼 증가 (effective batch = 4*4*12 = 192)
MAX_SEQ_LENGTH=1024
LEARNING_RATE=1e-4
WARMUP_STEPS=2000
NUM_EPOCHS=2

# Output directory
OUTPUT_DIR="outputs/pretrain-korean-instruction-${CONFIG_SIZE}"

# ============================================================================
# Dataset Configuration
# ============================================================================

# 여러 HuggingFace 데이터셋 사용 (콤마 없이!)
# config가 필요한 경우: "dataset_name:config_name" 형식 사용
DATASETS=(
    "sh2orc/bccard-maywell-jojo0217-markai-lcw99-kendamarron-microsoft"
    "izumi-lab/llm-japanese-dataset"
    "saldra/sakura_japanese_dataset"
    "fujiki/japanese_alpaca_data"
    "BCCard/BCAI-Finance-Kor-1862K"
    "HAERAE-HUB/KOREAN-WEBTEXT"
)

# 데이터셋을 공백으로 구분된 문자열로 변환
DATASET_ARGS=""
for ds in "${DATASETS[@]}"; do
    DATASET_ARGS="$DATASET_ARGS $ds"
done

# ============================================================================
# Environment Setup
# ============================================================================

# CUDA settings
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
export TOKENIZERS_PARALLELISM=false

# NCCL settings (P2P 비활성화 - 혼합 GPU 환경이나 P2P 오류 시 필요) RTX 4090, 5090 
export NCCL_P2P_DISABLE=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 추가 속도 최적화
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# TF32 활성화 (Ampere+ GPU, ~2x matmul 속도)
export NVIDIA_TF32_OVERRIDE=1

# HuggingFace cache (optional)
# export HF_HOME="/path/to/cache"
# export HF_DATASETS_CACHE="/path/to/datasets/cache"

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================================================"
echo "🚀 MOAI-LLM Pretrain Script (Multi-Dataset)"
echo "========================================================================"
echo "Model Config:          $MODEL_CONFIG"
echo "Config Size:           $CONFIG_SIZE"
echo "Tokenizer:             $TOKENIZER_PATH"
echo "Output:                $OUTPUT_DIR"
echo "========================================================================"
echo "Datasets:"
for ds in "${DATASETS[@]}"; do
    echo "  - $ds"
done
echo "========================================================================"
echo "GPUs:                  $NUM_GPUS"
echo "Batch Size (per GPU):  $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective Batch Size:  $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "Max Seq Length:        $MAX_SEQ_LENGTH"
echo "Learning Rate:         $LEARNING_RATE"
echo "Warmup Steps:          $WARMUP_STEPS"
echo "Epochs:                $NUM_EPOCHS"
echo "Packing:               Enabled"
echo "========================================================================"

# ============================================================================
# Run Training
# ============================================================================

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --mode pretrain \
    --dataset $DATASET_ARGS \
    --tokenizer_path $TOKENIZER_PATH \
    --model_config $MODEL_CONFIG \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --num_epochs $NUM_EPOCHS \
    --bf16 \
    --gradient_checkpointing \
    --packing \
    --sequential \
    --flash_attention \
    --num_proc 8 \
    --dataloader_num_workers 4 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3

echo "========================================================================"
echo "✅ Pretrain completed!"
echo "📁 Model saved to: $OUTPUT_DIR/final_model"
echo "========================================================================"

