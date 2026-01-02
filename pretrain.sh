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
GPU_MEMORY=${GPU_MEMORY:-32}  # GPU memory in GB (32, 48, 80)

# Model config based on size and GPU memory
case $CONFIG_SIZE in
    2b)
        MODEL_CONFIG="configs/model_config_2b.json"
        case $GPU_MEMORY in
            32)
                BATCH_SIZE=4   # RTX 5090 32GB
                GRADIENT_ACCUMULATION_STEPS=24  # effective = 4*4*24 = 384
                ;;
            48)
                BATCH_SIZE=12  # A40 48GB
                GRADIENT_ACCUMULATION_STEPS=4   # effective = 12*8*4 = 384 (8 GPUs)
                ;;
            80)
                BATCH_SIZE=24  # A100 80GB
                GRADIENT_ACCUMULATION_STEPS=4   # effective = 24*4*4 = 384
                ;;
            *)
                BATCH_SIZE=4
                GRADIENT_ACCUMULATION_STEPS=24
                ;;
        esac
        ;;
    5b)
        MODEL_CONFIG="configs/model_config.json"
        BATCH_SIZE=1
        GRADIENT_ACCUMULATION_STEPS=96  # effective = 1*4*96 = 384
        ;;
    *)
        echo "Unknown config size: $CONFIG_SIZE (use 2b or 5b)"
        exit 1
        ;;
esac

# Common settings
TOKENIZER_PATH="tokenizers/moai"
MAX_SEQ_LENGTH=1024
LEARNING_RATE=1e-4
WARMUP_STEPS=2000
NUM_EPOCHS=2

# Output directory
OUTPUT_DIR="outputs/moai-${CONFIG_SIZE}"

# ============================================================================
# Dataset Configuration
# ============================================================================

# 여러 HuggingFace 데이터셋 사용 (콤마 없이!)
# config가 필요한 경우: "dataset_name:config_name" 형식 사용
DATASETS=(
    "BCCard/BCCard-Finance-Kor-QnA"
    "sh2orc/bccard-maywell-jojo0217-markai-lcw99-kendamarron-microsoft"
    "nvidia/Nemotron-CC-Math-v1:3"
    "nvidia/OpenCodeGeneticInstruct"
    "BCCard/BCAI-Finance-Kor-1862K"
    "HAERAE-HUB/KOREAN-WEBTEXT"
)

# 데이터셋 배열은 그대로 사용 (문자열 변환 불필요)

# ============================================================================
# Environment Setup
# ============================================================================

# CUDA settings - Generate GPU list based on NUM_GPUS
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_LIST=""
    for ((i=0; i<NUM_GPUS; i++)); do
        if [ $i -eq 0 ]; then
            GPU_LIST="$i"
        else
            GPU_LIST="$GPU_LIST,$i"
        fi
    done
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi
# Use tokenizers' internal Rust parallelism (shared memory, no process duplication)
export TOKENIZERS_PARALLELISM=true

# NCCL settings
# P2P 비활성화: RTX 계열 + A40 (A40은 P2P 이슈 있음)
# A100, H100 등은 P2P 지원
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" == *"RTX"* ]] || [[ "$GPU_NAME" == *"GeForce"* ]] || [[ "$GPU_NAME" == *"A40"* ]]; then
    echo "⚠️  P2P disabled for: $GPU_NAME"
    export NCCL_P2P_DISABLE=1
else
    echo "✓ P2P enabled for: $GPU_NAME"
    export NCCL_P2P_DISABLE=0
fi

# NCCL 추가 최적화
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 비동기 에러 처리
export NCCL_IB_DISABLE=1            # InfiniBand 비활성화 (PCIe 환경)
export NCCL_NET_GDR_LEVEL=0         # GPU Direct RDMA 비활성화 (호환성)

# NCCL Timeout 설정 (chunked CE + gradient checkpointing으로 인한 느린 연산 대응)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30분 (기본 480초)
export NCCL_TIMEOUT=1800                       # 30분
export TORCH_DISTRIBUTED_DEBUG=OFF             # 디버그 비활성화 (성능)

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# CUDA 최적화
export CUDA_LAUNCH_BLOCKING=0       # 비동기 커널 실행
export TORCH_CUDNN_V8_API_ENABLED=1 # cuDNN v8 API

# CPU 최적화
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# TF32 활성화 (Ampere+ GPU, ~2x matmul 속도)
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

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
echo "GPU Memory:            ${GPU_MEMORY}GB"
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

# Debug: print command
echo "DEBUG: Running command:"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py \\"
echo "  --mode pretrain \\"
echo "  --dataset ${DATASETS[*]} \\"
echo "  --tokenizer_path $TOKENIZER_PATH \\"
echo "  --model_config $MODEL_CONFIG \\"
echo "  --output_dir $OUTPUT_DIR \\"
echo "  ..."

# Find available port
MASTER_PORT=${MASTER_PORT:-29500}
while lsof -Pi :$MASTER_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; do
    MASTER_PORT=$((MASTER_PORT + 1))
done
echo "Using master port: $MASTER_PORT"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train.py \
    --mode pretrain \
    --dataset "${DATASETS[@]}" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --model_config "$MODEL_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_steps "$WARMUP_STEPS" \
    --num_epochs "$NUM_EPOCHS" \
    --bf16 \
    --gradient_checkpointing \
    --packing \
    --sequential \
    --flash_attention \
    --num_proc 4 \
    --dataloader_num_workers 8 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3

echo "========================================================================"
echo "✅ Pretrain completed!"
echo "📁 Model saved to: $OUTPUT_DIR/final_model"
echo "========================================================================"

