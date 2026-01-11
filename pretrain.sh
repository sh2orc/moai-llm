#!/bin/bash
# MOAI-LLM Mamba Pretrain Script (Multi-Dataset)
#
# Pure Mamba SSM Architecture Training
#
# Datasets:
# - sh2orc/bccard-maywell-jojo0217-markai-lcw99-kendamarron-microsoft (instruction/output)
# - BCCard/BCAI-Finance-Kor-1862K (instruction/output)
# - HAERAE-HUB/KOREAN-WEBTEXT (text)
#
# Usage: ./pretrain.sh [config_size]
# Example:
#   ./pretrain.sh 2b                              # Use tensorboard (default)
#   USE_WANDB=true ./pretrain.sh 2b               # Use W&B with default project
#   USE_WANDB=true WANDB_PROJECT=my-project ./pretrain.sh 2b  # Custom W&B project

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
        MODEL_CONFIG="configs/mamba_config_2b.json"
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
    8b)
        MODEL_CONFIG="configs/mamba_config_8b.json"
        case $GPU_MEMORY in
            48)
                BATCH_SIZE=4   # A40 48GB
                GRADIENT_ACCUMULATION_STEPS=16  # effective = 4*8*16 = 512
                ;;
            80)
                BATCH_SIZE=8   # A100 80GB
                GRADIENT_ACCUMULATION_STEPS=8   # effective = 8*8*8 = 512
                ;;
            *)
                echo "8B model requires at least 48GB GPU memory"
                exit 1
                ;;
        esac
        ;;
    16b)
        MODEL_CONFIG="configs/mamba_config_16b.json"
        case $GPU_MEMORY in
            48)
                BATCH_SIZE=2   # A40 48GB
                GRADIENT_ACCUMULATION_STEPS=32  # effective = 2*8*32 = 512
                ;;
            80)
                BATCH_SIZE=4   # A100 80GB
                GRADIENT_ACCUMULATION_STEPS=16  # effective = 4*8*16 = 512
                ;;
            *)
                echo "16B model requires at least 48GB GPU memory"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Unknown config size: $CONFIG_SIZE (use 2b, 8b or 16b)"
        exit 1
        ;;
esac

# Common settings
TOKENIZER_PATH="tokenizers/moai"
MAX_SEQ_LENGTH=2048  # Mamba supports longer sequences natively
LEARNING_RATE=1e-4
WARMUP_STEPS=2000
NUM_EPOCHS=2

# Output directory
OUTPUT_DIR="outputs/mamba-${CONFIG_SIZE}"

# Logging settings (W&B or Tensorboard)
USE_WANDB=${USE_WANDB:-false}  # Set to "true" to use W&B
WANDB_PROJECT=${WANDB_PROJECT:-"moai-mamba"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"pretrain-${CONFIG_SIZE}-$(date +%Y%m%d-%H%M%S)"}

# Mamba-specific settings
USE_QUANTIZATION=${USE_QUANTIZATION:-false}  # Set to "true" for 4-bit quantization
QUANT_BITS=${QUANT_BITS:-4}

# ============================================================================
# Dataset Configuration
# ============================================================================

DATASETS=(
    "BCCard/BCCard-Finance-Kor-QnA"
    "sh2orc/bccard-maywell-jojo0217-markai-lcw99-kendamarron-microsoft"
    "nvidia/OpenCodeGeneticInstruct:qwen2.5-32b-instruct"
    "BCCard/BCAI-Finance-Kor-1862K"
    "HAERAE-HUB/KOREAN-WEBTEXT"
)

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

# NCCL settings - GPU type detection
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$GPU_NAME" == *"RTX"* ]] || [[ "$GPU_NAME" == *"GeForce"* ]] || [[ "$GPU_NAME" == *"A40"* ]]; then
    echo "P2P disabled for: $GPU_NAME"
    export NCCL_P2P_DISABLE=1
else
    echo "P2P enabled for: $GPU_NAME"
    export NCCL_P2P_DISABLE=0
fi

# NCCL optimization
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_DEBUG=OFF

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# CUDA optimization
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Python optimization
export PYTHONUNBUFFERED=1

# HuggingFace optimization
export HF_DATASETS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================================================"
echo "MOAI-LLM Mamba Pretrain Script (Pure SSM Architecture)"
echo "========================================================================"
echo "Model Config:          $MODEL_CONFIG"
echo "Config Size:           $CONFIG_SIZE"
echo "Architecture:          Pure Mamba SSM (O(L) complexity)"
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
echo "Quantization:          $USE_QUANTIZATION ($QUANT_BITS-bit)"
echo "Logging:               $([ "$USE_WANDB" = "true" ] && echo "W&B ($WANDB_PROJECT)" || echo "Tensorboard")"
echo "========================================================================"

# ============================================================================
# Step 1: Tokenize Datasets (Single Process)
# ============================================================================

echo "========================================================================"
echo "Step 1: Tokenizing datasets (single process)..."
echo "========================================================================"

python tokenize_datasets.py \
    --dataset "${DATASETS[@]}" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --packing

if [ $? -ne 0 ]; then
    echo "❌ Tokenization failed!"
    exit 1
fi

echo ""
echo "✅ Tokenization completed!"
echo ""

# ============================================================================
# Step 2: Run Training (Multi-GPU with torchrun)
# ============================================================================

echo "========================================================================"
echo "Step 2: Training (multi-GPU with Mamba SSM)..."
echo "========================================================================"

# Find available port
MASTER_PORT=${MASTER_PORT:-29500}
while lsof -Pi :$MASTER_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; do
    MASTER_PORT=$((MASTER_PORT + 1))
done
echo "Using master port: $MASTER_PORT"

# Build quantization arguments
QUANT_ARGS=""
if [ "$USE_QUANTIZATION" = "true" ]; then
    QUANT_ARGS="--quantize --bits $QUANT_BITS"
fi

# Build wandb arguments conditionally
WANDB_ARGS=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_ARGS="--use_wandb --wandb_project $WANDB_PROJECT --wandb_run_name $WANDB_RUN_NAME"
fi

# Use Mamba training script
python scripts/train_mamba.py \
    --config_path "$MODEL_CONFIG" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --data_path "data/tokenized" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max_length "$MAX_SEQ_LENGTH" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_steps "$WARMUP_STEPS" \
    --num_epochs "$NUM_EPOCHS" \
    --gradient_checkpointing \
    $QUANT_ARGS \
    $WANDB_ARGS

echo "========================================================================"
echo "Mamba Pretrain completed!"
echo "Model saved to: $OUTPUT_DIR/final"
echo "========================================================================"
echo ""
echo "To test the model:"
echo "  python scripts/test_mamba.py --config $CONFIG_SIZE"
echo ""
echo "To generate text:"
echo "  python scripts/generate_mamba.py --model_path $OUTPUT_DIR/final --chat"
echo "========================================================================"
