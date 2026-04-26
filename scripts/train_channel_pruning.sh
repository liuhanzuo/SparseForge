#!/bin/bash
# =============================================================================
# Channel-level Structured Pruning Training Script
# =============================================================================
# This script trains a channel-pruned model using HASAST-style mask dynamics
# combined with SlimLLM-style importance scoring.
#
# Usage:
#   bash scripts/train_channel_pruning.sh [OPTIONS]
#
# Options:
#   --model MODEL       Model name (default: Qwen/Qwen3-1.7B)
#   --keep_ratio RATIO  FFN keep ratio (default: 0.75)
#   --dataset DATASET   Dataset name (default: c4_qwen)
#   --iters ITERS       Max iterations (default: 10000)
#   --mode MODE         single/ddp (default: single)
# =============================================================================

set -e

# Default values
MODEL="Qwen/Qwen3-1.7B"
MODEL_TYPE="qwen"
KEEP_RATIO=0.75
DATASET="c4_qwen"
MAX_ITERS=10000
BATCH_SIZE=4
GRAD_ACCUM=8
LEARNING_RATE=1e-4
MODE="single"
WANDB_LOG=false
OUT_DIR="outputs/channel_pruning"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --keep_ratio)
            KEEP_RATIO="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --iters)
            MAX_ITERS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --wandb)
            WANDB_LOG=true
            shift
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create output directory
mkdir -p "$OUT_DIR"

# Generate run name
KEEP_PCT=$(echo "$KEEP_RATIO * 100" | bc | cut -d'.' -f1)
MODEL_SHORT=$(echo "$MODEL" | sed 's/.*\///')
RUN_NAME="${MODEL_SHORT}_ffn${KEEP_PCT}pct_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "  Channel-level Structured Pruning Training"
echo "=============================================="
echo "Model:          $MODEL"
echo "Model Type:     $MODEL_TYPE"
echo "FFN Keep Ratio: $KEEP_RATIO (${KEEP_PCT}%)"
echo "Dataset:        $DATASET"
echo "Max Iters:      $MAX_ITERS"
echo "Batch Size:     $BATCH_SIZE x $GRAD_ACCUM (grad accum)"
echo "Learning Rate:  $LEARNING_RATE"
echo "Mode:           $MODE"
echo "Output:         $OUT_DIR"
echo "Run Name:       $RUN_NAME"
echo "=============================================="

# Build command
CMD="python -m channel_pruning.train_channel_pruning \
    --model_name $MODEL \
    --model_type $MODEL_TYPE \
    --ffn_keep_ratio $KEEP_RATIO \
    --dataset $DATASET \
    --max_iters $MAX_ITERS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --out_dir $OUT_DIR \
    --use_distillation \
    --export_pruned"

if [ "$WANDB_LOG" = true ]; then
    CMD="$CMD --wandb_log --wandb_project channel-pruning"
fi

# Run based on mode
cd "$PROJECT_DIR"

if [ "$MODE" = "single" ]; then
    echo ""
    echo "Running single-GPU training..."
    $CMD
    
elif [ "$MODE" = "ddp" ]; then
    # Count GPUs
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo ""
    echo "Running DDP training on $NUM_GPUS GPUs..."
    
    torchrun --standalone --nproc_per_node=$NUM_GPUS \
        -m channel_pruning.train_channel_pruning \
        --model_name $MODEL \
        --model_type $MODEL_TYPE \
        --ffn_keep_ratio $KEEP_RATIO \
        --dataset $DATASET \
        --max_iters $MAX_ITERS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LEARNING_RATE \
        --out_dir $OUT_DIR \
        --use_distillation \
        --export_pruned \
        --ddp \
        $([ "$WANDB_LOG" = true ] && echo "--wandb_log --wandb_project channel-pruning")
else
    echo "Unknown mode: $MODE"
    exit 1
fi

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
echo "Output saved to: $OUT_DIR"
echo "Pruned model at: $OUT_DIR/pruned_model"
