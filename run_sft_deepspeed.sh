#!/bin/bash
# Run full fine-tuning with DeepSpeed ZeRO-3 on 4x T4 GPUs
# 
# Prerequisites:
#   pip install -r requirements.txt
#   wandb login  (optional, for logging)
#
# Usage:
#   bash run_sft_deepspeed.sh

set -e

# Check GPU count
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 4 ]; then
    echo "Warning: Expected 4 GPUs, found $NUM_GPUS. Adjusting..."
fi

# Set environment variables
export WANDB_PROJECT="wordle-sft"
export TOKENIZERS_PARALLELISM=false

# Run with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS train_sft_deepspeed.py

echo "Training complete! Model saved to outputs/wordle_sft_full/final"
