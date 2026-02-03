#!/bin/bash
# Run full fine-tuning on A100
#
# Usage:
#   bash run_sft_a100.sh

set -e

# Install deps
pip install -q -r requirements.txt
pip install -q flash-attn --no-build-isolation

# Set environment
export WANDB_PROJECT="wordle-sft"
export TOKENIZERS_PARALLELISM=false

# Run
python train_sft_full.py

echo "Done! Model saved to outputs/wordle_sft_full/final"
