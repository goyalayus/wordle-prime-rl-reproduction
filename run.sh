#!/bin/bash
# LoRA SFT experiment: Qwen 0.6B, 400 steps, push to HF every 40 steps.
# Target: Lightning AI single T4 GPU.
#
# Prerequisites:
#   export HF_TOKEN=your_token
#   export HF_REPO_ID=yourusername/wordle-lora-qwen06b
#
# Usage:
#   bash run.sh

set -euo pipefail

if [ -f "$HOME/miniconda3/bin/activate" ]; then
  source "$HOME/miniconda3/bin/activate"
elif [ -f "/home/zeus/miniconda3/bin/activate" ]; then
  source "/home/zeus/miniconda3/bin/activate"
fi

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN"
  exit 1
fi

if [ -z "${HF_REPO_ID:-}" ]; then
  echo "ERROR: Set HF_REPO_ID (e.g. username/wordle-lora-qwen06b)"
  exit 1
fi

pip install -q -r requirements.txt
export TOKENIZERS_PARALLELISM=false

python train_sft_lora.py --wandb
