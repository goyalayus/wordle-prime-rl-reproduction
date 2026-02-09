#!/bin/bash
# Run inference on 10 LoRA checkpoints. Use on T4 GPU.
#
# Prerequisites:
#   - HF_TOKEN in .env or env (for goyalayus/wordle-lora-qwen06b)
#   - pip install -r requirements-inference.txt
#
# Usage:
#   bash run_inference.sh

set -euo pipefail

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python run_inference.py
