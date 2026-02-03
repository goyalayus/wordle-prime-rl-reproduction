#!/bin/bash
# Run this ON the Lightning.ai Studio (after SSH), where the GPU and LoRA weights are.
# From your local terminal: ssh s_XXXX@ssh.lightning.ai
# Then on the Studio:
#   cd wordle-prime-rl-reproduction
#   source /home/zeus/miniconda3/bin/activate
#   bash run_play_wordle_on_lightning.sh

set -e
cd "$(dirname "$0")"
echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"
python3 play_wordle.py --words plane store trace price crane
