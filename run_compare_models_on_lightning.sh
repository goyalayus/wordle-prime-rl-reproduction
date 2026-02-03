#!/bin/bash
# Run this ON Lightning.ai Studio (after SSH).
# From local: ssh s_XXXX@ssh.lightning.ai
# Then on Studio:
#   cd wordle-prime-rl-reproduction
#   source /home/zeus/miniconda3/bin/activate
#   bash run_compare_models_on_lightning.sh
#
# After it finishes, pull the JSON files locally and open viewer.html.

set -e
cd "$(dirname "$0")"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"
echo "Running 3-model comparison (no truncation)..."
echo ""

python3 compare_models.py --output-dir . --max-tokens 32768

echo ""
echo "Done. Pull JSON files to your machine and open viewer.html to compare."
