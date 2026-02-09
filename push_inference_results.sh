#!/bin/bash
# Push inference_results.json to GitHub (inference branch).
# Run this AFTER run_inference.py completes.
#
# Usage:
#   bash push_inference_results.sh

set -euo pipefail

if [ ! -f inference_results.json ]; then
  echo "ERROR: inference_results.json not found. Run inference first."
  exit 1
fi

git add inference_results.json
git status
git commit -m "Add inference results: 10 LoRA checkpoints x 3 Wordle games"
git push origin inference
echo "Pushed to origin/inference. Pull locally with: git pull origin inference"
