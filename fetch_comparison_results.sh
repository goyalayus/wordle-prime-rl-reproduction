#!/bin/bash
# Fetch comparison JSON results from Lightning Studio.
# Replace LIGHTNING_SSH with your actual SSH host (e.g. s_XXXX@ssh.lightning.ai)
# Usage: bash fetch_comparison_results.sh

set -e

LIGHTNING_SSH="${LIGHTNING_SSH:-s_01kgh1r0496amc9yj1r2r7snk8@ssh.lightning.ai}"
REMOTE_DIR="wordle-prime-rl-reproduction"

FILES=(
    "results_primeintellect_1p7b.json"
    "results_qwen_1p7b.json"
    "results_qwen_0p6b.json"
)

echo "Fetching comparison results from Lightning..."
for f in "${FILES[@]}"; do
    echo "  - $f"
    scp "${LIGHTNING_SSH}:~/${REMOTE_DIR}/${f}" ./ 2>/dev/null || echo "    (file not found, run compare on Lightning first)"
done

echo ""
echo "Done. Open viewer.html in your browser to compare."
