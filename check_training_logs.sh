#!/bin/bash
# Poll Lightning training logs every 30 seconds.
# Usage: ./check_training_logs.sh [ssh_target]
# Example: ./check_training_logs.sh s_01kgh1r0496amc9yj1r2r7snk8@ssh.lightning.ai

SSH_TARGET="${1:-s_01kgh1r0496amc9yj1r2r7snk8@ssh.lightning.ai}"
REPO_PATH="/home/zeus/content/wordle-prime-rl-reproduction"
INTERVAL=30

echo "Checking logs every ${INTERVAL}s from ${SSH_TARGET}"
echo "Press Ctrl+C to stop"
echo ""

while true; do
  echo "========== $(date) =========="
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$SSH_TARGET" \
    "cd $REPO_PATH 2>/dev/null && echo '--- train_1.7b.log (last 15) ---' && tail -15 train_1.7b.log 2>/dev/null || echo 'No log' && echo '' && echo '--- train_0.6b.log (last 15) ---' && tail -15 train_0.6b.log 2>/dev/null || echo 'No log'" 2>/dev/null || echo "SSH failed"
  echo ""
  sleep "$INTERVAL"
done
