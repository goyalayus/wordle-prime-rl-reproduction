#!/bin/bash
# Run this ON Lightning (after SSH) to watch logs every 30 seconds.
# Usage: ./watch_logs_on_lightning.sh

INTERVAL=30

echo "Watching logs every ${INTERVAL}s. Press Ctrl+C to stop."
echo ""

while true; do
  clear
  echo "========== $(date) =========="
  echo ""
  echo "--- train_1.7b.log (last 20) ---"
  tail -20 train_1.7b.log 2>/dev/null || echo "No log yet"
  echo ""
  echo "--- train_0.6b.log (last 20) ---"
  tail -20 train_0.6b.log 2>/dev/null || echo "No log yet"
  sleep "$INTERVAL"
done
