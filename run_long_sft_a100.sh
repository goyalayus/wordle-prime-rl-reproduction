#!/bin/bash
# Long SFT run for Qwen/Qwen3-0.6B with:
# - effective/global batch size = 60
# - validation loss logging every 20 optimizer steps
# - gameplay eval (5 secret words) every 20 optimizer steps
#
# Intended for a single A100 (e.g., Lightning.ai Studio via SSH).
#
# Usage:
#   bash run_long_sft_a100.sh
#
# Notes:
# - Adjust PER_DEVICE_BS to maximize GPU utilization; keep GLOBAL_BS divisible.
# - Ensure W&B is authenticated (WANDB_API_KEY in .env or `wandb login`).

set -euo pipefail

# Lightning images commonly ship with conda but may not have `pip` on PATH.
# Activate conda if available.
if [ -f "$HOME/miniconda3/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/bin/activate"
elif [ -f "/home/zeus/miniconda3/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "/home/zeus/miniconda3/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install -q -r requirements.txt

# Optional performance dependency; install only when INSTALL_FLASH_ATTENTION is non-empty.
if [ -n "${INSTALL_FLASH_ATTENTION:-}" ]; then
  "$PYTHON_BIN" -m pip install -q flash-attn --no-build-isolation || true
else
  echo "Skipping flash-attn install (set INSTALL_FLASH_ATTENTION=1 to enable)" >&2
fi

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-wordle-sft}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-1024}"
LR="${LR:-1e-5}"

# Effective/global batch size target (optimizer-step batch size in examples)
GLOBAL_BS="${GLOBAL_BS:-60}"

# Tune this to use more A100 memory; GLOBAL_BS must be divisible by PER_DEVICE_BS.
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"

# Eval every N optimizer steps (no eval at step 0)
EVAL_EVERY="${EVAL_EVERY:-20}"

# Save checkpoints less frequently than evals to limit disk usage
SAVE_EVERY="${SAVE_EVERY:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

VAL_SIZE="${VAL_SIZE:-100}"

SECRET_WORDS="${SECRET_WORDS:-crane,alert,amaze,plane,store}"
GAMEPLAY_MAX_NEW_TOKENS="${GAMEPLAY_MAX_NEW_TOKENS:-8192}"

WANDB_NAME="${WANDB_NAME:-qwen3-0p6b__long-sft__$(date +%Y%m%d_%H%M%S)}"

"$PYTHON_BIN" train_sft_full.py \
  --model "$MODEL" \
  --output-dir "outputs/wordle_sft_long" \
  --dataset "willcb/V3-wordle" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --lr "$LR" \
  --global-batch-size "$GLOBAL_BS" \
  --per-device-batch-size "$PER_DEVICE_BS" \
  --val-size "$VAL_SIZE" \
  --eval-every-steps "$EVAL_EVERY" \
  --save-every-steps "$SAVE_EVERY" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name "$WANDB_NAME" \
  --gameplay-eval \
  --gameplay-eval-every-steps "$EVAL_EVERY" \
  --gameplay-secret-words "$SECRET_WORDS" \
  --gameplay-max-new-tokens "$GAMEPLAY_MAX_NEW_TOKENS"

echo ""
echo "Done."
echo "To view evals:"
echo "  $PYTHON_BIN -m http.server 8000"
echo "  open: http://localhost:8000/viewer_long_run.html?root=outputs/wordle_sft_long/$($PYTHON_BIN -c \"print('$MODEL'.replace('/', '-'))\")/evals"
