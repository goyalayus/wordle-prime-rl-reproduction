#!/bin/bash
# Run SFT training and evaluation on multiple Qwen3 models
# Usage: bash run_comparison.sh
set -e

echo "=========================================="
echo "Wordle SFT Multi-Model Comparison"
echo "=========================================="

# Common training args (matching PrimeIntellect config)
COMMON_ARGS="--max-steps 20 --seq-len 1024 --learning-rate 1e-5 --per-device-batch-size 1 --gradient-accumulation-steps 64 --warmup-steps 5 --logging-steps 1 --gradient-checkpointing"

# ==========================================
# Step 1: Train Qwen/Qwen3-1.7B (official)
# ==========================================
echo ""
echo "[1/4] Training Qwen/Qwen3-1.7B (official)..."
echo "=========================================="
if [ -d "outputs/qwen3_1p7b_official" ]; then
    echo "Model already exists, skipping training..."
else
    python train_sft_trl_qwen3_1p7b.py \
        --model Qwen/Qwen3-1.7B \
        --output-dir outputs/qwen3_1p7b_official \
        --wandb-project wordle-comparison \
        --wandb-name qwen3-1p7b-official \
        $COMMON_ARGS
fi

# ==========================================
# Step 2: Train Qwen/Qwen3-0.6B
# ==========================================
echo ""
echo "[2/4] Training Qwen/Qwen3-0.6B..."
echo "=========================================="
if [ -d "outputs/qwen3_0p6b" ]; then
    echo "Model already exists, skipping training..."
else
    python train_sft_trl_qwen3_1p7b.py \
        --model Qwen/Qwen3-0.6B \
        --output-dir outputs/qwen3_0p6b \
        --wandb-project wordle-comparison \
        --wandb-name qwen3-0p6b \
        $COMMON_ARGS
fi

# ==========================================
# Step 3: Test all models
# ==========================================
echo ""
echo "[3/4] Testing all models..."
echo "=========================================="

# Test PrimeIntellect/Qwen3-1.7B (already trained)
echo "Testing PrimeIntellect/Qwen3-1.7B..."
python test_wordle_detailed.py \
    --model outputs/wordle_sft_full/final \
    --output results_primeintellect_1p7b.json

# Test Qwen/Qwen3-1.7B official
echo "Testing Qwen/Qwen3-1.7B (official)..."
python test_wordle_detailed.py \
    --model outputs/qwen3_1p7b_official \
    --output results_qwen_1p7b.json

# Test Qwen/Qwen3-0.6B
echo "Testing Qwen/Qwen3-0.6B..."
python test_wordle_detailed.py \
    --model outputs/qwen3_0p6b \
    --output results_qwen_0p6b.json

# ==========================================
# Step 4: Summary
# ==========================================
echo ""
echo "[4/4] Done! Results saved:"
echo "=========================================="
echo "  - results_primeintellect_1p7b.json"
echo "  - results_qwen_1p7b.json"
echo "  - results_qwen_0p6b.json"
echo ""
echo "Open viewer.html in your browser to compare results!"
