# Qwen3-4B LoRA SFT for Wordle

LoRA fine-tuning of Qwen3-4B on the Wordle dataset with completion-only loss.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `.env` file with your tokens (already configured):
- `HF_TOKEN` - HuggingFace token
- `HF_REPO_ID` - Target repo (default: goyalayus/wordle-lora-qwen3-4b)
- `WANDB_API_KEY` - Weights & Biases API key (optional)

## Usage

### Basic training
```bash
python train_sft_lora_qwen4b.py
```

### With wandb logging
```bash
python train_sft_lora_qwen4b.py --wandb
```

### Custom settings
```bash
python train_sft_lora_qwen4b.py \
  --per-device-batch-size 8 \
  --lr 2e-5 \
  --wandb \
  --wandb-name my-run
```

## Features

- **Model**: Qwen/Qwen3-4B with 4-bit quantization
- **LoRA**: r=16, alpha=32, target_modules=[gate_proj, up_proj, down_proj]
- **Dataset**: willcb/V3-wordle
- **Loss**: Completion-only (only on assistant tokens)
- **Batch Size**: 16 per device, grad_accum=4 (global=64)
- **Eval**: Every 20 steps
- **Checkpointing**: Push to HF Hub every 20 steps
- **Early Stopping**: 
  - Stop after 3 evaluations with no improvement, OR
  - Stop after 2 consecutive evaluations with increasing loss

## Expected VRAM Usage on T4

- Model (4-bit): ~2.5GB
- LoRA weights: ~100MB
- Activations (batch=16, seq=1024): ~4-6GB
- Optimizer (8-bit): ~1-2GB
- **Total**: ~8-11GB / 15GB available

If OOM, reduce `--per-device-batch-size` to 8 or 4.

## Output

Checkpoints pushed to: `https://huggingface.co/goyalayus/wordle-lora-qwen3-4b`

Revisions:
- `step-20`, `step-40`, `step-60`, ... - Intermediate checkpoints
- `final` - Final checkpoint after training completes or early stopping

## Monitoring

- **Weights & Biases**: Set `--wandb` to enable
- **Console**: Real-time loss and eval metrics
- **HF Hub**: Check adapter revisions at the repo URL
