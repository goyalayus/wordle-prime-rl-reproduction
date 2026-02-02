# Wordle SFT (TRL, full fine-tune)

This reproduces the PrimeIntellect Wordle SFT setup using standard TRL on a single GPU
runtime (Lightning.ai T4).

## Quick start (Git + SSH)

1. **Push to GitHub** (from your machine):
   ```bash
   cd wordle-prime-rl-reproduction
   git init && git add . && git commit -m "Wordle SFT script"
   git remote add origin https://github.com/YOUR_USERNAME/wordle-prime-rl-reproduction.git
   git push -u origin main
   ```

2. **SSH into Lightning Studio** and run:
   ```bash
   ssh s_01kgf6rvtqvq07rkr85w4pwf62@ssh.lightning.ai
   ```

3. **On the Studio**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/wordle-prime-rl-reproduction.git
   cd wordle-prime-rl-reproduction
   source /home/zeus/miniconda3/bin/activate
   pip install -r requirements.txt
   wandb login   # paste API key from wandb.ai/authorize
   python train_sft_trl_qwen3_1p7b.py --gradient-checkpointing \
     --wandb-project wordle-sft --wandb-name my-run
   ```

## Running on Lightning.ai

### Option A: SSH from your terminal

1. Create a Studio at [studio.lightning.ai](https://studio.lightning.ai) and pick a GPU (e.g. T4).
2. In the Studio UI, open **Connect** → **SSH** to get the SSH command (e.g. `ssh user@host -p port -i ~/.ssh/lightning`).
3. From your local terminal:
   ```bash
   # Copy your code to the Studio
   scp -P <port> -i ~/.ssh/lightning -r wordle-prime-rl-reproduction user@host:~/
   # SSH in
   ssh user@host -p <port> -i ~/.ssh/lightning
   ```
4. Inside the Studio shell:
   ```bash
   pip install torch transformers trl datasets accelerate wandb
   cd wordle-prime-rl-reproduction
   wandb login   # paste your API key from wandb.ai/authorize
   python train_sft_trl_qwen3_1p7b.py --gradient-checkpointing \
     --wandb-project wordle-sft --wandb-name qwen3-1p7b-run1
   ```

### Option B: Connect your IDE (VS Code / Cursor)

1. In the Studio, go to **Connect** → **Local IDE**.
2. Follow the steps to connect VS Code or Cursor via Remote-SSH.
3. Edit and run the script from your IDE; execution happens on the cloud GPU.

## Environment setup

```bash
pip install -U "torch>=2.1" "transformers>=4.56.2" "trl>=0.22.2" "datasets>=2.18.0" accelerate wandb
```

## Run SFT

```bash
python train_sft_trl_qwen3_1p7b.py \
  --model PrimeIntellect/Qwen3-1.7B \
  --dataset willcb/V3-wordle \
  --split train \
  --seq-len 1024 \
  --max-steps 20 \
  --learning-rate 1e-5 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 64 \
  --output-dir outputs/wordle_sft_qwen3_1p7b \
  --gradient-checkpointing \
  --wandb-project wordle-sft \
  --wandb-name my-run
```

Training curves will appear at [wandb.ai](https://wandb.ai) under the project you specify.

## W&B API key

- **Local**: The script loads `WANDB_API_KEY` from `.env` (gitignored). Create `.env` with:
  ```
  WANDB_API_KEY=your_key_from_wandb.ai/authorize
  ```
- **Lightning**: Run `wandb login` once after SSH, or copy your `.env` to the Studio (it is not in git).

## Notes

- This is **full-parameter** fine-tuning (no LoRA, no quantization).
- **Loss is computed only on assistant tokens** (uses TRL's built-in `completion_only_loss` with prompt-completion format).
- The dataset is exploded into one row per assistant turn so each example has a single completion to predict.
- If you see OOM on T4, reduce `--seq-len` or increase gradient accumulation.
- Checkpoints are written to the `--output-dir` path.
