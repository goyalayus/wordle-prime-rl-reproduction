# ML Learnings (project copy)

**Canonical location for agents**: `~/.cursor/ML_LEARNINGS.md`

This file is a pointer. All ML training learnings are stored in `~/.cursor/ML_LEARNINGS.md` so any agent can access them across projects without being told.

---

## 1. Python vs python3

- **Issue**: `python` command often not found on Linux; many systems only have `python3`.
- **Fix**: Use `python3` explicitly when running scripts, or ensure `python` symlinks to `python3`.

---

## 2. PIL/Pillow and transformers

- **Issue**: `AttributeError: module 'PIL.Image' has no attribute 'Resampling'` when importing TRL/transformers.
- **Cause**: Old Pillow (< 9.1) lacks `PIL.Image.Resampling`.
- **Fix**: `pip install --upgrade Pillow` (e.g. Pillow 12.x).

---

## 3. tee and missing directories

- **Issue**: `tee outputs/sft_train.log` fails with "No such file or directory" if `outputs/` does not exist.
- **Fix**: Run `mkdir -p outputs` before piping to `tee outputs/sft_train.log`.

---

## 4. Batch size vs gradient accumulation

- **per_device_train_batch_size**: Number of examples per forward pass on each GPU. With batch_size=64, 64 examples are processed in parallel (subject to GPU memory).
- **gradient_accumulation_steps**: Number of forward passes before an optimizer step. Gradients are accumulated and averaged.
- **Effective batch size** = `per_device_batch_size × gradient_accumulation_steps × num_gpus`.
- **Why use gradient accumulation**: When GPU memory limits batch_size to 1, use gradient_accumulation_steps=64 to mimic batch_size=64.

---

## 5. PrimeIntellect SFT config (from sft.toml)

- `batch_size = 64`, `micro-batch-size = 1` → effective batch 64 via gradient accumulation.
- `max_steps = 20`, `seq_len = 1024`, `lr = 1e-5`.
- Dataset: `willcb/V3-wordle` (multi-turn Wordle traces).

---

## 6. Completion-only loss (agent tokens only)

- **Goal**: Compute loss only on assistant/agent tokens, not on user or system tokens.
- **TRL**: Use `completion_only_loss=True` in `SFTConfig` with prompt-completion format.
- **Data format**: Provide `prompt` (messages before assistant) and `completion` (assistant message) so the trainer masks non-completion tokens.

---

## 7. Model loading and mixed precision

- **Issue**: Loading with `torch_dtype=torch.float16` can conflict with Accelerate's gradient scaling → `ValueError: Attempting to unscale FP16 gradients`.
- **Fix**: Load model in `torch.float32`; let the trainer handle fp16/bf16 via `fp16`/`bf16` in the config.

---

## 8. W&B API key persistence

- **Local**: Use `.env` with `WANDB_API_KEY=...` and `python-dotenv`; add `.env` to `.gitignore`.
- **Remote (Lightning)**: Run `wandb login` once over SSH; key is stored in `~/.netrc` or `~/.config/wandb/`.
- **For agents**: Store key in a known location (e.g. `~/.config/wandb/api_key` or project `.env`) so future runs don't need manual login.

---

## 9. Long-running training commands

- Training can run 30+ minutes (downloads, 20 steps, etc.).
- **Options**: Run in background with `is_background=true`, or use `nohup`/`tmux`/`screen` for persistence.
- Avoid short timeouts when running training.

---

## 10. Hugging Face downloads

- Model/dataset downloads can be slow; the progress bar may stay at "Fetching X files: 0%" for several minutes.
- First run downloads to `~/.cache/huggingface/`. Subsequent runs use cache.
- Ensure network access for `all` when running (HF needs to fetch).

## 11. Lightning.ai SSH workflow

- **SSH**: `ssh s_<session_id>@ssh.lightning.ai` (from Connect → SSH in Studio).
- **Workflow**: Push code to GitHub → SSH in → `git clone` → `pip install -r requirements.txt` → `wandb login` → run training.
- **Connect local Cursor**: Opens a new Cursor window connected to the remote filesystem; your local project files are not there by default. Use Git + SSH for a simpler flow.
