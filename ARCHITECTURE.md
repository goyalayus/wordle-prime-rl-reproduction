# Architecture

This document is intentionally detailed and intended for readers who are comfortable with PyTorch/Transformers/TRL training loops and evaluation harnesses.

## High-level overview

This repository contains scripts to reproduce a small-scale Wordle **supervised fine-tuning (SFT)** setup on a conversational dataset (`willcb/V3-wordle`). The core idea is:
- Format each dataset row into a chat-style transcript (`prompt` + `completion`) using the model’s chat template.
- Train a causal language model (e.g., Qwen3 variants) with TRL’s `SFTTrainer`.
- Evaluate model behavior by prompting for Wordle guesses and comparing responses across models.

Primary user flows:
1) **Train** a model checkpoint (full fine-tune, bf16) on the Wordle SFT dataset.
2) **Evaluate** the trained model:
   - either via fixed “turn prompts” (prompt-based evaluation), or
   - via **full Wordle gameplay** on a small fixed suite of secret words (periodic evals during training).
3) **View/compare** JSON outputs in a browser via a local HTML viewer (static files served over HTTP).

## Repo map (directory-by-directory)

### Repo root
- `SFT_README.md`: Instructions for SFT setup and running, including cloud (Lightning.ai) workflow.
- `MODELS.md`: Links to example fine-tuned models hosted on Hugging Face.
- `requirements.txt`: Python dependencies for training/eval (Torch, Transformers, TRL, etc.).
- `train_sft_full.py`: Main full fine-tune script using TRL `SFTTrainer` and `SFTConfig` (supports W&B, validation loss, and optional periodic gameplay eval callbacks).
- `run_sft_a100.sh`: Convenience script to install dependencies and run `train_sft_full.py` on an A100.
- `run_long_sft_a100.sh`: Convenience script for the “long SFT run” experiment (Qwen3-0.6B default) with effective/global batch size = 60, validation loss logging, and gameplay eval every N optimizer steps.
- `run_sft_deepspeed.sh`: Convenience script for multi-GPU DeepSpeed training (expects `train_sft_deepspeed.py`, which is currently not present in this repo snapshot).
- `test_wordle_detailed.py`: Prompt-based evaluation script that generates detailed outputs and extracts guesses from model responses.
- `compare_models.py`: Runs the same prompts over multiple models (HF model ids) and saves results to JSON for viewing.
- `wordle_gameplay.py`: Wordle gameplay simulator + evaluation harness used for periodic “play 5 secret words” evals and for standalone gameplay eval runs.
- `eval_wordle_gameplay.py`: Standalone CLI entrypoint that loads a model and runs gameplay eval (writes `gameplay.json` + `summary.json`).
- `viewer.html`: Browser viewer that loads 1–3 JSON files and renders side-by-side comparisons.
- `viewer_long_run.html`: Browser viewer that loads an eval root directory (manifest + per-step artifacts) produced by periodic gameplay evals during training.
- `upload_to_hf.py`: Upload helper to push trained model folders (e.g. `.../final`) to the Hugging Face Hub; supports both the repo’s legacy default list and arbitrary `--local-path`/`--repo-id` uploads, and auto-loads `.env` when available.
- `results_*.json`, `wordle_test_results.json`: Example evaluation outputs (JSON artifacts).
- `training_log_poll.txt`, `watch_logs_on_lightning.sh`, `check_training_logs.sh`: Utility scripts/artifacts for log monitoring.
- `.env` (gitignored): Typically contains secrets like `WANDB_API_KEY`.

### `outputs/`
Training outputs directory. The existing scripts write trained model artifacts/checkpoints here. The structure is driven by `--output-dir` and the model name (sanitized).

### `plan/`
Task planning and execution logs. Each change to the repo should have a corresponding plan file describing goals, constraints, steps, and verification.

## Components / modules

### Training pipeline (`train_sft_full.py`)
Responsibilities:
- Load model + tokenizer (`AutoModelForCausalLM`, `AutoTokenizer`).
- Load dataset via `datasets.load_dataset`.
- Convert dataset examples into a single `text` field using `tokenizer.apply_chat_template`.
- Configure and run TRL’s `SFTTrainer`:
  - bf16 training
  - sequence length truncation
  - optimizer steps limited by `max_steps`
  - periodic checkpoint saving
- Optionally split into train/validation sets and log validation loss periodically.
- Optionally run **periodic gameplay evaluation** every N optimizer steps and persist artifacts to disk.
- Save final model and tokenizer to `<output_dir>/final`.

Important invariants and assumptions:
- Tokenizer must have `pad_token` configured; the script sets it to `eos_token` if missing.
- The dataset is converted to plain text; the trainer uses `dataset_text_field="text"`.
- Current script notes that it trains on the full sequence (not completion-only masking).
- “Optimizer step” semantics:
  - Periodic evaluation is keyed off `trainer.state.global_step`, i.e., after gradient accumulation is applied.
  - There is intentionally **no** gameplay eval at step 0; first eval happens at step N (e.g. 20).

Key configuration knobs:
- **Effective/global batch size**:
  - Either specify `--grad-accum` directly, or specify `--global-batch-size` along with `--per-device-batch-size`, and the script derives `gradient_accumulation_steps = global_batch_size / per_device_batch_size` (single-GPU assumption).
- **Validation loss**:
  - `--val-size` controls the held-out subset size (random split with `--seed`).
  - If validation is enabled, `evaluation_strategy="steps"` and `eval_steps=--eval-every-steps` are set so that `eval_loss` is logged.
- **W&B**:
  - Enabled via `--wandb`.
  - The project is set via `--wandb-project` (forwarded to `WANDB_PROJECT`) and the run name via `--wandb-name` / `run_name`.

### Gameplay evaluation harness (`wordle_gameplay.py`)
Responsibilities:
- Implement Wordle feedback logic (`G/Y/B`) including duplicate-letter handling (`compute_feedback`).
- Implement robust guess extraction from model output (`extract_guess`):
  - prefer `<guess>…</guess>`; fall back to `[...]`.
- Simulate turn-by-turn gameplay (`play_game`):
  - build a user prompt containing prior turns and feedback
  - generate model output
  - extract guess
  - compute feedback from the secret word
  - stop early when solved or after 6 turns
- Aggregate over a fixed suite (`run_gameplay_eval`) and compute summary metrics:
  - solved rate, average turns, invalid guess counts.

Important invariants:
- Secrets are never revealed to the model; prompts contain only public game state.
- Invalid or unparseable guesses are recorded with `error` fields and count toward the 6-turn limit.
- Generation is bounded by `GenerationConfig.max_new_tokens` so evaluation cannot “run away” with long responses.

### Periodic gameplay eval callback (`train_sft_full.py`)
Responsibilities:
- Trigger `run_gameplay_eval` every `--gameplay-eval-every-steps` optimizer steps.
- Write artifacts under:
  - `<OUTPUT_DIR>/evals/step_<global_step>/gameplay.json`
  - `<OUTPUT_DIR>/evals/step_<global_step>/summary.json`
  - `<OUTPUT_DIR>/evals/manifest.json` (list of step directories, updated incrementally)
- Print a terminal banner after each eval so the operator notices evaluation completion without polling.
- If W&B is enabled, log gameplay summary metrics under `gameplay/*` at the same `step`.

### Evaluation harness (`test_wordle_detailed.py`, `compare_models.py`)
Responsibilities:
- Load a model (either from a local path or HF model id).
- Construct a chat prompt consisting of a system prompt and a user prompt describing the Wordle state.
- Generate a response and attempt to parse a guess.
- Save full responses to JSON so they can be reviewed later (including very long `<think>` sections).

Important invariants:
- Guess extraction is regex-based and expects `<guess>...</guess>` or `[...]`.
- Evaluation is not full gameplay; it is a set of fixed “turn prompts” that mimic a partial game state.

### Viewer (`viewer.html`)
Responsibilities:
- Load JSON output files produced by the evaluation scripts.
- Render per-test responses for up to three models side-by-side.

Operational considerations:
- Loading local JSON via `fetch()` typically requires serving files via an HTTP server (e.g. `python3 -m http.server`) rather than opening `file://` URLs in some browsers due to CORS restrictions.

### Viewer (long-run) (`viewer_long_run.html`)
Responsibilities:
- Load an eval root directory (provided manually or via a `?root=...` query parameter).
- Fetch `<root>/manifest.json` to list available `step_XXXXXX` directories.
- Fetch `<root>/<step>/gameplay.json` and render:
  - per-step summary metrics
  - per-secret-word game outcomes
  - per-turn feedback + raw model outputs (expandable).

## Runtime architecture

This repo is script-driven; there are no long-running services. Typical processes:
- **Training**: a single Python process that runs forward/backward passes on GPU.
- **Evaluation**: a Python process that loads a model and performs generation.
- **Viewing**: a static HTML page opened in a browser (optionally via a local HTTP file server).

## Data architecture

### Dataset (`willcb/V3-wordle`)
The dataset is pulled from the Hugging Face Hub. Relevant fields:
- `prompt`: list of chat messages (roles + content) representing the game context.
- `completion`: list of chat messages (assistant outputs).
- `answer`, `reward`, `task`: additional metadata used by the original dataset; not directly consumed by current training scripts after formatting.

### Output artifacts
- Model checkpoints and final model weights under `outputs/`.
- Prompt-based evaluation results as JSON under repo root (e.g. `results_*.json`) or user-chosen directories.
- Periodic gameplay evaluation artifacts (produced during training) under:
  - `<OUTPUT_DIR>/evals/manifest.json`
  - `<OUTPUT_DIR>/evals/step_XXXXXX/{gameplay.json,summary.json}`

The intent is that the viewer (`viewer_long_run.html`) uses the manifest to navigate the per-step artifacts without requiring any special backend.

## Integration points
- **Hugging Face Hub**: model weights and dataset downloads.
- **Weights & Biases (wandb)**: optional experiment tracking (API key in `.env` or `wandb login`).
- **Lightning.ai**: recommended runtime environment for A100/T4 experiments.

## Build / test / run
- Install dependencies: `pip install -r requirements.txt`.
- Train on A100: `bash run_sft_a100.sh` (installs deps and runs `train_sft_full.py`).
- Long SFT run on A100 (Qwen3-0.6B + periodic evals): `bash run_long_sft_a100.sh`.
- Evaluate: `python3 test_wordle_detailed.py --model <model_path_or_hf_id>`.
- Standalone gameplay eval: `python3 eval_wordle_gameplay.py --model <model_path_or_hf_id> --output-dir <dir>`.
- Compare models: `python3 compare_models.py`.
- View results: run a local web server in the repo and open `viewer.html`.
- View long-run evals:
  - serve repo root via HTTP and open `viewer_long_run.html?root=<eval_root>`, where `<eval_root>` is an `.../evals` directory containing a `manifest.json`.

## Observability
- Training scripts print progress; W&B logging is optional depending on script configuration.
- Utilities exist to poll/watch logs in cloud environments (`training_log_poll.txt`, `watch_logs_on_lightning.sh`).
- When enabled, W&B logs:
  - training loss (Trainer logging)
  - validation loss (`eval_loss`) at `eval_steps`
  - gameplay summary metrics (`gameplay/*`) when periodic gameplay eval is enabled.

## Security model
- Secrets are expected to live in `.env` (gitignored) and/or be provided via environment variables.
  - `WANDB_API_KEY` for W&B.
  - `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) for Hugging Face Hub authentication / rate limits.
- A template file `.env.example` is provided to document required/optional variables without committing secrets.
- No authentication/authorization logic exists in the repo; this is a local research workflow.

## Operational notes / failure modes
- **OOM**: sequence length, batch size, and gradient accumulation must be tuned for the target GPU.
- **Tokenizer special tokens**: missing `pad_token` can break batching/generation; scripts patch this to `eos_token`.
- **Browser CORS**: local JSON loads may fail if `viewer.html` is opened directly; serving via HTTP avoids this.
- **Disk usage**: periodic evaluation writes JSON each interval; checkpoint saving frequency should be decoupled from eval frequency to avoid excessive storage consumption.

## Request / data flow (critical path): long SFT run with periodic gameplay eval

This describes the runtime flow when running `run_long_sft_a100.sh` (or equivalent flags).

1) **Process initialization**
   - Shell script installs dependencies, sets env vars, and invokes `python3 train_sft_full.py ...`.
   - Optional `.env` is loaded (via `python-dotenv`) to populate `WANDB_API_KEY` etc.
   - If `--wandb` is set, `WANDB_PROJECT` is set from `--wandb-project`; Transformers/TRL integrates with W&B via `report_to=["wandb"]`.

2) **Model and tokenizer load**
   - `AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)`; `pad_token` is set to `eos_token` if missing.
   - `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bf16, attn_implementation="sdpa")`.

3) **Dataset load + formatting**
   - HF dataset `willcb/V3-wordle` is loaded (`split="train"`).
   - Each row is transformed into a single `text` string by concatenating `prompt + completion` and applying the tokenizer’s chat template.

4) **Train/val split**
   - If `--val-size > 0`, a random split is created (seeded by `--seed`).
   - The validation subset is used only for `eval_loss`; it does not drive early stopping.

5) **Trainer configuration**
   - `SFTConfig` sets `max_steps`, batch size, gradient accumulation, bf16, checkpointing cadence, etc.
   - If validation is enabled, `evaluation_strategy="steps"` and `eval_steps=N` triggers periodic `evaluate()` calls (after step N, 2N, …).

6) **Training loop**
   - For each optimizer step:
     - TRL/Trainer performs forward/backward passes across `gradient_accumulation_steps` micro-batches.
     - Updates `state.global_step`.
     - Logs training loss and (optionally) evaluation loss to W&B.

7) **Periodic gameplay eval callback**
   - On step end, if `global_step % gameplay_eval_every_steps == 0` and `global_step > 0`:
     - `run_gameplay_eval` plays the model against the fixed 5 secret words.
     - Artifacts are written under `<OUTPUT_DIR>/evals/step_<global_step>/`.
     - `manifest.json` is updated so the viewer can discover new steps.
     - A terminal banner is printed (operator notification).
     - Summary metrics are logged to W&B (if enabled).

8) **Final save**
   - The final model and tokenizer are saved to `<OUTPUT_DIR>/final`.
