---
date: 2026-02-08
branch: long-sft-run-experiment
owner: ayush
status: planning
---

# Long SFT run experiment (Qwen3-0.6B) with periodic gameplay evals

## Goal
Run **longer** supervised fine-tuning (SFT) on **`Qwen/Qwen3-0.6B`** and, every **20 optimizer steps**, run a small **gameplay evaluation** (5 fixed Wordle secret words) to visually verify whether the model’s behavior improves over training. Log **train loss** and **validation loss** to **Weights & Biases (W&B)**, and write eval artifacts in a form that’s easy to inspect via HTML.

## Context
- Repo: `Desktop/code/wordle-prime-rl-reproduction/`
- Current training script: `train_sft_full.py` (TRL `SFTTrainer`, bf16, no W&B, no eval loop).
- Current prompt-based evaluation scripts:
  - `test_wordle_detailed.py` (4 fixed prompts, generates long completions; not true gameplay)
  - `compare_models.py` + `viewer.html` (side-by-side JSON viewer for fixed prompts)
- Desired evaluation change: **actual Wordle gameplay** over **5 fixed secret words** (6 turns max each), not just single-turn prompts, with **full transcripts preserved**.
- Dataset: `willcb/V3-wordle` train split has **1000 rows** (columns: `prompt`, `completion`, `answer`, `reward`, `task`).
  - With an intended **global batch size ~60 examples / optimizer step**, this is ~**17 steps per epoch** (1000 / 60 ≈ 16.7).
  - Therefore: `max_steps=340` ≈ 20 epochs, `850` ≈ 50 epochs, `1000` ≈ 59 epochs (approx).
- Runtime target: **single A100** via SSH on Lightning.ai.
- Notification requirement: **terminal** is sufficient (“eval finished” banners with paths).

## Constraints / non-goals
- **Don’t** run gameplay eval at step 0. First eval should occur at step 20.
- Eval frequency is in **optimizer steps** (`trainer.state.global_step`), not micro-steps.
- Training should try to “use the full GPU” while preserving an **effective/global batch size of 60** (implementation should support either:
  - user-provided `per_device_train_batch_size` + derived `gradient_accumulation_steps`, or
  - explicit `gradient_accumulation_steps` if the user wants full control).
- Eval should be **lightweight** in *count* (only 5 games), but we should **not artificially cap “thinking”**; we want to let the model produce long reasoning if it chooses.
  - Practical constraint: generation must still terminate reliably. Prefer stopping once `</guess>` is emitted rather than truncating the model mid-response.
- W&B should log:
  - training loss (already emitted by Trainer’s logging)
  - validation loss (via `evaluation_strategy="steps"`)
  - gameplay metrics (success rate, avg turns, invalid guess count, etc.) + links/paths to artifacts.

## Proposed changes (files / components)
1) **Training**
   - Update `train_sft_full.py` to support:
     - W&B reporting and run naming
     - train/val split for a small validation set
     - step-based evaluation (`eval_steps=20`) that logs `eval_loss` without running at step 0
     - configuration knobs for effective/global batch size = 60 on a single GPU
     - a callback that triggers gameplay eval every `N` steps (default 20), prints terminal notifications, and writes artifacts.
2) **Gameplay evaluator**
   - Add a new script `eval_wordle_gameplay.py` that:
     - simulates a Wordle game (secret word, feedback computation)
     - prompts the model turn-by-turn with prior guesses + feedback
     - extracts guesses robustly (`<guess>…</guess>` or `[apple]`)
     - saves JSON transcript + summary metrics for a run/step.
3) **HTML inspection**
   - Add a lightweight “long-run eval viewer” HTML page that can:
     - load a manifest of eval steps
     - display per-step results and per-word transcripts
     - be served via `python3 -m http.server` for local inspection.
   - The viewer should point at the eval artifact directory produced during training.
4) **Run wrapper**
   - Add a shell script for Lightning A100 that runs the long SFT experiment with sensible defaults:
     - model: `Qwen/Qwen3-0.6B`
     - `max_steps`: default target (see below)
     - `eval_every_steps=20`
     - effective/global batch size = 60
     - W&B project and run name.
5) **Docs**
   - Update `SFT_README.md` to document the “long SFT + evals” workflow and how to view results.
6) **Architecture**
   - Update `ARCHITECTURE.md` after implementation so the repo’s docs reflect the new periodic-eval pipeline.

## Key design decisions (draft)
### Max steps target
Because the dataset is only 1000 rows, “how much can we go” is not a hard ceiling; we will **loop over data for many epochs**. The plan is to make `--max-steps` configurable and set a default that is “meaningfully longer than 20 steps” but still practical:
- Proposed default: **`--max-steps 1000`** (~59 epochs at global batch size 60).
- Rationale: enough to test the hypothesis (“0.6B is initially bad but learns”) with many evaluation points (50 evals at step interval 20).

### Validation loss
We will hold out a small validation slice (e.g. 100 examples) from the same HF dataset and use the Trainer’s built-in `evaluate()` on that set every `eval_steps`.
- This produces `eval_loss` and allows tracking overfitting vs general learning.
- Note: this is **not** the same as gameplay success, but complements it.

### Gameplay eval format
We will run 5 secret words (fixed list in config so results are comparable across steps). Each game:
- up to 6 turns
- model must output a guess; we extract a 5-letter token from `<guess>…</guess>` or `[…]`
- compute Wordle feedback (G/Y/B)
- feed that back into the next user prompt
- stop early if correct

To accommodate “let it think as long as it wants” while keeping evaluation robust:
- We set a high `max_new_tokens` ceiling.
- We add a **stopping criterion** that ends generation once the model emits the token sequence corresponding to `</guess>`.
  - This does not prevent long `<think>`; it only stops after the guess is produced.

We will store per-step results as:
- `outputs/<run_dir>/evals/step_<global_step>/gameplay.json` (full transcripts)
- `outputs/<run_dir>/evals/step_<global_step>/summary.json` (metrics)
- `outputs/<run_dir>/evals/manifest.json` updated incrementally (list of step dirs)

### Terminal notifications
At the end of each eval:
- print a high-signal banner including step number and artifact paths
- (optional later) print a short table of secret-word outcomes.

## Steps (checkable)
1) Bootstrap repo artifacts:
   - Ensure `plan/` exists.
   - Ensure `ARCHITECTURE.md` exists (initial snapshot).
2) Implement gameplay evaluator script and verify it runs on an existing local model directory (or HF model id).
3) Extend training pipeline to:
   - enable W&B logging
   - create a small validation set and log `eval_loss`
   - run gameplay eval callback every 20 optimizer steps (not step 0)
   - write eval artifacts + manifest + viewer.
4) Add run script for Lightning A100 with recommended flags (Qwen3-0.6B long run).
5) Update docs (`SFT_README.md`) to include:
   - “long SFT run” invocation
   - how to view results (start http server, open viewer)
6) Run a short smoke test locally (tiny `--max-steps 2` + `eval_every_steps 1` on CPU or small GPU, if feasible) to validate integration without incurring full training cost.
7) Update `ARCHITECTURE.md` to reflect the new components and flows.

## Acceptance criteria
- Branch `long-sft-run-experiment` exists and contains:
  - a reproducible “long SFT run” entrypoint that targets `Qwen/Qwen3-0.6B`
  - training loss and validation loss visible in W&B
  - gameplay eval runs every 20 optimizer steps (starting at step 20) and prints terminal notification when complete
  - eval artifacts are written per-step and are viewable with an HTML viewer.
- The evaluation output is stable/reproducible with fixed secret words (given a fixed seed) and does not produce unbounded output (bounded tokens per turn).

## Risks / rollback
- **Disk usage**: frequent checkpoints can fill storage. Mitigation: decouple `save_steps` from eval steps and keep a small `save_total_limit`.
- **Eval runtime**: generation can be slow if the model produces very long reasoning. Mitigation:
  - Stop generation after `</guess>` (instead of truncating earlier).
  - Keep eval suite small (5 games).
- **Prompt drift**: small prompt changes can dominate behavior. Mitigation: keep prompts stable and versioned in evaluator script.
- **W&B auth**: missing API key will cause failures if `report_to="wandb"`. Mitigation: allow disabling W&B with a flag; load `.env` if present.
- **Overfitting**: many epochs might cause memorization; validation loss will help detect this.
- **Rollback**: revert commit(s) touching training/eval scripts; old scripts (`train_sft_full.py` behavior) should remain runnable with minimal flags.

## Status log
- 2026-02-08 17:07
  - Created branch `long-sft-run-experiment`.
  - Verified dataset size for `willcb/V3-wordle` (1000 rows).
  - Confirmed user requirements:
    - eval every 20 optimizer steps, not at step 0
    - “batch size = 60” effective/global batch size
    - 5 secret words with full gameplay
    - terminal notifications on eval completion
    - log train + validation loss in W&B.

- 2026-02-08 17:23
  - Bootstrapped required repo artifacts:
    - created `plan/`
    - created initial `ARCHITECTURE.md` snapshot (pre-change).
  - Implemented gameplay evaluation as a reusable module + CLI:
    - `wordle_gameplay.py`: Wordle feedback logic, guess extraction, turn-by-turn gameplay loop, aggregate eval runner, JSON writer.
    - `eval_wordle_gameplay.py`: CLI wrapper that loads a model (HF id or local path) and writes `gameplay.json` + `summary.json`.
  - Extended training script (`train_sft_full.py`) to support long-run experiment needs:
    - effective/global batch size support (`--global-batch-size` + `--per-device-batch-size` -> derived `gradient_accumulation_steps`)
    - small train/val split (`--val-size`) so we can log validation loss
    - step-based validation evaluation via Trainer (`evaluation_strategy="steps"`, `eval_steps=--eval-every-steps`)
    - W&B logging toggle (`--wandb`) + run naming (`--wandb-name`) and project (`--wandb-project`, forwarded via `WANDB_PROJECT`)
    - gameplay eval callback (triggered on `trainer.state.global_step`, every `--gameplay-eval-every-steps`, not at step 0):
      - plays 5 fixed secret words
      - writes artifacts under `<OUTPUT_DIR>/evals/step_XXXXXX/`
      - updates `<OUTPUT_DIR>/evals/manifest.json`
      - prints a terminal banner with solved rate and artifact paths
      - logs gameplay summary metrics to W&B when enabled.
  - Added `viewer_long_run.html`:
    - loads `<eval_root>/manifest.json`, lists steps, and renders per-step gameplay transcripts.
  - Added `run_long_sft_a100.sh`:
    - A100-oriented wrapper with sensible defaults:
      - `Qwen/Qwen3-0.6B`, `MAX_STEPS=1000`, `GLOBAL_BS=60`, eval every 20 steps, save every 200 steps
      - enables W&B + gameplay eval by default
      - prints a URL template for `viewer_long_run.html` with the correct `root=` query param.
  - Updated `SFT_README.md` with a “Long SFT run + periodic gameplay evals” section and viewer instructions.
  - Performed lightweight validation:
    - `python3 -m py_compile train_sft_full.py wordle_gameplay.py eval_wordle_gameplay.py` passes.

- 2026-02-08 17:34
  - Updated plan constraints based on new experiment requirements:
    - do not cap “thinking”; allow long model outputs
    - termination should be controlled via stop-after-`</guess>` rather than low `max_new_tokens`.
  - Captured operational workflow constraint:
    - edit locally → pull on Lightning → train on Lightning → push from Lightning → pull locally (no editing on Lightning).
  - Clarified training/eval interaction expectation:
    - On a single GPU, eval will *pause* the training loop while generating, then training resumes.
    - True parallel eval would require a second GPU/process loading checkpoints; not planned unless explicitly required.

- 2026-02-08 17:40
  - Added `.env.example` template (no secrets committed) documenting required tokens:
    - `WANDB_API_KEY`
    - `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`
  - Updated docs (`SFT_README.md`) and architecture (`ARCHITECTURE.md`) to reflect HF token usage and `.env.example`.

- 2026-02-08 17:48
  - Updated `upload_to_hf.py` to support the long-run workflow without editing on Lightning:
    - auto-load `.env` if present
    - supports `--local-path` + `--repo-id` to upload arbitrary outputs (e.g. `outputs/wordle_sft_long/.../final`)
    - retains backwards-compatible default upload list.
- 2026-02-08 18:17
  - Fixed gameplay logging crash: `_format_turn_history` now tolerates missing `feedback` (invalid guesses) so the callback doesn’t throw during early evals.

- 2026-02-08 18:25
  - Adjusted `run_long_sft_a100.sh` to skip the heavy `flash-attn` build unless `INSTALL_FLASH_ATTENTION=1`, reducing compile time on Lightning. Training now restarts cleanly after pulling latest branch.
