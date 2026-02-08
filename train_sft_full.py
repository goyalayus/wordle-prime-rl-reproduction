"""
Full fine-tuning of Qwen3 on Wordle SFT data. Model-agnostic.
Works on a single A100 (40GB or 80GB).

Usage:
    python train_sft_full.py --model PrimeIntellect/Qwen3-1.7B
    python train_sft_full.py --model Qwen/Qwen3-1.7B
    python train_sft_full.py --model Qwen/Qwen3-0.6B
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer, SFTConfig

from wordle_gameplay import (
    DEFAULT_SYSTEM_PROMPT,
    GenerationConfig,
    run_gameplay_eval,
    write_json,
)

# ============ Defaults ============
DEFAULT_MODEL = "PrimeIntellect/Qwen3-1.7B"
DATASET_NAME = "willcb/V3-wordle"
BASE_OUTPUT_DIR = "outputs/wordle_sft"
MAX_SEQ_LENGTH = 1024
MAX_STEPS = 20
LEARNING_RATE = 1e-5
BATCH_SIZE = 1  # per-device
GRADIENT_ACCUMULATION = 64  # optimizer-step accumulation

DEFAULT_VAL_SIZE = 100
DEFAULT_EVAL_EVERY_STEPS = 20
DEFAULT_SAVE_EVERY_STEPS = 200

DEFAULT_SECRET_WORDS = "crane,alert,amaze,plane,store"


def model_to_folder(model_name: str) -> str:
    """Convert model name to a safe folder name (e.g. PrimeIntellect/Qwen3-1.7B -> PrimeIntellect-Qwen3-1.7B)."""
    return model_name.replace("/", "-")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Qwen3 on Wordle (model-agnostic)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--output-dir", default=BASE_OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--seq-len", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Alias for --per-device-batch-size")
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="If set, compute grad-accum so that per_device_batch_size * grad_accum == global_batch_size (single GPU).",
    )
    parser.add_argument("--seed", type=int, default=123)

    # Validation loss logging
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--eval-every-steps", type=int, default=DEFAULT_EVAL_EVERY_STEPS)

    # Saving
    parser.add_argument("--save-every-steps", type=int, default=DEFAULT_SAVE_EVERY_STEPS)
    parser.add_argument("--save-total-limit", type=int, default=2)

    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "wordle-sft"))
    parser.add_argument("--wandb-name", default=None, help="W&B run name")

    # Gameplay eval
    parser.add_argument("--gameplay-eval", action="store_true", help="Enable gameplay evaluation callback")
    parser.add_argument("--gameplay-eval-every-steps", type=int, default=DEFAULT_EVAL_EVERY_STEPS)
    parser.add_argument("--gameplay-secret-words", default=DEFAULT_SECRET_WORDS)
    parser.add_argument("--gameplay-max-new-tokens", type=int, default=8192)
    parser.add_argument("--gameplay-temperature", type=float, default=0.7)
    parser.add_argument("--gameplay-top-p", type=float, default=0.9)
    return parser.parse_args()


args = parse_args()
MODEL_NAME = args.model
OUTPUT_DIR = f"{args.output_dir}/{model_to_folder(MODEL_NAME)}"

# Resolve batch size / grad accumulation
per_device_batch_size = args.per_device_batch_size if args.per_device_batch_size is not None else args.batch_size
grad_accum = args.grad_accum
if args.global_batch_size is not None:
    if args.global_batch_size % per_device_batch_size != 0:
        raise ValueError(
            f"--global-batch-size ({args.global_batch_size}) must be divisible by per-device batch size ({per_device_batch_size})"
        )
    grad_accum = args.global_batch_size // per_device_batch_size

# ============ Load Model & Tokenizer ============
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")

# Optional: load secrets (WANDB_API_KEY) from .env if present.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # A100 supports bf16
    trust_remote_code=True,
    attn_implementation="sdpa",  # PyTorch native scaled dot product attention
)

# ============ Load & Format Dataset ============
print(f"Loading dataset: {args.dataset}")
dataset = load_dataset(args.dataset, split="train")


def format_conversation(example):
    """Combine prompt and completion into a single conversation."""
    messages = example["prompt"] + example["completion"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
print(f"Dataset size: {len(dataset)}")
print(f"Example:\n{dataset[0]['text'][:500]}...")

# ============ Train/Val split ============
val_size = max(0, min(args.val_size, len(dataset)))
if val_size > 0:
    split = dataset.train_test_split(test_size=val_size, seed=args.seed, shuffle=True)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train size: {len(train_dataset)} | Val size: {len(eval_dataset)}")
else:
    train_dataset = dataset
    eval_dataset = None
    print("Val size: 0 (no validation loss will be logged)")

# ============ Note ============
# Training on full sequence (no completion-only masking)
# This matches PrimeIntellect's approach


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_eval_manifest(evals_root: str, step_dir_name: str) -> str:
    """
    Maintain a simple manifest of eval step directories so an HTML viewer can list them.
    """
    _safe_makedirs(evals_root)
    manifest_path = os.path.join(evals_root, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
    else:
        manifest = {}
    steps: List[str] = list(manifest.get("steps", []))
    if step_dir_name not in steps:
        steps.append(step_dir_name)
    steps = sorted(steps)
    manifest.update(
        {
            "steps": steps,
            "updated_at": datetime.now().isoformat(),
        }
    )
    write_json(manifest_path, manifest)
    return manifest_path


class GameplayEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer: Any,
        output_dir: str,
        eval_every_steps: int,
        secret_words: List[str],
        gen: GenerationConfig,
        seed: int,
        enabled: bool,
        wandb_enabled: bool,
    ):
        self._tokenizer = tokenizer
        self._output_dir = output_dir
        self._eval_every_steps = eval_every_steps
        self._secret_words = secret_words
        self._gen = gen
        self._seed = seed
        self._enabled = enabled
        self._wandb_enabled = wandb_enabled

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self._enabled:
            return control
        if not getattr(state, "is_world_process_zero", True):
            return control

        step = int(getattr(state, "global_step", 0))
        if step <= 0:
            return control
        if self._eval_every_steps <= 0:
            return control
        if step % self._eval_every_steps != 0:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        evals_root = os.path.join(self._output_dir, "evals")
        step_dir_name = f"step_{step:06d}"
        step_dir = os.path.join(evals_root, step_dir_name)
        _safe_makedirs(step_dir)

        results = run_gameplay_eval(
            model=model,
            tokenizer=self._tokenizer,
            secret_words=self._secret_words,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_turns=6,
            gen=self._gen,
            seed=self._seed + step,
        )
        results.update(
            {
                "model": MODEL_NAME,
                "global_step": step,
                "timestamp": datetime.now().isoformat(),
            }
        )

        gameplay_path = os.path.join(step_dir, "gameplay.json")
        summary_path = os.path.join(step_dir, "summary.json")
        write_json(gameplay_path, results)
        write_json(summary_path, results["summary"])
        manifest_path = _write_eval_manifest(evals_root, step_dir_name)

        summary = results["summary"]
        banner = (
            "\n"
            + "=" * 88
            + "\n"
            + f"[GAMEPLAY EVAL DONE] step={step} solved_rate={summary['solved_rate']:.2f} "
            + f"solved={summary['solved_count']}/{len(self._secret_words)} avg_turns={summary['avg_turns']:.2f}\n"
            + f"- {gameplay_path}\n"
            + f"- {summary_path}\n"
            + f"- {manifest_path}\n"
            + "=" * 88
            + "\n"
        )
        print(banner, flush=True)

        if self._wandb_enabled:
            try:
                import wandb

                wandb.log(
                    {
                        "gameplay/solved_rate": summary["solved_rate"],
                        "gameplay/solved_count": summary["solved_count"],
                        "gameplay/avg_turns": summary["avg_turns"],
                        "gameplay/invalid_guess_total": summary["invalid_guess_total"],
                    },
                    step=step,
                )
            except Exception as e:
                print(f"[WARN] Failed to wandb.log gameplay metrics: {e}", flush=True)

        return control


# ============ Training Config ============
if args.wandb:
    os.environ["WANDB_PROJECT"] = args.wandb_project

report_to: Any = ["wandb"] if args.wandb else "none"
run_name = args.wandb_name
if run_name is None and args.wandb:
    run_name = f"{model_to_folder(MODEL_NAME)}__sft__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

training_kwargs = dict(
    output_dir=OUTPUT_DIR,
    max_length=args.seq_len,
    max_steps=args.max_steps,
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=2,
    logging_steps=1,
    save_steps=args.save_every_steps,
    save_total_limit=args.save_total_limit,
    bf16=True,  # A100 supports bf16
    gradient_checkpointing=True,
    report_to=report_to,
    run_name=run_name,
    dataset_text_field="text",
    seed=args.seed,
)
if eval_dataset is not None:
    training_kwargs.update(
        dict(
            do_eval=True,
            eval_strategy="steps",
            eval_steps=args.eval_every_steps,
        )
    )

training_args = SFTConfig(**training_kwargs)

# ============ Trainer ============
secret_words = [w.strip().lower() for w in args.gameplay_secret_words.split(",") if w.strip()]
gameplay_gen = GenerationConfig(
    max_new_tokens=args.gameplay_max_new_tokens,
    temperature=args.gameplay_temperature,
    top_p=args.gameplay_top_p,
    do_sample=True,
)

callbacks: List[TrainerCallback] = []
callbacks.append(
    GameplayEvalCallback(
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        eval_every_steps=args.gameplay_eval_every_steps,
        secret_words=secret_words,
        gen=gameplay_gen,
        seed=args.seed,
        enabled=args.gameplay_eval,
        wandb_enabled=args.wandb,
    )
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=callbacks,
)

# ============ Train ============
print("Starting training...")
trainer.train()

# ============ Save ============
print(f"Saving model to {OUTPUT_DIR}/final")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Done!")
