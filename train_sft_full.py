"""
Full fine-tuning of Qwen3 on Wordle SFT data. Model-agnostic.
Works on a single A100 (40GB or 80GB).

Usage:
    python train_sft_full.py --model PrimeIntellect/Qwen3-1.7B
    python train_sft_full.py --model Qwen/Qwen3-1.7B
    python train_sft_full.py --model Qwen/Qwen3-0.6B
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ============ Defaults ============
DEFAULT_MODEL = "PrimeIntellect/Qwen3-1.7B"
DATASET_NAME = "willcb/V3-wordle"
BASE_OUTPUT_DIR = "outputs/wordle_sft"
MAX_SEQ_LENGTH = 1024
MAX_STEPS = 20
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 64


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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION)
    return parser.parse_args()


args = parse_args()
MODEL_NAME = args.model
OUTPUT_DIR = f"{args.output_dir}/{model_to_folder(MODEL_NAME)}"

# ============ Load Model & Tokenizer ============
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_DIR}")
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

# ============ Note ============
# Training on full sequence (no completion-only masking)
# This matches PrimeIntellect's approach

# ============ Training Config ============
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=args.seq_len,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=2,
    logging_steps=1,
    save_steps=args.max_steps,
    save_total_limit=2,
    bf16=True,  # A100 supports bf16
    gradient_checkpointing=True,
    report_to="none",  # Disable wandb (not logged in)
    dataset_text_field="text",
)

# ============ Trainer ============
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ============ Train ============
print("Starting training...")
trainer.train()

# ============ Save ============
print(f"Saving model to {OUTPUT_DIR}/final")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Done!")
