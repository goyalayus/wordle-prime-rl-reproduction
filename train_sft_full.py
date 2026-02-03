"""
Full fine-tuning of Qwen3-1.7B on Wordle SFT data.
Works on a single A100 (40GB or 80GB).

Usage:
    python train_sft_full.py
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ============ Config ============
MODEL_NAME = "PrimeIntellect/Qwen3-1.7B"  # Their clone with multi-turn chat template
DATASET_NAME = "willcb/V3-wordle"
OUTPUT_DIR = "outputs/wordle_sft_full"
MAX_SEQ_LENGTH = 1024
MAX_STEPS = 20  # Same as PrimeIntellect
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 64  # Total batch = 64

# ============ Load Model & Tokenizer ============
print(f"Loading model: {MODEL_NAME}")
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
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")


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
    max_length=MAX_SEQ_LENGTH,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=2,
    logging_steps=1,
    save_steps=MAX_STEPS,
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
