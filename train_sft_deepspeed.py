"""
Full fine-tuning of Qwen3-1.7B on Wordle SFT data using DeepSpeed ZeRO-3.
Requires 4x T4 GPUs (or similar ~64GB total VRAM).

Usage:
    deepspeed --num_gpus=4 train_sft_deepspeed.py
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ============ Config ============
MODEL_NAME = "Qwen/Qwen3-1.7B"  # Use base Qwen (same as PrimeIntellect's base)
DATASET_NAME = "willcb/V3-wordle"
OUTPUT_DIR = "outputs/wordle_sft_full"
MAX_SEQ_LENGTH = 1024
MAX_STEPS = 20  # Same as PrimeIntellect's config
LEARNING_RATE = 1e-5
BATCH_SIZE_PER_GPU = 1
GRADIENT_ACCUMULATION = 16  # Total batch = 4 GPUs * 1 * 16 = 64

# ============ Load Model & Tokenizer ============
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # FP16 for memory efficiency
    trust_remote_code=True,
    use_cache=False,  # Required for gradient checkpointing
)
model.gradient_checkpointing_enable()

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

# ============ Data Collator for Completion-Only Loss ============
# Only compute loss on assistant responses, not on prompts
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# ============ Training Config ============
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE_PER_GPU,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=2,
    logging_steps=1,
    save_steps=MAX_STEPS,
    save_total_limit=2,
    bf16=False,
    fp16=True,
    gradient_checkpointing=True,
    deepspeed="ds_config_zero3.json",
    report_to="wandb",
    run_name="wordle-sft-full-finetune",
    dataset_text_field="text",
)

# ============ Trainer ============
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=collator,
)

# ============ Train ============
print("Starting training...")
trainer.train()

# ============ Save ============
print(f"Saving model to {OUTPUT_DIR}/final")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Done!")
