import argparse
import importlib.util
import os
import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_task_module(module_path: str):
    module_name = Path(module_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load task module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Generic Unsloth SFT Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--task-module", type=str, required=True, help="Path to task plugin python file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    task = load_task_module(args.task_module)

    # Validate task module
    if not hasattr(task, "get_sft_dataset"):
        raise AttributeError(f"Task module {args.task_module} must implement 'get_sft_dataset(config)'")

    # Import Unsloth and TRL after parsing args so we don't slow down help/error messages
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainingArguments

    # 1. Unpack Config
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})

    model_name = model_cfg.get("name", "Qwen/Qwen3-4B")
    seq_len = model_cfg.get("max_seq_length", 2048)

    print(f"Loading model {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=seq_len,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        load_in_8bit=model_cfg.get("load_in_8bit", False),
        fast_inference=model_cfg.get("fast_inference", False),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        target_modules=lora_cfg.get("target_modules", ["gate_proj", "up_proj", "down_proj"]),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("gradient_checkpointing", "unsloth"),
        random_state=train_cfg.get("seed", 3407),
        max_seq_length=seq_len,
    )

    print("Loading dataset from task module...")
    # The task module is responsible for loading the dataset and applying any chat templates
    # It must return a dataset ready for SFTTrainer
    train_dataset = task.get_sft_dataset(config, tokenizer)
    eval_dataset = None
    if hasattr(task, "get_sft_eval_dataset"):
        eval_dataset = task.get_sft_eval_dataset(config, tokenizer)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    print(f"Dataset loaded. Train size: {len(train_dataset)}")

    # formatting_func = None
    # If the task module provides a formatting function, use it
    formatting_func = getattr(task, "get_sft_formatting_func", lambda cfg, tok: None)(config, tokenizer)

    run_name = train_cfg.get("run_name", "sft_run")
    output_dir = train_cfg.get("output_dir", f"./outputs/{run_name}")

    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=seq_len,
        max_steps=train_cfg.get("max_steps", 1000),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=train_cfg.get("warmup_steps", 10),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        optim=train_cfg.get("optim", "adamw_8bit"),
        report_to=train_cfg.get("report_to", "none"),
        run_name=run_name,
        seed=train_cfg.get("seed", 3407),
        packing=train_cfg.get("packing", False),
        dataset_text_field=train_cfg.get("dataset_text_field", "text") if not formatting_func else None,
    )

    if eval_dataset:
        training_args.eval_strategy = "steps"
        training_args.eval_steps = train_cfg.get("eval_steps", 100)
        training_args.per_device_eval_batch_size = train_cfg.get("per_device_eval_batch_size", 2)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    if formatting_func:
        trainer_kwargs["formatting_func"] = formatting_func
        # Remove dataset_text_field if formatting_func is provided
        if "dataset_text_field" in trainer_kwargs:
            del trainer_kwargs["dataset_text_field"]

    # Optional: custom data collator for masking (like completion-only loss)
    if hasattr(task, "get_sft_data_collator"):
        collator = task.get_sft_data_collator(config, tokenizer)
        if collator:
            trainer_kwargs["data_collator"] = collator

    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    hf_repo_id = train_cfg.get("hf_repo_id")
    if hf_repo_id:
        print(f"Pushing to Hub: {hf_repo_id}")
        model.push_to_hub(hf_repo_id)
        tokenizer.push_to_hub(hf_repo_id)
    else:
        print("Saving locally...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
