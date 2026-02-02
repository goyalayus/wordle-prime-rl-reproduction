import argparse

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "PrimeIntellect/Qwen3-1.7B"
DEFAULT_DATASET = "willcb/V3-wordle"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = "outputs/wordle_sft_qwen3_1p7b"


def explode_to_prompt_completion(example, system_prompt=None):
    """Explode willcb/V3-wordle rows into one (prompt, completion) pair per assistant turn."""
    prompt_msgs = list(example["prompt"])
    completion_msgs = list(example["completion"])

    if system_prompt:
        if prompt_msgs and prompt_msgs[0].get("role") == "system":
            prompt_msgs[0] = {"role": "system", "content": system_prompt}
        else:
            prompt_msgs.insert(0, {"role": "system", "content": system_prompt})

    rows = []
    history = list(prompt_msgs)
    for msg in completion_msgs:
        if msg.get("role") == "assistant":
            rows.append({"prompt": list(history), "completion": [msg]})
        history.append(msg)

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Qwen3-1.7B on Wordle with TRL.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--wandb-project", default=None, help="W&B project name. If set, enables wandb logging.")
    parser.add_argument("--wandb-name", default=None, help="W&B run name.")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataset = load_dataset(args.dataset, split=args.split)

    def flat_map_fn(batch):
        all_prompts, all_completions = [], []
        for i in range(len(batch["prompt"])):
            rows = explode_to_prompt_completion(
                {"prompt": batch["prompt"][i], "completion": batch["completion"][i]},
                system_prompt=args.system_prompt,
            )
            for r in rows:
                all_prompts.append(r["prompt"])
                all_completions.append(r["completion"])
        return Dataset.from_dict({"prompt": all_prompts, "completion": all_completions})

    dataset = dataset.flat_map(flat_map_fn)

    report_to = "wandb" if args.wandb_project else "none"
    training_kwargs = dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        max_length=args.seq_len,
        completion_only_loss=True,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        lr_scheduler_type="linear",
        weight_decay=0.0,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        report_to=report_to,
    )
    if args.wandb_project:
        training_kwargs["project"] = args.wandb_project
    if args.wandb_name:
        training_kwargs["run_name"] = args.wandb_name
    training_args = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        packing=False,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
