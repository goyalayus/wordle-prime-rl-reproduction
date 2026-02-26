#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import Dataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    parser = argparse.ArgumentParser(description="Generic Unsloth GRPO RL Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--task-module", type=str, required=True, help="Path to task plugin python file")
    return parser.parse_args()


def build_vllm_sampling_params(tokenizer: Any, config: Dict[str, Any]):
    vllm_cfg = config.get("vllm", {})
    if not vllm_cfg.get("enabled", False):
        return None
    from vllm import SamplingParams

    max_tokens = int(vllm_cfg.get("max_tokens", 2048))
    return SamplingParams(
        temperature=float(vllm_cfg.get("temperature", 1.0)),
        min_p=float(vllm_cfg.get("min_p", 0.1)),
        top_p=float(vllm_cfg.get("top_p", 1.0)),
        top_k=int(vllm_cfg.get("top_k", -1)),
        seed=int(config.get("training", {}).get("seed", 3407)),
        max_tokens=max_tokens,
        stop=[tokenizer.eos_token] if getattr(tokenizer, "eos_token", None) is not None else None,
        include_stop_str_in_output=True,
    )


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    task = load_task_module(args.task_module)

    # Validate task module
    if not hasattr(task, "get_rl_dataset"):
        raise AttributeError(f"Task module {args.task_module} must implement 'get_rl_dataset(config)'")
    if not hasattr(task, "get_reward_funcs"):
        raise AttributeError(f"Task module {args.task_module} must implement 'get_reward_funcs(config)'")

    print("Loading model and tokenizer via Unsloth...")
    # Import Unsloth and TRL inside main to allow fast arg parsing
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    vllm_cfg = config.get("vllm", {})

    model_name = model_cfg.get("name", "Qwen/Qwen3-4B")
    seq_len = model_cfg.get("max_seq_length", 4096)
    load_in_4bit = model_cfg.get("load_in_4bit", True)
    fast_inference = vllm_cfg.get("enabled", False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=seq_len,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=lora_cfg.get("r", 16),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        target_modules=lora_cfg.get("target_modules", ["up_proj", "gate_proj", "down_proj"]),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("gradient_checkpointing", "unsloth"),
        random_state=train_cfg.get("seed", 3407),
    )

    # Optional: init adapter loading
    init_repo = model_cfg.get("init_adapter_repo")
    if init_repo:
        from safetensors.torch import load_file as safetensors_load_file
        from huggingface_hub import hf_hub_download
        print(f"Loading initial adapter from {init_repo}...")
        init_rev = model_cfg.get("init_adapter_revision", "main")
        try:
            adapter_weights_path = hf_hub_download(
                repo_id=init_repo,
                filename="adapter_model.safetensors",
                revision=init_rev,
            )
            adapter_state = safetensors_load_file(adapter_weights_path)
            
            # Key mapping logic as handled in previous RL scripts
            def _maybe_default(k: str) -> str:
                if ".lora_A.weight" in k: return k.replace(".lora_A.weight", ".lora_A.default.weight")
                if ".lora_B.weight" in k: return k.replace(".lora_B.weight", ".lora_B.default.weight")
                return k

            model_state = model.state_dict()
            mapped = {(_maybe_default(k)): v for k, v in adapter_state.items()}
            mapped_state = {k: v for k, v in mapped.items() if k in model_state}
            model.load_state_dict(mapped_state, strict=False)
            print("Successfully loaded init adapter.")
        except Exception as e:
            print(f"Warning: Failed to load init adapter from {init_repo}: {e}")

    model.train()

    print("Loading dataset from task module...")
    dataset = task.get_rl_dataset(config)
    print(f"Loaded {len(dataset)} examples.")

    print("Loading reward functions from task module...")
    reward_funcs = task.get_reward_funcs(config, tokenizer)

    run_name = train_cfg.get("run_name", f"rl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = Path(train_cfg.get("output_dir", f"./outputs/{run_name}"))
    ensure_dir(output_dir)

    # Dump config for reproducibility
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    vllm_sampling_params = build_vllm_sampling_params(tokenizer, config)

    # Setup Trainer
    cfg_kwargs = dict(
        output_dir=str(output_dir / "trainer_out"),
        max_steps=train_cfg.get("max_steps", 1000),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        temperature=float(vllm_cfg.get("temperature", 1.0)) if vllm_cfg.get("enabled", False) else 1.0,
        use_vllm=fast_inference,
        vllm_sampling_params=vllm_sampling_params,
        optim=train_cfg.get("optim", "adamw_8bit"),
        logging_steps=1,
        report_to=train_cfg.get("report_to", ["none"]),
        run_name=run_name,
        seed=train_cfg.get("seed", 3407),
        remove_unused_columns=False,
        num_generations=train_cfg.get("num_generations", 4),
        max_prompt_length=train_cfg.get("max_prompt_length", 2048),
        max_completion_length=train_cfg.get("max_completion_length", 2048),
        bf16=False,
        fp16=False,
    )

    import inspect
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    grpo_args = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in accepted})

    callbacks = getattr(task, "get_rl_callbacks", lambda cfg, tkz, pth: [])(config, tokenizer, output_dir)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainable_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_count}")

    print("Starting GRPO training...")
    trainer.train()

    hf_repo_id = train_cfg.get("hf_repo_id")
    if hf_repo_id:
        print(f"Pushing final model to Hub: {hf_repo_id}")
        trainer.model.push_to_hub(hf_repo_id)

    print("Done!")

if __name__ == "__main__":
    main()
