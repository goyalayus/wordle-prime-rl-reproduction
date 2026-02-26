#!/usr/bin/env python3
import argparse
import importlib.util
import json
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
    parser = argparse.ArgumentParser(description="Generic Evaluation Runner (vLLM/Unsloth)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--task-module", type=str, required=True, help="Path to task plugin python file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    task = load_task_module(args.task_module)

    if not hasattr(task, "get_eval_dataset"):
        raise AttributeError(f"Task module {args.task_module} must implement 'get_eval_dataset(config)'")
    if not hasattr(task, "compute_metrics"):
        raise AttributeError(f"Task module {args.task_module} must implement 'compute_metrics(prompts, completions, dataset_rows, config, tokenizer)'")

    from unsloth import FastLanguageModel

    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    vllm_cfg = config.get("vllm", {})

    model_name = model_cfg.get("name", "Qwen/Qwen3-4B")
    seq_len = model_cfg.get("max_seq_length", 4096)

    print(f"Loading model {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=seq_len,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        fast_inference=vllm_cfg.get("enabled", True),
        max_lora_rank=model_cfg.get("lora_r", 16),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    init_repo = model_cfg.get("init_adapter_repo")
    if init_repo:
        print(f"Loading LoRA adapter from {init_repo}...")
        init_rev = model_cfg.get("init_adapter_revision", "main")
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_cfg.get("lora_r", 16),
            target_modules=model_cfg.get("lora_target_modules", ["up_proj", "gate_proj", "down_proj"]),
            lora_alpha=model_cfg.get("lora_alpha", 32),
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
        # We manually load safetensors to avoid wrapping the model
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        
        adapter_path = hf_hub_download(repo_id=init_repo, filename="adapter_model.safetensors", revision=init_rev)
        adapter_state = load_file(adapter_path)
        
        def _maybe_default(k: str) -> str:
            if ".lora_A.weight" in k: return k.replace(".lora_A.weight", ".lora_A.default.weight")
            if ".lora_B.weight" in k: return k.replace(".lora_B.weight", ".lora_B.default.weight")
            return k

        model_state = model.state_dict()
        mapped = {(_maybe_default(k)): v for k, v in adapter_state.items()}
        mapped_state = {k: v for k, v in mapped.items() if k in model_state}
        model.load_state_dict(mapped_state, strict=False)

    print("Putting model into Fast Inference mode...")
    model.train(False)
    # FastLanguageModel.for_inference enables vllm automatically if fast_inference=True was passed to from_pretrained
    model = FastLanguageModel.for_inference(model)

    print("Loading eval dataset from task module...")
    dataset = task.get_eval_dataset(config)
    print(f"Loaded {len(dataset)} examples for evaluation.")

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=float(vllm_cfg.get("temperature", 0.0)),
        top_p=float(vllm_cfg.get("top_p", 1.0)),
        max_tokens=int(vllm_cfg.get("max_tokens", 1024)),
        stop=[tokenizer.eos_token] if getattr(tokenizer, "eos_token", None) is not None else None,
    )

    prompts = [row["prompt"] for row in dataset]

    print(f"Starting batched generation with vLLM (temp={sampling_params.temperature})...")
    outputs = model.fast_generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    completions = []
    for out in outputs:
        # Assuming single generation per prompt for eval
        completions.append(out.outputs[0].text)

    print("Generation complete. Computing metrics...")
    results = task.compute_metrics(prompts, completions, dataset, config, tokenizer)

    output_dir = Path(eval_cfg.get("output_dir", "./outputs/evals"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = output_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation Results saved to {out_file}:")
    for k, v in results.get("metrics", {}).items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
