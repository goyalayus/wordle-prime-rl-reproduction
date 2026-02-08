#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordle_gameplay import (
    DEFAULT_SYSTEM_PROMPT,
    GenerationConfig,
    run_gameplay_eval,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Gameplay eval: play Wordle against 5 secret words")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--secret-words",
        default="crane,alert,amaze,plane,store",
        help="Comma-separated list of 5-letter secret words",
    )
    p.add_argument("--output-dir", required=True, help="Directory to write results")
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


def main():
    args = parse_args()
    secret_words: List[str] = [w.strip().lower() for w in args.secret_words.split(",") if w.strip()]
    if not secret_words:
        raise SystemExit("No secret words provided")

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded on: {model.device}")

    gen = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )
    results = run_gameplay_eval(
        model=model,
        tokenizer=tokenizer,
        secret_words=secret_words,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_turns=args.max_turns,
        gen=gen,
        seed=args.seed,
    )
    results["timestamp"] = datetime.now().isoformat()
    results["model"] = args.model

    out_path = os.path.join(args.output_dir, "gameplay.json")
    write_json(out_path, results)

    summary_path = os.path.join(args.output_dir, "summary.json")
    write_json(summary_path, results["summary"])

    print("Done.")
    print(f"- {out_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
