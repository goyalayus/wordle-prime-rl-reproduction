#!/usr/bin/env python3
"""
Compare all 3 Wordle SFT models on the same prompts.
Saves full responses (no truncation) to JSON files for viewer.html.
"""
import argparse
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same prompts and logic as test_wordle_detailed.py
SYSTEM_PROMPT = """You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags."""

WORDLE_PROMPTS = [
    {
        "name": "Turn 1 - First Guess",
        "turn": 1,
        "prompt": """You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be:
- G (Green): Correct letter in the correct position
- Y (Yellow): Correct letter in the wrong position
- B (Black): Letter not in the word

This is turn 1. Make your first guess.""",
    },
    {
        "name": "Turn 2 - After CRANE feedback",
        "turn": 2,
        "prompt": """You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be:
- G (Green): Correct letter in the correct position
- Y (Yellow): Correct letter in the wrong position
- B (Black): Letter not in the word

Turn 1: You guessed [crane]
Feedback:
B B Y B Y

This is turn 2. Make your next guess based on the feedback.""",
    },
    {
        "name": "Turn 3 - After ALERT feedback",
        "turn": 3,
        "prompt": """You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be:
- G (Green): Correct letter in the correct position
- Y (Yellow): Correct letter in the wrong position
- B (Black): Letter not in the word

Turn 1: You guessed [crane]
Feedback:
B B Y B Y

Turn 2: You guessed [alert]
Feedback:
G B Y B B

This is turn 3. Make your next guess based on the feedback.""",
    },
    {
        "name": "Turn 4 - Getting closer",
        "turn": 4,
        "prompt": """You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be:
- G (Green): Correct letter in the correct position
- Y (Yellow): Correct letter in the wrong position
- B (Black): Letter not in the word

Turn 1: You guessed [crane]
Feedback: B B Y B Y

Turn 2: You guessed [alert]
Feedback: G B Y B B

Turn 3: You guessed [amaze]
Feedback: G B G B Y

This is turn 4. Make your next guess based on the feedback.""",
    },
]

# Default models from MODELS.md (Hugging Face)
DEFAULT_MODELS = [
    ("goyalayus/wordle-primeintellect-1.7b", "results_primeintellect_1p7b.json"),
    ("goyalayus/wordle-qwen-1.7b", "results_qwen_1p7b.json"),
    ("goyalayus/wordle-qwen-0.6b", "results_qwen_0p6b.json"),
]

# No truncation - allow full verbose <think> responses
MAX_NEW_TOKENS = 32768


def extract_guess(response):
    import re

    match = re.search(r"<guess>\s*\[?(\w+)\]?\s*</guess>", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match = re.search(r"\[([a-zA-Z]{5})\]", response)
    if match:
        return match.group(1).lower()
    return None


def test_model(model_path, output_json, max_new_tokens=MAX_NEW_TOKENS):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on: {model.device}")
    print(f"max_new_tokens = {max_new_tokens} (no truncation)")

    results = {
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": max_new_tokens,
        "tests": [],
    }

    for test_case in WORDLE_PROMPTS:
        print(f"\n  Running: {test_case['name']}...")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_case["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        extracted_guess = extract_guess(response)

        test_result = {
            "name": test_case["name"],
            "turn": test_case["turn"],
            "prompt": test_case["prompt"],
            "response": response,
            "extracted_guess": extracted_guess,
            "has_think_tag": "<think>" in response.lower(),
            "has_guess_tag": "<guess>" in response.lower(),
            "response_length": len(response),
        }
        results["tests"].append(test_result)
        print(f"    Extracted: {extracted_guess} | len={len(response)} chars")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Wordle SFT models")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save JSON files (default: current dir)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max new tokens per response (default: {MAX_NEW_TOKENS}, no truncation)",
    )
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("Wordle Model Comparison (no truncation)")
    print("=" * 50)

    for model_id, filename in DEFAULT_MODELS:
        output_path = os.path.join(args.output_dir, filename)
        print(f"\n--- {model_id} ---")
        test_model(model_id, output_path, max_new_tokens=args.max_tokens)

    print("\n" + "=" * 50)
    print("Done! Results:")
    for _, filename in DEFAULT_MODELS:
        print(f"  - {os.path.join(args.output_dir, filename)}")
    print("\nOpen viewer.html locally to compare.")


if __name__ == "__main__":
    main()
