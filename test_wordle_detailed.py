import argparse
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "outputs/wordle_sft_full/final"
DEFAULT_OUTPUT_JSON = "wordle_test_results.json"

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

This is turn 1. Make your first guess."""
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

This is turn 2. Make your next guess based on the feedback."""
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

This is turn 3. Make your next guess based on the feedback."""
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

This is turn 4. Make your next guess based on the feedback."""
    },
]


def extract_guess(response):
    """Extract guess from response"""
    import re
    # Try <guess>...</guess>
    match = re.search(r'<guess>\s*\[?(\w+)\]?\s*</guess>', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Try [word] pattern
    match = re.search(r'\[([a-zA-Z]{5})\]', response)
    if match:
        return match.group(1).lower()
    return None


def test_model(model_path, output_json):
    print("Loading model from:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on: {model.device}")

    # Explicit high limit - no truncation. Model can be verbose in <think>.
    max_new_tokens = 8192
    print(f"max_new_tokens = {max_new_tokens} (no truncation)")

    results = {
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": max_new_tokens,
        "tests": []
    }

    for test_case in WORDLE_PROMPTS:
        print(f"\nRunning: {test_case['name']}...")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_case["prompt"]}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
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
        print(f"  Extracted guess: {extracted_guess}")
        print(f"  Has <think>: {test_result['has_think_tag']}")
        print(f"  Has <guess>: {test_result['has_guess_tag']}")
        print(f"  Response length: {len(response)} chars")

    # Save results
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Test Wordle SFT model")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to model")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSON, help="Output JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_model(args.model, args.output)
