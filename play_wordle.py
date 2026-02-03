#!/usr/bin/env python3
"""
Play Wordle with the SFT-trained model. Runs 4-5 example words to evaluate performance.
"""
import argparse
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


SYSTEM_PROMPT = """You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step, then give your guess inside <guess>...</guess> tags (e.g. <guess>[apple]</guess>)."""

GAME_INSTRUCTIONS = """You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be given as follows:
 - G (green): correct letter in the correct position
 - Y (yellow): letter exists in the word but in the wrong position
 - X (wrong): letter is not in the word
Enter your guess to begin.
"""


def compute_feedback(guess: str, target: str) -> str:
    """Compute Wordle feedback: G=green, Y=yellow, X=wrong."""
    guess = guess.lower()[:5]
    target = target.lower()[:5]
    result = ["X"] * 5
    target_counts = {}
    for c in target:
        target_counts[c] = target_counts.get(c, 0) + 1

    # First pass: greens
    for i in range(5):
        if i < len(guess) and guess[i] == target[i]:
            result[i] = "G"
            target_counts[target[i]] -= 1

    # Second pass: yellows
    for i in range(5):
        if result[i] == "G":
            continue
        if i < len(guess) and guess[i] in target_counts and target_counts[guess[i]] > 0:
            result[i] = "Y"
            target_counts[guess[i]] -= 1

    return " ".join(result)


def extract_guess(response: str) -> str | None:
    """Extract guess from <<guess>>[word]<</guess>> or <guess>word</guess>."""
    # Try <<guess>>[word]<</guess>>
    m = re.search(r"<<guess>>\s*\[([a-zA-Z]{5})\]\s*<<\/guess>>", response, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Try <guess>[word]</guess>
    m = re.search(r"<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>", response, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Try <guess>word</guess>
    m = re.search(r"<guess>\s*([a-zA-Z]{5})\s*</guess>", response, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Try [word] as last resort
    m = re.search(r"\[([a-zA-Z]{5})\]", response)
    if m:
        return m.group(1).lower()
    return None


def play_one_game(model, tokenizer, target_word: str, max_guesses: int = 6, max_new_tokens: int = 512) -> dict:
    """Play one Wordle game. Returns dict with win, num_guesses, history."""
    target = target_word.lower()[:5]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GAME_INSTRUCTIONS},
    ]
    history = []
    won = False

    for turn in range(max_guesses):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        guess = extract_guess(response)

        if guess is None:
            history.append(
                {
                    "guess": "(no valid guess)",
                    "feedback": "X X X X X",
                    "response": response,
                    "turn": turn + 1,
                }
            )
            messages.append({"role": "assistant", "content": response})
            feedback_str = "X X X X X"
            messages.append({"role": "user", "content": f"\n(no valid guess)\n{feedback_str}\n\nYou have {max_guesses - turn - 1} guesses left."})
            continue

        feedback = compute_feedback(guess, target)
        won = guess == target
        remaining = max_guesses - turn - 1

        history.append(
            {
                "guess": guess,
                "feedback": feedback,
                "turn": turn + 1,
                "response": response,
            }
        )

        feedback_msg = f"\n{' '.join(guess.upper())}\n{feedback}\n\nYou have {remaining} guesses left."
        if won:
            feedback_msg += "\n[GAME] Player 0 won the game. Reason: Player 0 guessed the word correctly!"
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": feedback_msg})

        if won:
            break

    return {"won": won, "num_guesses": len(history), "target": target, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PrimeIntellect/Qwen3-1.7B")
    parser.add_argument("--lora-path", default="outputs/wordle_sft_qwen3_1p7b")
    parser.add_argument("--words", nargs="+", default=["plane", "store", "trace", "price", "crane"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-4bit", action="store_true", help="Load in 4-bit (for shared GPU / low VRAM)")
    args = parser.parse_args()

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = args.load_4bit and HAS_BITSANDBYTES and torch.cuda.is_available()
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("Loaded in 4-bit (low VRAM mode)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
    lora_path = Path(args.lora_path)
    if (lora_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(lora_path))
        print("Loaded LoRA adapters from", lora_path)
    else:
        print("WARNING: No LoRA at", lora_path, "- running base model (not SFT)")
    model.eval()

    print("\n" + "=" * 60)
    print("Playing Wordle with SFT model")
    print("=" * 60)

    wins = 0
    for word in args.words:
        print(f"\n--- Target word: {word.upper()} ---")
        result = play_one_game(model, tokenizer, word, max_new_tokens=args.max_new_tokens)
        for h in result["history"]:
            print(f"  Turn {h.get('turn', '?')}:")
            print(h.get("response", "").strip() or "(empty)")
        status = "WIN" if result["won"] else "LOSS"
        print(f"  Result: {status} ({result['num_guesses']} guesses)")
        if result["won"]:
            wins += 1

    print("\n" + "=" * 60)
    print(f"Summary: {wins}/{len(args.words)} games won ({100*wins/len(args.words):.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
