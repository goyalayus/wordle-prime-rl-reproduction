from __future__ import annotations

import dataclasses
import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


DEFAULT_SYSTEM_PROMPT = """You are a competitive Wordle player.

Rules:
- The secret word is 5 letters.
- You have up to 6 guesses.
- After each guess you receive feedback per letter:
  - G (green): correct letter in the correct position
  - Y (yellow): correct letter in the wrong position
  - B (black): letter not in the word (or already accounted for)

Output format requirement:
- Put your guess inside <guess>...</guess> tags, and inside the tags wrap it in [brackets], e.g. <guess>[crane]</guess>.
"""


def extract_guess(text: str) -> Optional[str]:
    """
    Extract a 5-letter guess from model output.

    Supported formats:
    - <guess>[crane]</guess>
    - <guess>crane</guess>
    - [crane]
    """
    # Prefer <guess>...</guess> so we don't match stray bracketed examples.
    m = re.search(r"<guess>\s*\[?([a-zA-Z]{5})\]?\s*</guess>", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\[([a-zA-Z]{5})\]", text)
    if m:
        return m.group(1).lower()
    return None


def compute_feedback(secret: str, guess: str) -> List[str]:
    """
    Standard Wordle feedback (G/Y/B) with duplicate-letter handling.

    - First mark greens.
    - For remaining letters, mark yellows if present in secret count remainder.
    """
    secret = secret.lower()
    guess = guess.lower()
    if len(secret) != 5 or len(guess) != 5:
        raise ValueError("secret and guess must both be 5 letters")

    feedback = ["B"] * 5
    secret_remaining: Dict[str, int] = {}

    for i in range(5):
        if guess[i] == secret[i]:
            feedback[i] = "G"
        else:
            secret_remaining[secret[i]] = secret_remaining.get(secret[i], 0) + 1

    for i in range(5):
        if feedback[i] == "G":
            continue
        ch = guess[i]
        if secret_remaining.get(ch, 0) > 0:
            feedback[i] = "Y"
            secret_remaining[ch] -= 1

    return feedback


def _format_turn_history(turns: Sequence[Dict[str, Any]]) -> str:
    if not turns:
        return "Turn 0: (no guesses yet)\n"
    lines: List[str] = []
    for t in turns:
        guess = t.get("guess") or "(no guess)"
        lines.append(f"Turn {t['turn']}: You guessed [{guess}]")
        feedback = t.get("feedback")
        lines.append("Feedback: " + (" ".join(feedback) if feedback else "N/A"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_user_prompt(secret_word: str, turns: Sequence[Dict[str, Any]], max_turns: int) -> str:
    # We do not reveal the secret word to the model; it's passed only so the caller can compute feedback.
    _ = secret_word
    turn_num = len(turns) + 1
    history = _format_turn_history(turns)
    return (
        "You are Player 0 in Wordle.\n"
        "A secret 5-letter word has been chosen. You have 6 attempts to guess it.\n"
        "For each guess, wrap your word in square brackets (e.g., [apple]).\n"
        "You must output exactly one guess using <guess>...</guess> tags.\n"
        "\n"
        f"{history}"
        f"This is turn {turn_num} of {max_turns}. Make your next guess based on the feedback.\n"
    )


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    # Intentionally high default to allow long reasoning; we stop once </guess> is emitted.
    max_new_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        self._stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if not self._stop_ids:
            return False
        if input_ids.shape[-1] < len(self._stop_ids):
            return False
        tail = input_ids[0, -len(self._stop_ids) :].tolist()
        return tail == self._stop_ids


def play_game(
    *,
    model: Any,
    tokenizer: Any,
    secret_word: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_turns: int = 6,
    gen: GenerationConfig = GenerationConfig(),
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Play a single Wordle game, returning a full transcript + summary.

    This function expects a chat model/tokenizer supporting `apply_chat_template`.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    secret_word = secret_word.lower()
    turns: List[Dict[str, Any]] = []
    invalid_guess_count = 0
    t0 = time.time()

    model_was_training = getattr(model, "training", False)
    model.eval()
    try:
        stop_ids = tokenizer.encode("</guess>", add_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([StopOnTokenSequence(stop_ids)])

        for _ in range(max_turns):
            user_prompt = build_user_prompt(secret_word, turns, max_turns=max_turns)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen.max_new_tokens,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    do_sample=gen.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=stopping_criteria,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            guess = extract_guess(response)

            turn_num = len(turns) + 1
            if guess is None:
                invalid_guess_count += 1
                turns.append(
                    {
                        "turn": turn_num,
                        "guess": None,
                        "feedback": None,
                        "raw_response": response,
                        "error": "could_not_extract_guess",
                    }
                )
                continue

            if not re.fullmatch(r"[a-z]{5}", guess):
                invalid_guess_count += 1
                turns.append(
                    {
                        "turn": turn_num,
                        "guess": guess,
                        "feedback": None,
                        "raw_response": response,
                        "error": "invalid_guess_format",
                    }
                )
                continue

            feedback = compute_feedback(secret_word, guess)
            turns.append(
                {
                    "turn": turn_num,
                    "guess": guess,
                    "feedback": feedback,
                    "raw_response": response,
                    "error": None,
                }
            )

            if guess == secret_word:
                break
    finally:
        if model_was_training:
            model.train()

    solved = any(t.get("guess") == secret_word for t in turns)
    elapsed_s = time.time() - t0

    return {
        "secret_word": secret_word,
        "solved": solved,
        "turns_taken": len(turns),
        "invalid_guess_count": invalid_guess_count,
        "turns": turns,
        "elapsed_s": elapsed_s,
    }


def run_gameplay_eval(
    *,
    model: Any,
    tokenizer: Any,
    secret_words: Sequence[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_turns: int = 6,
    gen: GenerationConfig = GenerationConfig(),
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    games: List[Dict[str, Any]] = []
    for i, w in enumerate(secret_words):
        game_seed = None if seed is None else seed + i
        games.append(
            play_game(
                model=model,
                tokenizer=tokenizer,
                secret_word=w,
                system_prompt=system_prompt,
                max_turns=max_turns,
                gen=gen,
                seed=game_seed,
            )
        )

    solved_count = sum(1 for g in games if g["solved"])
    avg_turns = sum(g["turns_taken"] for g in games) / max(1, len(games))
    invalid_total = sum(g["invalid_guess_count"] for g in games)

    return {
        "meta": {
            "n_games": len(games),
            "secret_words": list(secret_words),
            "max_turns": max_turns,
            "generation": dataclasses.asdict(gen),
            "seed": seed,
        },
        "summary": {
            "solved_count": solved_count,
            "solved_rate": solved_count / max(1, len(games)),
            "avg_turns": avg_turns,
            "invalid_guess_total": invalid_total,
        },
        "games": games,
    }


def write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
