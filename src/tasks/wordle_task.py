import json
import random
import re
from typing import Any, Dict, List, Sequence, Tuple
from datasets import Dataset

# ==============================================================================
# PROMPTS
# ==============================================================================

SYSTEM_PROMPT = (
    "You are an expert AI playing Wordle.\n"
    "GOAL: Guess the secret 5-letter word in 6 tries.\n\n"
    "GAME RULES:\n"
    "1. You must input a valid 5-letter English word.\n"
    "2. Feedback is given for each letter:\n"
    "   - G (Green): The letter is in the word and in the CORRECT position.\n"
    "   - Y (Yellow): The letter is in the word but in the WRONG position.\n"
    "   - X (Gray): The letter is NOT in the word (or no extra copies exist).\n\n"
    "FORMATTING:\n"
    "First, think step-by-step inside <think>...</think> tags.\n"
    "Then, output your guess inside <guess>[word]</guess> tags.\n"
)

USER_PROMPT_PREFIX = (
    "You are Player 0 in Wordle.\n"
    "A secret 5-letter word has been chosen. You have 6 attempts to guess it.\n"
    "For each guess, wrap your word in square brackets (e.g., [apple]).\n"
    "Feedback for each letter will be given as follows:\n"
    "  - G (green): correct letter in the correct position\n"
    "  - Y (yellow): letter exists in the word but in the wrong position\n"
    "  - X (wrong): letter is not in the word\n"
)

STRICT_FORMAT_RE = re.compile(
    r"</think>.*?<guess>\s*\[([a-zA-Z]{5})\]\s*</guess>",
    re.IGNORECASE | re.DOTALL,
)


def render_user_prompt(history_rows: Sequence[Tuple[str, str]]) -> str:
    turn_idx = len(history_rows) + 1
    attempts_left = max(0, 6 - turn_idx)
    lines = [f"Turn {i}: [{guess}] -> {feedback}" for i, (guess, feedback) in enumerate(history_rows, start=1)]
    history_block = "\n".join(lines)
    return (
        USER_PROMPT_PREFIX
        + f"\nThis is turn {turn_idx} of the game. You have {attempts_left} attempts left.\n\n"
        + "Prior turns and feedback:\n"
        + history_block
        + "\n\n"
        + "Enter your next guess."
    )


def build_prompt_text(history_rows: Sequence[Tuple[str, str]]) -> str:
    # Force the model to begin its completion inside a <think> block.
    return SYSTEM_PROMPT + "\n\n" + render_user_prompt(history_rows=history_rows) + "\n\n<think>"


# ==============================================================================
# DATASET GENERATION
# ==============================================================================

def get_wordlists(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    paths = config.get("task", {}).get("wordlists", {})
    # Provide default fallback paths based on the codebase structure
    base_dir = Path(__file__).parent.parent.parent / "qwen4b-lora-wordle" / "RL" / "wordlists"
    
    sol_path = paths.get("solutions", str(base_dir / "wordle_solutions.txt"))
    allow_path = paths.get("allowed", str(base_dir / "wordle_allowed_guesses.txt"))
    
    def load(p):
        with open(p) as f:
            return [w.strip().lower() for w in f if len(w.strip()) == 5]
    
    return load(sol_path), load(allow_path)


def generate_synthetic_wordle_dataset(config: Dict[str, Any], seed: int, n_samples: int) -> Dataset:
    solutions, allowed = get_wordlists(config)
    valid_words = sorted(list(set(solutions) | set(allowed)))
    
    rng = random.Random(seed)
    task_cfg = config.get("task", {})
    min_turns = task_cfg.get("min_history_turns", 1)
    max_turns = task_cfg.get("max_history_turns", 4)
    
    rows = []
    
    # Placeholder logic - in a real implementation this would generate actual history states
    # using the true wordle engine / feedback compute function
    for i in range(n_samples):
        secret = rng.choice(solutions)
        history_len = rng.randint(min_turns, max_turns)
        
        # Mocking history generation for the sake of the plugin example
        history = [("crane", "X X G X X")] * history_len
        
        prompt_text = build_prompt_text(history)
        
        rows.append({
            "id": i,
            "secret": secret,
            "history_len": history_len,
            "prompt": prompt_text,
        })
        
    return Dataset.from_list(rows)


# ==============================================================================
# SFT PLUGIN HOOKS
# ==============================================================================

def get_sft_dataset(config: Dict[str, Any], tokenizer: Any) -> Dataset:
    """Returns the training dataset for SFT."""
    from datasets import load_dataset
    dataset_name = config.get("task", {}).get("sft_dataset", "goyalayus/wordle-reasoning-sft-prefix-keep-think")
    
    # Example logic: load a dataset with a 'messages' column
    ds = load_dataset(dataset_name, split="train")
    
    # Val split if requested
    val_size = config.get("training", {}).get("val_size", 0)
    if val_size > 0:
        ds = ds.train_test_split(test_size=val_size)["train"]
        
    return ds


def get_sft_eval_dataset(config: Dict[str, Any], tokenizer: Any) -> Dataset:
    """Returns the evaluation dataset for SFT."""
    from datasets import load_dataset
    dataset_name = config.get("task", {}).get("sft_dataset", "goyalayus/wordle-reasoning-sft-prefix-keep-think")
    
    ds = load_dataset(dataset_name, split="train")
    val_size = config.get("training", {}).get("val_size", 0)
    if val_size > 0:
        return ds.train_test_split(test_size=val_size)["test"]
    return None


def get_sft_formatting_func(config: Dict[str, Any], tokenizer: Any):
    """Applies the chat template during SFT."""
    def format_conversation(example):
        messages = example["messages"]
        if messages and isinstance(messages[0], list):
            return [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in messages
            ]
        return [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ]
    return format_conversation


# ==============================================================================
# RL PLUGIN HOOKS
# ==============================================================================

def get_rl_dataset(config: Dict[str, Any]) -> Dataset:
    """Returns the dataset for GRPO training (just prompts)."""
    task_cfg = config.get("task", {})
    n_samples = task_cfg.get("synthetic_samples", 1024)
    seed = config.get("training", {}).get("seed", 3407)
    
    return generate_synthetic_wordle_dataset(config, seed, n_samples)


def get_reward_funcs(config: Dict[str, Any], tokenizer: Any) -> List[Any]:
    """Returns a list of callable reward functions for GRPO."""
    
    rewards_cfg = config.get("task", {}).get("rewards", {})
    format_reward = float(rewards_cfg.get("format", 0.2))
    
    # Generic format reward
    def reward_format_exact(prompts: Sequence[Any], completions: Sequence[Any], **kwargs) -> List[float]:
        rewards = []
        for comp in completions:
            text = comp.get("content", str(comp)) if isinstance(comp, dict) else str(comp)
            m = STRICT_FORMAT_RE.search(text)
            rewards.append(format_reward if m else 0.0)
        return rewards

    # Return list of reward functions. Add Wordle logic here.
    return [reward_format_exact]


# ==============================================================================
# EVALS PLUGIN HOOKS
# ==============================================================================

def get_eval_dataset(config: Dict[str, Any]) -> Dataset:
    """Returns the dataset for the Evals run."""
    task_cfg = config.get("task", {})
    n_samples = task_cfg.get("eval_samples", 100)
    seed = task_cfg.get("eval_seed", 42)
    
    return generate_synthetic_wordle_dataset(config, seed, n_samples)


def compute_metrics(prompts: List[str], completions: List[str], dataset_rows: Dataset, config: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    """Scores completions and returns an aggregated metric dictionary."""
    valid = 0
    total = len(completions)
    
    for comp in completions:
        if STRICT_FORMAT_RE.search(comp):
            valid += 1
            
    return {
        "metrics": {
            "format_accuracy": valid / max(total, 1),
            "total_samples": total
        }
    }
