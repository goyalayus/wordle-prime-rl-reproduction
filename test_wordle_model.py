"""
Test the fine-tuned Wordle model on sample prompts.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "outputs/wordle_sft_full/final"

def test_wordle_response(model, tokenizer, wordle_prompt, max_new_tokens=512):
    """Test model on a Wordle game prompt."""
    
    messages = [
        {
            "role": "system",
            "content": """You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step inside <think>...</think> tags, then follow the instructions inside <guess>...</guess> tags."""
        },
        {
            "role": "user", 
            "content": wordle_prompt
        }
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
    return response


# Sample Wordle game prompts
WORDLE_PROMPTS = [
    # First turn - no feedback yet
    """You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be:
- G (Green): Correct letter in the correct position
- Y (Yellow): Correct letter in the wrong position
- B (Black): Letter not in the word

This is turn 1. Make your first guess.""",

    # Second turn with feedback
    """You are Player 0 in Wordle.
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

    # Third turn with more feedback
    """You are Player 0 in Wordle.
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
]


def main():
    print("Loading model from:", MODEL_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Model loaded on: {model.device}")
    print("=" * 60)
    
    for i, prompt in enumerate(WORDLE_PROMPTS, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: Turn {i}")
        print("=" * 60)
        print("\nPrompt (truncated):")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("\n" + "-" * 40)
        print("Model Response:")
        print("-" * 40)
        
        response = test_wordle_response(model, tokenizer, prompt)
        print(response)
        print()


if __name__ == "__main__":
    main()
