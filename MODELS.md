# Wordle SFT Models

Trained on `willcb/V3-wordle` for 20 steps. Use with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("goyalayus/wordle-primeintellect-1.7b")
tokenizer = AutoTokenizer.from_pretrained("goyalayus/wordle-primeintellect-1.7b")
```

## Model Locations

| Model | Hugging Face | Base |
|-------|--------------|------|
| Prime Intellect 1.7B | [goyalayus/wordle-primeintellect-1.7b](https://huggingface.co/goyalayus/wordle-primeintellect-1.7b) | PrimeIntellect/Qwen3-1.7B |
| Qwen 1.7B | [goyalayus/wordle-qwen-1.7b](https://huggingface.co/goyalayus/wordle-qwen-1.7b) | Qwen/Qwen3-1.7B |
| Qwen 0.6B | [goyalayus/wordle-qwen-0.6b](https://huggingface.co/goyalayus/wordle-qwen-0.6b) | Qwen/Qwen3-0.6B |
