# Inference branch

Run inference on 10 LoRA checkpoints from Hugging Face (`goyalayus/wordle-lora-qwen06b`).

**Checkpoints:** step-40, step-80, step-120, step-160, step-200, step-240, step-280, step-320, step-360, step-400

**Per checkpoint:** 3 full Wordle games (words: corny, align, sober). No max token cap on model output.

**Output:** `inference_results.json` — structured for later HTML comparison.

## Run on T4 GPU

```bash
pip install -r requirements-inference.txt
# Ensure HF_TOKEN in .env or env
python run_inference.py
# or: bash run_inference.sh
```

## JSON structure

```json
{
  "step-40": [
    {"target_word": "corny", "won": true, "turns": [...], "num_turns": 3},
    ...
  ],
  "step-80": [...],
  ...
}
```

Each turn includes `assistant_response` (full model output) and `extracted_guess`.
