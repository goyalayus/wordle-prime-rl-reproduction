#!/usr/bin/env python3
"""Upload Wordle SFT models to Hugging Face Hub."""
import os
from huggingface_hub import HfApi, login

# Set your HF username - change if different
HF_USER = os.environ.get("HF_USER", "goyalayus")

MODELS = [
    ("outputs/wordle_sft/PrimeIntellect-Qwen3-1.7B/final", f"{HF_USER}/wordle-primeintellect-1.7b"),
    ("outputs/wordle_sft/Qwen-Qwen3-1.7B/final", f"{HF_USER}/wordle-qwen-1.7b"),
    ("outputs/wordle_sft/Qwen-Qwen3-0.6B/final", f"{HF_USER}/wordle-qwen-0.6b"),
]

def main():
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("ERROR: Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable.")
        print("Get your token from https://huggingface.co/settings/tokens")
        return 1

    api = HfApi()
    for local_path, repo_id in MODELS:
        if not os.path.exists(local_path):
            print(f"SKIP {repo_id}: {local_path} not found")
            continue
        print(f"Creating repo {repo_id} (if needed)...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Uploading {local_path} -> {repo_id}...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Done: https://huggingface.co/{repo_id}")
    print("All uploads complete.")
    return 0

if __name__ == "__main__":
    exit(main())
