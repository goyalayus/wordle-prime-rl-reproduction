#!/usr/bin/env python3
"""Upload Wordle SFT models to Hugging Face Hub."""
import argparse
import os
from huggingface_hub import HfApi, login

# Set your HF username - change if different
HF_USER = os.environ.get("HF_USER", "goyalayus")

MODELS = [
    ("outputs/wordle_sft/PrimeIntellect-Qwen3-1.7B/final", f"{HF_USER}/wordle-primeintellect-1.7b"),
    ("outputs/wordle_sft/Qwen-Qwen3-1.7B/final", f"{HF_USER}/wordle-qwen-1.7b"),
    ("outputs/wordle_sft/Qwen-Qwen3-0.6B/final", f"{HF_USER}/wordle-qwen-0.6b"),
]

def parse_args():
    p = argparse.ArgumentParser(description="Upload a trained model folder to Hugging Face Hub")
    p.add_argument(
        "--local-path",
        default=None,
        help="Local folder containing model weights/tokenizer (e.g. outputs/.../final). If omitted, uploads the default MODELS list.",
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="Target repo id (e.g. yourname/wordle-qwen-0.6b-long). Required if --local-path is set.",
    )
    p.add_argument(
        "--repo-type",
        default="model",
        choices=["model"],
        help="Repo type (only 'model' is used in this repo).",
    )
    return p.parse_args()


def main():
    # Load .env if present so HF_TOKEN/WANDB_API_KEY can be set without exporting manually.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("ERROR: Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable.")
        print("Get your token from https://huggingface.co/settings/tokens")
        return 1

    args = parse_args()

    api = HfApi()
    models = MODELS
    if args.local_path is not None:
        if args.repo_id is None:
            print("ERROR: --repo-id is required when --local-path is provided.")
            return 1
        models = [(args.local_path, args.repo_id)]

    for local_path, repo_id in models:
        if not os.path.exists(local_path):
            print(f"SKIP {repo_id}: {local_path} not found")
            continue
        print(f"Creating repo {repo_id} (if needed)...")
        api.create_repo(repo_id=repo_id, repo_type=args.repo_type, exist_ok=True)
        print(f"Uploading {local_path} -> {repo_id}...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type=args.repo_type,
        )
        print(f"  Done: https://huggingface.co/{repo_id}")
    print("All uploads complete.")
    return 0

if __name__ == "__main__":
    exit(main())
