import argparse
import os
import subprocess
import sys

try:
    import modal
except ImportError:
    print("Modal SDK not found. Please install: pip install modal")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Unsloth scripts on Modal",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--script", required=True, help="Path to runner script (e.g., src/runners/sft_run.py)")
    parser.add_argument("--gpu", default="H100", choices=["A10G", "A100", "A100-80GB", "H100"], help="GPU type")
    parser.add_argument("--timeout", type=int, default=86400, help="Timeout in seconds (default 24h)")
    parser.add_argument("--profile", default=None, help="Modal profile to use (overrides MODAL_PROFILE env var)")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script")
    return parser.parse_args()


# Set up the modal App
app = modal.App("wordle-unsloth-runner")


# Define the Docker image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "vllm==0.7.3",
        "huggingface_hub",
        "hf_transfer",
        "wandb",
        "python-dotenv",
        "datasets",
        "pyyaml",
    )
    # Install Unsloth & TRL from source for latest fixes
    .run_commands(
        "pip install unsloth unsloth-zoo",
        "pip install git+https://github.com/huggingface/trl.git",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Read HF / W&B secrets if they exist on the modal account
# Alternatively, they can be passed as .env files using mounts
secrets = []
try:
    secrets.append(modal.Secret.from_name("huggingface-secret"))
except modal.exception.NotFoundError:
    pass

try:
    secrets.append(modal.Secret.from_name("wandb-secret"))
except modal.exception.NotFoundError:
    pass


@app.function(
    image=image,
    gpu=None,  # Will be overridden in local_entrypoint
    timeout=86400,  # Will be overridden
    secrets=secrets,
    # Mount the whole repository root to /workspace
    mounts=[modal.Mount.from_local_dir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), remote_path="/workspace")],
)
def run_script_remotely(script_path: str, script_args: list):
    import sys
    import subprocess
    import os

    # Change to workspace directory so relative paths in configs work
    os.chdir("/workspace")
    
    # Prepend 'python' to the command
    cmd = [sys.executable, script_path] + script_args
    
    print(f"🚀 Running on Modal: {' '.join(cmd)}", flush=True)
    
    # Run the script and stream output
    result = subprocess.run(
        cmd,
        env=os.environ.copy(),
        check=False,
    )
    
    if result.returncode != 0:
        print(f"❌ Script failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    else:
        print("✅ Script completed successfully.", flush=True)


@app.local_entrypoint()
def main():
    args = parse_args()
    
    if args.profile:
        os.environ["MODAL_PROFILE"] = args.profile
        print(f"Using Modal profile: {args.profile}")
        
    if not os.path.exists(args.script):
        print(f"Error: Script {args.script} not found locally.")
        sys.exit(1)

    # Dynamic GPU setting
    gpu_map = {
        "A10G": modal.gpu.A10G(),
        "A100": modal.gpu.A100(),
        "A100-80GB": modal.gpu.A100(size="80GB"),
        "H100": modal.gpu.H100(),
    }
    gpu_request = gpu_map[args.gpu]

    # Convert the absolute path back to a relative path against the mounted workspace root
    # e.g. "src/runners/sft_run.py" instead of local absolute path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_abs = os.path.abspath(args.script)
    if not script_abs.startswith(repo_root):
        print("Error: script must be inside the repository root.")
        sys.exit(1)
        
    script_rel = os.path.relpath(script_abs, repo_root)

    print(f"Deploying to Modal ({args.gpu}) with timeout {args.timeout}s...")
    
    # Run synchronously
    # To override the @app.function decorators dynamically, we use `.with_options()`
    run_script_remotely.with_options(gpu=gpu_request, timeout=args.timeout).remote(script_rel, args.script_args)


if __name__ == "__main__":
    main()
