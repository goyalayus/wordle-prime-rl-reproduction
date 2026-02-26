import argparse
import boto3
import os
import subprocess
import sys
import time
from botocore.exceptions import ClientError


def parse_args():
    parser = argparse.ArgumentParser(description="Run Unsloth scripts on AWS EC2")
    parser.add_argument("--script", required=True, help="Path to runner script (e.g., src/runners/sft_run.py)")
    parser.add_argument("--instance-type", default="g5.2xlarge", help="AWS Instance type (default: g5.2xlarge)")
    parser.add_argument("--ami", default=None, help="AMI ID (default: latest Ubuntu DLAMI)")
    parser.add_argument("--key-name", required=True, help="AWS Key Pair name for SSH access")
    parser.add_argument("--key-path", required=True, help="Local path to the .pem file for SSH")
    parser.add_argument("--security-group", required=True, help="Security Group ID (must allow SSH)")
    parser.add_argument("--subnet", default=None, help="Subnet ID (optional)")
    parser.add_argument("--region", default="us-east-1", help="AWS Region")
    parser.add_argument("--terminate-after", action="store_true", help="Terminate instance after run finishes")
    parser.add_argument("--profile", default=None, help="AWS profile to use")
    
    # Rest of the args go to the script itself
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script")
    return parser.parse_args()


def get_latest_dlami(ec2_client):
    """Fetch the latest Ubuntu 22.04 Deep Learning AMI."""
    response = ec2_client.describe_images(
        Filters=[
            {"Name": "name", "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]},
            {"Name": "state", "Values": ["available"]}
        ],
        Owners=["amazon"]
    )
    images = response["Images"]
    if not images:
        raise ValueError("Could not find a valid Ubuntu Deep Learning AMI.")
    
    # Sort by creation date (newest first)
    images.sort(key=lambda x: x["CreationDate"], reverse=True)
    return images[0]["ImageId"]


def run_ssh_command(host: str, key_path: str, user: str, command: str, stream: bool = True):
    ssh_cmd = [
        "ssh", "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"{user}@{host}", command
    ]
    if stream:
        result = subprocess.run(ssh_cmd)
        return result.returncode == 0
    else:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout


def rsync_to_host(host: str, key_path: str, user: str, local_dir: str, remote_dir: str):
    rsync_cmd = [
        "rsync", "-avz", "--exclude", ".git", "--exclude", "outputs", "--exclude", "__pycache__",
        "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
        f"{local_dir}/", f"{user}@{host}:{remote_dir}/"
    ]
    result = subprocess.run(rsync_cmd)
    return result.returncode == 0


def main():
    args = parse_args()
    
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    ec2 = session.client('ec2')
    ec2_resource = session.resource('ec2')

    ami_id = args.ami if args.ami else get_latest_dlami(ec2)
    print(f"Using AMI: {ami_id} in {args.region}")

    # 1. Launch Instance
    print(f"Launching EC2 instance: {args.instance_type}...")
    run_args = {
        "ImageId": ami_id,
        "InstanceType": args.instance_type,
        "KeyName": args.key_name,
        "SecurityGroupIds": [args.security_group],
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": 250, "VolumeType": "gp3"}
            }
        ]
    }
    if args.subnet:
        run_args["SubnetId"] = args.subnet

    instances = ec2_resource.create_instances(**run_args)
    instance = instances[0]
    
    print(f"Instance {instance.id} launched. Waiting for it to run...")
    instance.wait_until_running()
    instance.reload()
    public_ip = instance.public_ip_address
    
    if not public_ip:
        print("Instance has no public IP. Assuming you have a VPN/VPC connection and using Private IP.")
        public_ip = instance.private_ip_address

    print(f"Instance is running. IP: {public_ip}")
    print("Waiting for SSH to become available...")
    
    user = "ubuntu"
    max_retries = 30
    for _ in range(max_retries):
        success, _ = run_ssh_command(public_ip, args.key_path, user, "echo 'SSH is up'", stream=False)
        if success:
            break
        time.sleep(10)
    else:
        print("SSH did not become available in time. Exiting.")
        if args.terminate_after:
            instance.terminate()
        sys.exit(1)

    print("SSH is up! Preparing environment...")
    
    # 2. Setup Environment
    # The DLAMI uses conda by default. We install unsloth/vllm/trl inside the active pytorch env.
    setup_cmds = [
        "source activate pytorch",
        "pip install unsloth unsloth-zoo",
        "pip install vllm==0.7.3",
        "pip install git+https://github.com/huggingface/trl.git",
        "pip install wandb datasets pyyaml",
        "mkdir -p ~/workspace"
    ]
    setup_cmd = " && ".join(setup_cmds)
    run_ssh_command(public_ip, args.key_path, user, f"bash -c '{setup_cmd}'")

    # 3. Rsync Codebase
    print("Syncing codebase to instance...")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rsync_to_host(public_ip, args.key_path, user, repo_root, "~/workspace")

    # Sync HF / Wandb tokens from local env if available
    env_vars = []
    for var in ["HF_TOKEN", "WANDB_API_KEY"]:
        val = os.environ.get(var)
        if val:
            env_vars.append(f"export {var}={val}")

    # 4. Run the Script
    print(f"Executing: {args.script} {' '.join(args.script_args)}")
    
    # We must relative-path the script from the workspace dir
    script_abs = os.path.abspath(args.script)
    script_rel = os.path.relpath(script_abs, repo_root)
    
    run_script_cmd = (
        f"source activate pytorch && "
        f"{' && '.join(env_vars) if env_vars else ':'} && "
        f"cd ~/workspace && "
        f"python {script_rel} {' '.join(args.script_args)}"
    )
    
    success = run_ssh_command(public_ip, args.key_path, user, f"bash -c '{run_script_cmd}'")
    
    if not success:
        print("Script failed.")
    else:
        print("Script completed successfully.")
        
    # 5. Optional Teardown
    if args.terminate_after:
        print(f"Terminating instance {instance.id}...")
        instance.terminate()
        print("Terminated.")
    else:
        print(f"Instance left running. IP: {public_ip}")

if __name__ == "__main__":
    main()
