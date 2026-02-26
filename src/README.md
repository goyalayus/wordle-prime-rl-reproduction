# Abstract Runners Architecture

This directory structure completely abstracts the training logic away from the task logic, allowing rapid experimentation across tasks (e.g. Wordle, Chess, Math) without duplicating massive boilerplate scripts.

## Directory Layout

```
src/
├── runners/
│   ├── sft_run.py        # Generic Unsloth SFT runner
│   ├── rl_run.py         # Generic Unsloth/vLLM GRPO runner
│   └── evals.py          # Generic vLLM evaluation generator
├── tasks/
│   └── wordle_task.py    # Example plugin holding Wordle datasets & reward functions
├── configs/
│   └── rl_config.yaml    # Example YAML configuration
├── run_modal.py          # Deploy any script to Modal
└── run_aws.py            # Deploy any script to AWS EC2
```

## How It Works

Instead of hardcoding Wordle inside `train_grpo.py`, we now have `src/runners/rl_run.py`.

The runner takes two arguments:
1. `--config`: A YAML file configuring the model, Unsloth, vLLM, and LoRA.
2. `--task-module`: A Python file that acts as a plugin.

The runner will import the task module and expect it to have specific hooks:

### RL Hooks
- `get_rl_dataset(config)` -> returns a HuggingFace `Dataset` containing prompts.
- `get_reward_funcs(config, tokenizer)` -> returns a list of Python functions matching the GRPO reward signature.

### SFT Hooks
- `get_sft_dataset(config, tokenizer)` -> returns a HuggingFace `Dataset` ready for SFT.
- `get_sft_eval_dataset(config, tokenizer)` -> (optional) returns validation dataset.
- `get_sft_formatting_func(config, tokenizer)` -> (optional) returns a chat template formatting function.
- `get_sft_data_collator(config, tokenizer)` -> (optional) returns a custom data collator (e.g. for completion-only masking).

### Evals Hooks
- `get_eval_dataset(config)` -> returns dataset of prompts to evaluate.
- `compute_metrics(prompts, completions, dataset_rows, config, tokenizer)` -> returns a dictionary of metrics, which the runner writes to disk.

## Running Locally

```bash
# Run the RL pipeline using the Wordle task plugin
python src/runners/rl_run.py \
    --config src/configs/rl_config.yaml \
    --task-module src/tasks/wordle_task.py
```

## Running on Cloud

### Modal
```bash
# Run the SFT pipeline on an A100 using Modal
python src/run_modal.py --gpu A100 --script src/runners/sft_run.py \
    --config src/configs/sft_config.yaml \
    --task-module src/tasks/wordle_task.py
```

### AWS EC2
```bash
# Run the RL pipeline on a g5.2xlarge in AWS, auto-terminating when done
python src/run_aws.py \
    --key-name my-aws-key \
    --key-path ~/.ssh/my-aws-key.pem \
    --security-group sg-0123456789 \
    --terminate-after \
    --script src/runners/rl_run.py \
    --config src/configs/rl_config.yaml \
    --task-module src/tasks/wordle_task.py
```
