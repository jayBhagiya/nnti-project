# Multilingual XGLM Evaluation and Quechua Adaptation

Coursework experiments for multilingual language-model evaluation, representation extraction, and Quechua adaptation with full fine-tuning, BitFit, IA³, and LoRA.

## Dependencies

Python 3.11 and [uv](https://docs.astral.sh/uv/) manage the locked environment. Core dependencies include PyTorch 2.2.2, Transformers, Datasets, SentencePiece, HDF5, and Weights & Biases. Task 3 requires an NVIDIA GPU; the GPU environment targets CUDA 11.8.

## Environment Setup

```bash
# Local CPU environment
uv sync --locked --extra cpu

# CUDA 11.8 environment
uv sync --locked --extra cu118
```

Accept access to `facebook/flores`, then authenticate before downloading data or starting online training:

```bash
uv run --frozen --no-sync hf auth login
uv run --frozen --no-sync wandb login --verify
```

## Run the Tasks

```bash
# Task 1: multilingual language-model evaluation
uv run --locked --extra cpu python scripts/lm_eval.py \
  --device cpu --max-samples 200 --output runs/task1.csv

# Task 2: layer-wise token and sentence representations
uv run --locked --extra cpu python scripts/task2.py \
  --device cpu --output runs/task2/base.h5

# Task 3: one GPU adaptation run
uv run --locked --extra cu118 python scripts/task3.py \
  --method lora --rank 4 --output-dir runs/task3/lora-r4
```

For HTCondor, update `project_dir`, `data_dir`, `initialdir`, and `wandb_entity` in `submit_files/*.sub`. Create the directories listed in `uv_setup.sub`, then run:

```bash
cd submit_files
condor_submit uv_setup.sub
condor_submit task3_smoke.sub
condor_submit task1.sub
condor_submit task2.sub
condor_submit -batch-name nnti-final-v1 task3.sub
```

`task3.sub` queues the seven adaptation configurations as independent jobs.

## Checks

```bash
uv run --locked --extra cpu python -m unittest discover -s tests
uv run --locked --extra cpu python scripts/task2.py --self-test
```
