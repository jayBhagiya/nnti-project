# Multilingual XGLM Evaluation and Quechua Adaptation

Course project for *Neural Networks: Theory and Implementation* (Saarland University, WS 2023/24).

The project studies how the multilingual language model [XGLM-564M](https://huggingface.co/facebook/xglm-564M) handles languages it saw little of, and how to adapt it to one of them, Ayacucho Quechua:

1. **Task 1, evaluation:** language-modelling loss of XGLM-564M and GPT-2 on parallel [FLORES](https://huggingface.co/datasets/facebook/flores) sentences in nine languages.
2. **Task 2, representations:** layer-wise token and sentence representations of XGLM, saved to HDF5 and visualised with PCA and t-SNE.
3. **Task 3, adaptation:** adapting XGLM to Quechua with full fine-tuning and three parameter-efficient methods implemented from scratch: [BitFit](https://arxiv.org/abs/2106.10199), [LoRA](https://arxiv.org/abs/2106.09685), and [IA³](https://arxiv.org/abs/2205.05638). Each run tracks how the other languages change.

The findings are in the [project write-up](https://jaybhagiya.me/projects/xglm-quechua-adaptation/). The original assignment text is in [`tasks/`](tasks/).

## Repository layout

| Path | Contents |
|---|---|
| `scripts/lm_eval.py` | Task 1: per-language loss, perplexity, and bits per byte |
| `scripts/task2.py` | Task 2: layer-wise hidden states to HDF5 |
| `scripts/task3.py` | Task 3: one adaptation run (training, evaluation, checkpoint) |
| `scripts/task3_custom_peft.py` | BitFit, LoRA, and IA³ implementations |
| `scripts/task3_data_preparation.py` | Quechua training data and FLORES evaluation subsets |
| `notebooks/` | Task 1 walkthrough and Task 2 PCA / t-SNE visualisations |
| `tests/` | Unit tests for the evaluation, PEFT, and data code |
| `submit_files/` | HTCondor jobs for running everything on a GPU cluster |

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.11 and the exact locked dependencies (PyTorch 2.2.2, Transformers 4.38) into `.venv/`.

```bash
git clone https://github.com/jayBhagiya/nnti-project.git
cd nnti-project

# CPU only (any machine)
uv sync --locked --extra cpu

# or NVIDIA GPU (CUDA 11.8 build of PyTorch)
uv sync --locked --extra cu118
```

The commands below use `--extra cpu`; replace it with `--extra cu118` if you installed the GPU build.

FLORES is a gated dataset. Accept its terms on the [dataset page](https://huggingface.co/datasets/facebook/flores) (approval is automatic), create an access token in your Hugging Face settings, and log in once:

```bash
uv run --locked --extra cpu hf auth login
```

The model and datasets download on first use (about 1.2 GB for XGLM-564M, plus 0.5 GB for GPT-2 in Task 1).

## Running the tasks

Each script takes `--help` for all options. Every command has a quick version that finishes on a laptop CPU in a few minutes, and a full version.

### Task 1: evaluate language models

```bash
# Quick: XGLM on two languages, 4 sentences each
uv run --locked --extra cpu python scripts/lm_eval.py \
  --models facebook/xglm-564M --languages eng_Latn quy_Latn \
  --max-samples 4 --output runs/task1-quick.csv

# Full: XGLM and GPT-2, nine languages, 200 sentences each
uv run --locked --extra cpu python scripts/lm_eval.py \
  --max-samples 200 --output runs/task1.csv
```

Each row of the CSV holds one model and language: negative log-likelihood, perplexity, bits per byte, and token counts. The script picks the GPU automatically when one is available (`--device cpu` forces the CPU).

### Task 2: extract and visualise representations

```bash
# Quick: 4 sentences per language
uv run --locked --extra cpu python scripts/task2.py --num-samples 4 --output runs/task2/base.h5

# Full: 200 sentences per language (about 5 GB of HDF5)
uv run --locked --extra cpu python scripts/task2.py --output runs/task2/base.h5
```

The HDF5 file stores, for each of the eight languages, the hidden states of all 25 layers (embedding layer plus 24 transformer layers) for every non-padding token, mean-pooled sentence vectors, the tokens, and the source sentences.

To plot them, open `notebooks/task2.ipynb`. It reads `runs/task2/base.h5`:

```bash
uv run --locked --extra cpu --with jupyter jupyter lab notebooks/task2.ipynb
```

### Task 3: adapt XGLM to Quechua

One run trains one method, evaluates on held-out Quechua and on FLORES in eight languages before and after training, and keeps the best checkpoint by validation loss.

```bash
# Quick smoke test on CPU: LoRA rank 1, 8 training sentences, 1 epoch
uv run --locked --extra cpu python scripts/task3.py \
  --method lora --rank 1 --train-samples 8 --adaptation-eval-samples 4 \
  --flores-eval-samples 4 --test-samples 4 --epochs 1 --batch-size 2 \
  --gradient-accumulation 1 --output-dir runs/smoke/lora-r1

# Full run on a GPU (defaults: 2,000 sentences, 10 epochs, fp16)
uv run --locked --extra cu118 python scripts/task3.py \
  --method lora --rank 4 --output-dir runs/task3/lora-r4
```

To run all seven configurations compared in the project:

```bash
for config in "full 0" "bitfit 0" "ia3 0" "lora 1" "lora 2" "lora 4" "lora 8"; do
  set -- $config
  uv run --locked --extra cu118 python scripts/task3.py \
    --method "$1" --rank "$2" --output-dir "runs/task3/$1-r$2"
done
```

**Hardware:** BitFit, IA³, and LoRA fit on a 12 GB GPU; full fine-tuning needs about 24 GB. Full runs on a CPU work but take many hours, so on a CPU keep to the quick settings.

Each run writes to its output directory:

| File | Contents |
|---|---|
| `config.json` | Arguments, parameter counts, library versions, data hashes |
| `history.jsonl` | Training loss per epoch and every evaluation |
| `summary.json` | Best epoch, final metrics, runtime, peak GPU memory |
| `best.pt` | Best checkpoint (only the trained parameters for BitFit, IA³, and LoRA) |

Runs log to [Weights & Biases](https://wandb.ai/) in offline mode by default, so no account is needed and the local files above hold every result. To sync runs to your W&B account, run `wandb login` and pass `--wandb-mode online` (plus `--wandb-entity <your-team>` if needed).

To see how adaptation changed the representations, rerun Task 2 on a checkpoint. The method and rank are read from the checkpoint:

```bash
uv run --locked --extra cpu python scripts/task2.py \
  --checkpoint runs/task3/lora-r4/best.pt --output runs/task2/lora-r4.h5
```

## Running on an HTCondor cluster

`submit_files/` runs the same scripts as cluster jobs inside the `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime` Docker image. A setup job installs uv and the locked GPU environment once into shared storage; every task job then reuses it.

**1. Edit the variables at the top of each `.sub` file:**

| Variable | Set it to |
|---|---|
| `project_dir` | Where this repository is cloned, on a path the worker nodes can read |
| `data_dir` | Large shared storage for the environment, model cache, logs, and results (plan for about 20 GB) |
| `wandb_entity` | Your W&B user or team (`task3.sub`, `task3_smoke.sub`) |
| `wandb_project`, `campaign` | Optional: W&B project and run-group names |

**2. Adapt the resource lines to your cluster:**
- `requirements` selects GPUs by memory (`GPUs_GlobalMemoryMb`). Add any extra constraints your cluster needs, such as a `UidDomain` or machine pool.
- `+WantGPUHomeMounted = true` is a site-specific attribute that mounts the home directory in the container. Remove it if your cluster doesn't define it.
- Your cluster must support the Docker universe. If it doesn't, switch to `universe = vanilla` and make sure the workers have Python available for `uv_setup.sh`.

**3. Create the storage folders, log in, and submit:**

```bash
mkdir -p /path/to/large-storage/nnti-project/{logs,runs,wandb,cache,venvs,python,tools}
# The jobs use data_dir/cache/huggingface as HF_HOME, so store the token there
HF_HOME=/path/to/large-storage/nnti-project/cache/huggingface uv run --locked --extra cpu hf auth login
uv run --locked --extra cpu wandb login       # saved in ~/.netrc

cd submit_files
condor_submit uv_setup.sub        # once: install the environment
condor_submit task3_smoke.sub     # quick end-to-end check on a tiny sample
condor_submit task1.sub
condor_submit task2.sub
condor_submit task3.sub           # queues the seven adaptation runs as separate jobs
```

The W&B login is read from your home directory, so log in from a machine that shares it (and `data_dir`) with the workers. Results land in `data_dir/runs/` and job logs in `data_dir/logs/`.

## Tests

```bash
uv run --locked --extra cpu python -m unittest discover -s tests
```
