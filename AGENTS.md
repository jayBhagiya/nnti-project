# Repository Guidelines

## Project Structure & Module Organization

This coursework repository studies multilingual XGLM representations and Quechua adaptation. `notebooks/` contains the original Task 1–2 exploration. Reproducible entry points and PEFT helpers live in `scripts/`; focused regression checks live in `tests/`. `tasks/` holds the assignment specifications, while `submit_files/` contains the `uv` and HTCondor workflow. `presentation/` contains report figures and diagrams.

## Environment and Development Commands

Run commands from the repository root unless noted:

```bash
uv sync --locked --extra cpu
uv lock --check
uv run --locked --extra cpu python -m unittest discover -s tests
uv run --locked --extra cpu python scripts/task2.py --self-test
```

Use the `cpu` extra for local checks and `cu118` on the cluster. Task 1 evaluation is `scripts/lm_eval.py`; Task 2 extraction is `scripts/task2.py`; Task 3 trains exactly one configuration per invocation. On `<submit-node>`, create the locked GPU environment with `condor_submit submit_files/uv_setup.sub`, then run `task3_smoke.sub` before the parallel campaign in `task3.sub`. Task scripts download Hugging Face resources; Task 3 also requires private W&B authentication.

## Coding Style & Naming Conventions

Use PEP 8 Python with four spaces; do not copy the tabs present in older helper files. Name functions and variables `snake_case`, classes `PascalCase`, and module constants `UPPER_SNAKE_CASE`. Group standard-library, third-party, and local imports. Preserve comments that explain tensor shapes or training assumptions. No formatter or linter is configured, so keep formatting changes focused and avoid unrelated notebook-output churn.

## Testing Guidelines

Use the standard-library `unittest` suite; no coverage target is configured. Every adapter or metric change needs a small synthetic regression that avoids model downloads. Run the commands above and exercise the affected CLI. GPU changes additionally require the HTCondor smoke job. Record CUDA/GPU details, sample manifests, Git commit, and private W&B run links; say when a full run was impractical.

## Commit & Merge Request Guidelines

History favors short descriptive subjects, without a mandatory Conventional Commits format. Prefer an imperative, task-scoped subject such as `task3: fix IA3 projection`. Merge requests should summarize affected tasks and files, list validation performed, link relevant issues, and include plots or screenshots for visual changes. Call out environment, cluster, dataset, or W&B changes explicitly.

## Data and Secrets

Authenticate to Hugging Face and W&B outside the repository; never commit tokens. Keep generated HDF5 files, checkpoints, `runs/`, `logs/`, `wandb/`, and caches out of commits. Public web exports must omit raw FLORES text and private W&B URLs. Inspect `git status` before pushing.
