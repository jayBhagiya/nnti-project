# Repository Guidelines

## Project Structure & Module Organization

This coursework repository studies multilingual XGLM representations and Quechua adaptation. `notebooks/` contains interactive work for Tasks 1–2. `scripts/` contains the Task 2 embedding pipeline, the Task 3 training entry point, and Task 3 data, PEFT, and parameter-count helpers. Keep new task-specific helpers beside these scripts. `tasks/` holds the assignment specifications; `submit_files/` holds the Conda environment and HTCondor launch files. `presentation/` contains plots and diagrams, while submitted report artifacts live at the repository root.

## Environment and Development Commands

Run commands from the repository root unless noted:

```bash
conda env create -f submit_files/environment.yml
conda env update -f submit_files/environment.yml --prune
conda run -n nnti-project python scripts/task2.py
conda run -n nnti-project python scripts/task3.py
python -m compileall scripts
```

The first two commands create or refresh the CUDA-enabled environment. The next two run embedding extraction and model adaptation; both download Hugging Face resources, and Task 3 expects a practical GPU plus W&B authentication. `compileall` is the lightweight syntax check. For HTCondor, work inside `submit_files/` and use `condor_submit setup.sub`, then `condor_submit task2.sub`. Verify Task 3's launcher before submission: `task3.sub` currently references `conda_run_task2.sh`.

## Coding Style & Naming Conventions

Use PEP 8 Python with four spaces; do not copy the tabs present in older helper files. Name functions and variables `snake_case`, classes `PascalCase`, and module constants `UPPER_SNAKE_CASE`. Group standard-library, third-party, and local imports. Preserve comments that explain tensor shapes or training assumptions. No formatter or linter is configured, so keep formatting changes focused and avoid unrelated notebook-output churn.

## Testing Guidelines

There is no automated test suite or coverage target. Before a merge request, run `python -m compileall scripts` and exercise the affected entry point. Restart and run affected notebooks when visualization logic changes. Record GPU/CUDA details, sample sizes, and W&B run links for training changes; state clearly when a full run was impractical.

## Commit & Merge Request Guidelines

History favors short descriptive subjects, without a mandatory Conventional Commits format. Prefer an imperative, task-scoped subject such as `task3: fix IA3 projection`. Merge requests should summarize affected tasks and files, list validation performed, link relevant issues, and include plots or screenshots for visual changes. Call out environment, cluster, dataset, or W&B changes explicitly.

## Data and Secrets

Authenticate to Hugging Face and W&B outside the repository; never commit tokens. Keep generated `*.h5`, `*_model.pt`, `logs/`, `wandb/`, and cache files out of commits, and inspect `git status` before pushing.
