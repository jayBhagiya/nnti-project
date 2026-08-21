# Cluster Jobs

For local CPU development, create the locked environment once and then reuse it
without synchronization:

```bash
uv sync --locked --extra cpu
uv run --frozen --no-sync python -m unittest discover -s tests
```

Commit and push the locally validated changes first. Then connect with
`ssh <submit-node>`; keep the Git checkout in your home directory and generated files in
`/data`. Clone once, or fetch an existing checkout, and detach at the exact
pushed commit:

```bash
mkdir -p /path/to/projects
cd /path/to/projects
git clone git@github.com:jayBhagiya/nnti-project.git
cd nnti-project-llms
git fetch origin
git checkout --detach COMMIT_SHA
mkdir -p /path/to/large-storage/nnti-project/{logs,runs,wandb,cache,venvs,python,tools}
cd submit_files
condor_submit uv_setup.sub
```

For later updates, omit `git clone`; run `git fetch origin` and check out the
new commit. Do not run campaign jobs from an uncommitted remote tree.

Wait for the setup job to finish successfully. Accept the FLORES access terms
in your Hugging Face account, then authenticate Hugging Face and W&B once
without putting either token in Git or a submit file:

```bash
HF_HOME=/path/to/large-storage/nnti-project/cache/huggingface \
  /path/to/large-storage/nnti-project/venvs/nnti-project/bin/huggingface-cli login
/path/to/large-storage/nnti-project/venvs/nnti-project/bin/wandb login --verify
condor_submit task3_smoke.sub
```

Confirm the online smoke run, checkpoint, summary, and logs before submitting
the independent jobs:

```bash
condor_submit task1.sub
condor_submit task2.sub
condor_submit -batch-name nnti-final-v1 task3.sub
```

`task3.sub` queues seven independent one-GPU jobs. HTCondor runs them in
parallel as matching GPUs become available. Experiment jobs execute the locked
environment directly and never run `uv sync`. Model and dataset revisions are
pinned in the Python entry points; each artifact also records resolved commits,
fingerprints, and content manifests.

For the later adapted-model Task 2 extraction, set `run_label` and append the
checkpoint argument in `task2.sub`, for example:

```text
run_label = adapted-full-s42
model_args = --checkpoint /path/to/large-storage/nnti-project/runs/final-v1/<run>/best.pt
```

Use `condor_q`, `condor_q -hold`, and the files under the `/data` log directory
for monitoring. The legacy Conda files remain only as a manual rollback
reference; the locked `uv` path is the supported campaign workflow.
