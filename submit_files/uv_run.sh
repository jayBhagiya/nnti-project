#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"

PYTHON="${UV_PROJECT_ENVIRONMENT:-${DATA_DIR}/venvs/nnti-project}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Locked environment not found at ${PYTHON}; submit uv_setup.sub first." >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: uv_run.sh SCRIPT [ARG ...]" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-${DATA_DIR}/cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${WANDB_DIR:-${DATA_DIR}/wandb}"
cd "${PROJECT_DIR}"
exec "${PYTHON}" -u "$@"
