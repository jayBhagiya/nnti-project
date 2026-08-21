#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${HOME}/miniconda3"
CONDA="${CONDA_ROOT}/bin/conda"

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "PROJECT_ROOT is not set." >&2
    exit 1
fi
echo "PROJECT_ROOT=${PROJECT_ROOT}"

# Check if environment exists
if [[ ! -x "${CONDA}" ]]; then
    echo "Miniconda is not installed. Run condor_submit setup.sub first." >&2
    exit 1
fi

ENV_FILE="${PROJECT_ROOT}/environment.yml"
ENV_NAME="$(awk -F ': ' '/name:/ {print $2}' "${ENV_FILE}")"
: "${ENV_NAME:?Conda environment name is missing from ${ENV_FILE}}"

echo "Running Task 3 in Conda environment ${ENV_NAME}"

cd "${PROJECT_ROOT}"
exec "${CONDA}" run --no-capture-output -n "${ENV_NAME}" \
    bash "${PROJECT_ROOT}/run_task3.sh" "$@"
