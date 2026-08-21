#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${HOME}/miniconda3"
CONDA="${CONDA_ROOT}/bin/conda"

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "PROJECT_ROOT is not set." >&2
    exit 1
fi
echo "PROJECT_ROOT=${PROJECT_ROOT}"


if [[ ! -d "${CONDA_ROOT}" ]]; then
  MINICONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
  MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  echo "Installing miniconda at '$CONDA_ROOT'"
  wget -nc "${MINICONDA_URL}"
  bash "${MINICONDA_INSTALLER}" -b -p "${CONDA_ROOT}"
  "${CONDA}" config --set solver libmamba
  rm -f "${MINICONDA_INSTALLER}"
else
  echo "Miniconda is already installed at '$CONDA_ROOT'"
fi

ENV_FILE="${PROJECT_ROOT}/environment.yml"
ENV_NAME="$(awk -F ': ' '/name:/ {print $2}' "${ENV_FILE}")"
: "${ENV_NAME:?Conda environment name is missing from ${ENV_FILE}}"

if "${CONDA}" env list | grep -q "${ENV_NAME}"; then
  echo "An environment with this name already exists. Updating instead."
  "${CONDA}" env update -f "${ENV_FILE}" --prune
  TYPE="updated"
else
  echo "Creating environment '$ENV_NAME' based on '$ENV_FILE'."
  "${CONDA}" env create -f "${ENV_FILE}"
  TYPE="created"
fi

echo "Environment '${ENV_NAME}' ${TYPE} successfully"

echo "To add conda to you path modify ~/.bashrc and add export PATH=$HOME/miniconda3/bin:$PATH"
