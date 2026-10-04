#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"

UV_VERSION=0.11.30
UV_TOOL_DIR="${UV_TOOL_DIR:-${DATA_DIR}/tools/uv-${UV_VERSION}}"
UV_BIN="${UV_BIN:-${UV_TOOL_DIR}/bin/uv}"
if [[ ! -x "${UV_BIN}" ]]; then
    if ! command -v python >/dev/null; then
        echo "The worker image does not provide Python for bootstrapping uv." >&2
        exit 1
    fi
    python -m pip install \
        --disable-pip-version-check \
        --no-deps \
        --prefix "${UV_TOOL_DIR}" \
        "uv==${UV_VERSION}"
fi
UV_ACTUAL="$("${UV_BIN}" --version)"
if [[ "${UV_ACTUAL}" != "uv ${UV_VERSION}"* ]]; then
    echo "Expected uv ${UV_VERSION}, found ${UV_ACTUAL}." >&2
    exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${DATA_DIR}/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${DATA_DIR}/python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${DATA_DIR}/venvs/nnti-project}"

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
cd "${PROJECT_DIR}"

"${UV_BIN}" python install 3.11.15
"${UV_BIN}" sync --locked --extra cu118 --no-dev
"${UV_PROJECT_ENVIRONMENT}/bin/python" -c \
    'import torch; assert torch.version.cuda == "11.8", torch.version.cuda; print(torch.__version__, torch.version.cuda)'
