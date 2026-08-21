#!/usr/bin/env bash
set -euo pipefail

nvidia-smi
echo "${CUDA_VISIBLE_DEVICES:-}"
echo "${HOSTNAME:-}"
which python

exec python ../scripts/task3.py "$@"
