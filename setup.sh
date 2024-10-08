#!/bin/bash
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")

echo "The script is located at: $SCRIPT_PATH"

pip3 install poetry
poetry install

VENV_PATH=$(poetry env info -p)
source "$VENV_PATH/bin/activate"

# 定义相对路径
TORCH_SCATTER_RELATIVE_PATH="GMN/torch_scatter-2.0.9-cp37-cp37m-macosx_10_15_x86_64.whl"
TORCH_SPARSE_RELATIVE_PATH="GMN/torch_sparse-0.6.16-cp37-cp37m-macosx_10_15_x86_64.whl"

pip3 install "${SCRIPT_DIR}/${TORCH_SCATTER_RELATIVE_PATH}"
pip3 install "${SCRIPT_DIR}/${TORCH_SPARSE_RELATIVE_PATH}"