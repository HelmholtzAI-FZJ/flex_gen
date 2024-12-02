#!/bin/bash

# See https://stackoverflow.com/a/28336473
SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

[[ "$0" != "${SOURCE_PATH}" ]] && echo "The activation script must be sourced, otherwise the virtual environment will not work." || ( echo "Vars script must be sourced." && exit 1) ;

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh


#=======
ENV_PROJ_DIR="$(dirname "$(realpath $ENV_DIR)")"
export TMPDIR="${ENV_PROJ_DIR}/.tmp"
export TMP=${TMPDIR}
export TEMP=${TMPDIR}
export PIP_CACHE_DIR="${ENV_PROJ_DIR}/.cache/pip"

PROJECT_DIR="$(dirname "$(realpath $ABSOLUTE_PATH)")"
export CACHE_DIR="${PROJECT_DIR}/.cache"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch/hub"
export TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/.cache/torch/torch_extensions"
export MPLCONFIGDIR="${PROJECT_DIR}/.cache/matplotlib"
export WANDB_CACHE_DIR="${PROJECT_DIR}/.cache/wandb"
# ======
export PYTHONPATH="$(echo "${ENV_DIR}"/lib/python*/site-packages):${PYTHONPATH}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

export http_proxy=http://134.94.199.178:7008; 
export https_proxy=$http_proxy 
export HTTP_PROXY=$http_proxy 
export HTTPS_PROXY=$http_proxy
export PROXY_CONNECTED=$(nc -zv 134.94.199.178 7008 >/dev/null 2>&1 && echo "PROXY Accessible" || echo "PROXY not accessible")

export PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2 | sed 's/^/python/')

# SLURM_JOB_PARTITION=dc-wai
if [[ "$ENV_DIR" == *"jurecadc_wai"* ]]; then
    export cuda_arch="sm_90"
    export compute_capability="9.0"
fi


# PYTHON_BIN_CHECK=$(which python)
# CURRENT_PYTHON_BIN_PATH="$ENV_DIR/bin/python3"
# rm $CURRENT_PYTHON_BIN_PATH
# ln -s $PYTHON_BIN_CHECK $CURRENT_PYTHON_BIN_PATH
source "${ENV_DIR}/bin/activate"


