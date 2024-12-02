#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

# python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

if [ -d "$ENV_DIR" ]; then
    # Warn the user if the directory does exist    
    echo "Warning: The directory $ENV_DIR alreadys exist. The virtual environment will not be created."
    echo "Please remove the directory $ENV_DIR if you want to create a new virtual environment."
else
    # Create the virtual environment if the directory does not exist
    python3 -m venv --prompt "$ENV_NAME" --system-site-packages "$ENV_DIR"
    echo "Virtual environment created successfully."
fi

source "${ABSOLUTE_PATH}"/activate.sh

mkdir -p $CACHE_DIR
mkdir -p $TMPDIR
mkdir -p $PIP_CACHE_DIR

# sh "${ABSOLUTE_PATH}"/install_requirements.sh

rm -rf "${ABSOLUTE_PATH}/.gitignore"
echo -e ".cache/\n.tmp/\nvenv*/" >> "${ABSOLUTE_PATH}/.gitignore"

