SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

## Check if this script is sourced
[[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1) ;
## Determine location of this file
RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
####################################

### User Configuration
YOUR_ENV_NAME="venv"


suffix=""

# Check if SLURM_JOB_PARTITION is set
if [ -n "$SLURM_JOB_PARTITION" ]; then
    # Check if SLURM_JOB_PARTITION is 'dc-wai'
    if [ "$SLURM_JOB_PARTITION" = "dc-wai" ]; then
        suffix="_wai"
    fi
    if [ "$SLURM_JOB_PARTITION" = "dc-gh" ]; then
        suffix="_gh"
    fi


    if [ "$SLURM_JOB_PARTITION" = "dc-hawi" ]; then
        # From 16.Oct.2024, the dc-wai partiion will be renamed to dc-hwai, to adapt the users who 
        # already has *_wai venv folder, we need to point the suffix to `wai`
        # Check if the original '_wai' environment exists
        if [ -d "${ABSOLUTE_PATH}/${YOUR_ENV_NAME}_${SYSTEMNAME}_wai" ]; then
            suffix="_wai"
        else
            suffix="_hwai"
        fi
    fi
fi

YOUR_ENV_NAME="${YOUR_ENV_NAME}_$SYSTEMNAME$suffix"

export ENV_NAME="$(basename "$ABSOLUTE_PATH")"             # Default Name of the venv is the directory that contains this file
export ENV_DIR="${ABSOLUTE_PATH}/${YOUR_ENV_NAME}"         # Default location of this VENV is "./venv"
# echo " - Your Virtual Enviroment will be placed at ${ABSOLUTE_PATH}/${YOUR_ENV_NAME}_$SYSTEMNAME"