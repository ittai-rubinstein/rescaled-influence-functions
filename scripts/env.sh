#!/bin/bash

# ========== Config ==========
APP_NAME="rif"
CONFIG_FILE="${XDG_DATA_HOME:-$HOME/.local/share}/$APP_NAME/config.env"

# ========== Load and Export ==========

if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
else
    echo "Config file not found: $CONFIG_FILE" >&2
    return 1 2>/dev/null || exit 1
fi

# Export DATA_DIRECTORY
export DATA_DIRECTORY

# Prepend INSTALL_DIRECTORY to PYTHONPATH
if [[ -n "$INSTALL_DIRECTORY" ]]; then
    export PYTHONPATH="$INSTALL_DIRECTORY${PYTHONPATH:+:$PYTHONPATH}"
fi
