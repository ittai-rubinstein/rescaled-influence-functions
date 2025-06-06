#!/bin/bash

# ========== Config ==========
APP_NAME="rif"
CONFIG_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/$APP_NAME"
CONFIG_FILE="$CONFIG_DIR/config.env"

# ========== Helper Functions ==========

prompt_path() {
    local prompt="$1"
    local default="$2"
    local result
    read -rp "$prompt [$default]: " result
    echo "${result:-$default}"
}

clean_path() {
    realpath -m "$1"
}

load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$CONFIG_FILE"
    fi
}

save_config() {
    mkdir -p "$CONFIG_DIR"
    {
        echo "INSTALL_DIRECTORY=\"$INSTALL_DIRECTORY\""
        echo "DATA_DIRECTORY=\"$DATA_DIRECTORY\""
    } > "$CONFIG_FILE"
}

move_data() {
    if [[ "$1" != "$2" && -d "$1" ]]; then
        echo "Moving data from $1 to $2 ..."
        mkdir -p "$2"
        shopt -s dotglob nullglob
        mv "$1"/* "$2/" 2>/dev/null
        shopt -u dotglob nullglob
        echo "Data moved."
    fi
}

clear_data_prompt() {
    echo -n "Do you want to clear the contents of the current DATA_DIRECTORY ($DATA_DIRECTORY)? [y/N]: "
    read -r confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo "Clearing $DATA_DIRECTORY ..."
        rm -rf "$DATA_DIRECTORY"/*
        echo "Data directory cleared."
    else
        echo "Skipping data clearing."
    fi
}

# ========== Main Logic ==========

load_config

if [[ -n "$INSTALL_DIRECTORY" && -n "$DATA_DIRECTORY" ]]; then
    echo "Existing configuration found:"
    echo "  INSTALL_DIRECTORY: $INSTALL_DIRECTORY"
    echo "  DATA_DIRECTORY:    $DATA_DIRECTORY"
    echo

    read -rp "Do you want to edit these settings? [y/N]: " edit
    if [[ ! "$edit" =~ ^[Yy]$ ]]; then
        echo "Exiting without changes."
        exit 0
    fi
fi

# Default paths
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INSTALL_DIR="$SCRIPT_DIR"
DEFAULT_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/$APP_NAME/data"

# Prompt user
new_install_dir=$(prompt_path "Enter INSTALL_DIRECTORY (path to source code)" "$DEFAULT_INSTALL_DIR")
new_data_dir=$(prompt_path "Enter DATA_DIRECTORY (for large data files)" "$DEFAULT_DATA_DIR")

# Clean and normalize paths
new_install_dir=$(clean_path "$new_install_dir")
new_data_dir=$(clean_path "$new_data_dir")

# If changing DATA_DIRECTORY, move data
if [[ -n "$DATA_DIRECTORY" && "$new_data_dir" != "$DATA_DIRECTORY" ]]; then
    move_data "$DATA_DIRECTORY" "$new_data_dir"
fi

# Update config vars
INSTALL_DIRECTORY="$new_install_dir"
DATA_DIRECTORY="$new_data_dir"

# Save config
save_config

# Offer to clear data
clear_data_prompt

echo
echo "Setup complete."
echo "INSTALL_DIRECTORY: $INSTALL_DIRECTORY"
echo "DATA_DIRECTORY:    $DATA_DIRECTORY"
