#!/bin/bash

# Source the environment file to get DATA_DIRECTORY and INSTALL_DIRECTORY
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh" || {
    echo "Failed to source environment variables." >&2
    exit 1
}


# Run poisoning experiment
python "${INSTALL_DIRECTORY}/main_files/data_poisoning_experiment.py" \
    --dataset cifar10_auto_truck_resnet50  \
    --num_test 100 \
    --verbosity 2

# Generate plot
python "${INSTALL_DIRECTORY}/paper_plots/plot_data_poisoning.py" \
    --dataset cifar10_auto_truck_resnet50
