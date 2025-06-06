#!/bin/bash

# Source the environment file to get DATA_DIRECTORY and INSTALL_DIRECTORY
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh" || {
    echo "Failed to source environment variables." >&2
    exit 1
}


# Run influence experiments
for DATASET in esc50 cifar10_catdog_resnet50 cifar10_auto_truck_resnet50 DogFish Enron; do
    python "${INSTALL_DIRECTORY}/main_files/influence_experiment.py" \
        --dataset "$DATASET" \
        --verbosity 2 || { echo "Failed on dataset $DATASET"; exit 1; }
done

# Parse optional output directory
OUT_DIR="${1:-.}"

# Plot nonlinear version
python "${INSTALL_DIRECTORY}/paper_plots/plot_influence.py" \
    --datasets esc50 cifar10_catdog_resnet50 cifar10_auto_truck_resnet50 DogFish Enron \
    --titles "ESC-50" "Cat vs Dog" "Truck vs Automobile" "Dog vs Fish" "Spam vs Ham" \
    --exclude_strategies "High Loss" \
    --output "$OUT_DIR/figure1_nonlinear.pdf" \
    --include_strategy_legend \
    --selected_metrics influence_on_test_predictions influence_on_test_fixed_loss influence_on_total_loss

# Plot linear version
python "${INSTALL_DIRECTORY}/paper_plots/plot_influence.py" \
    --datasets esc50 cifar10_catdog_resnet50 cifar10_auto_truck_resnet50 DogFish Enron \
    --titles "ESC-50" "Cat vs Dog" "Truck vs Automobile" "Dog vs Fish" "Spam vs Ham" \
    --exclude_strategies "High Loss" \
    --output "$OUT_DIR/figure1_linear.pdf" \
    --include_strategy_legend \
    --selected_metrics influence_on_test_predictions influence_on_test_fixed_loss influence_on_total_loss \
    --use_linear
