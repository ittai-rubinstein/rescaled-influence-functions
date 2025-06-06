#!/bin/bash

# Source the environment file to get DATA_DIRECTORY and INSTALL_DIRECTORY
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh" || {
    echo "Failed to source environment variables." >&2
    exit 1
}




############################################
# 1. DogFish experiments: vary regularization
############################################

echo "===================="
echo " Running DogFish experiments (varying regularization)"
echo "===================="

DOGFISH_DATASET="DogFish"
DOGFISH_REGULARIZATIONS=(0.00001 0.0001 0.001 0.01 0.1 1.0)
DOGFISH_REG_NAMES=("1E_5" "1E_4" "1E_3" "1E_2" "1E_1" "1")

for i in "${!DOGFISH_REGULARIZATIONS[@]}"; do
    REG="${DOGFISH_REGULARIZATIONS[$i]}"
    REG_NAME="${DOGFISH_REG_NAMES[$i]}"

    echo
    echo "================================================================================"
    echo "Starting DogFish experiment: regularization=$REG"
    echo "================================================================================"

    python "${INSTALL_DIRECTORY}/main_files/influence_experiment.py" \
        --dataset "$DOGFISH_DATASET" \
        --reg_type "L2" \
        --regularization "$REG" \
        --experiment_name "${DOGFISH_DATASET}_${REG_NAME}" \
        --force_refresh 0 \
        --verbosity 2
done

############################################
# 2. IMDB experiments: vary max_train_samples
############################################
echo
echo "===================="
echo " Running IMDB experiments (varying max_train_samples)"
echo "===================="

IMDB_DATASET='*IMDB*'  # Fuzzy matching
IMDB_NAME="IMDB"       # Clean name for experiment naming
D=769
MULTIPLIERS=(1 2 4 8 16 32)
IMDB_EXP_NAMES=("1d" "2d" "4d" "8d" "16d" "32d")
IMDB_MAX_TRAIN_SAMPLES=()

for m in "${MULTIPLIERS[@]}"; do
    IMDB_MAX_TRAIN_SAMPLES+=($(($D * $m)))
done

for i in "${!IMDB_EXP_NAMES[@]}"; do
    NAME="${IMDB_EXP_NAMES[$i]}"
    MTS="${IMDB_MAX_TRAIN_SAMPLES[$i]}"

    echo
    echo "================================================================================"
    echo "Starting IMDB experiment: max_train_samples=$MTS"
    echo "================================================================================"

    python "${INSTALL_DIRECTORY}/main_files/influence_experiment.py" \
        --dataset "$IMDB_DATASET" \
        --experiment_name "${IMDB_NAME}_${NAME}" \
        --max_train_samples "$MTS" \
        --force_refresh 0\
        --verbosity 2
done

# Run plot generation
python "${INSTALL_DIRECTORY}/paper_plots/plot_influence_reg_sample_scaling.py" \
    --regularization_dataset DogFish \
    --sample_dataset IMDB \
    --output sample_vs_reg.pdf \
    --metric influence_on_total_loss \
    --exclude_strategies "High Loss"
