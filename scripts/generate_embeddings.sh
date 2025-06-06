#!/bin/bash

# Source the environment file to get DATA_DIRECTORY and INSTALL_DIRECTORY
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh" || {
    echo "Failed to source environment variables." >&2
    exit 1
}

# Needed for the Enron dataset
python -m spacy download en_core_web_sm


# Run embedding generation for key datasets and models
python "${INSTALL_DIRECTORY}/datasets/frozen_embeddings/generate_all_embeddings.py" \
    --datasets imdb cifar10_catdog cifar10_auto_truck \
    --vision_models resnet18 resnet50

# Try ESC-50 embeddings; failure is allowed
python "${INSTALL_DIRECTORY}/datasets/frozen_embeddings/generate_all_embeddings.py" \
    --datasets esc50 \
    --vision_models resnet18 resnet50 || echo "ESC-50 embedding generation failed, continuing..."
