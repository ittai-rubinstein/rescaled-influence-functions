# Reproducibility Materials for "Rescaled Influence Functions: Accurate Data Attribution in High Dimensions"

**Ittai Rubinstein, Samuel B. Hopkins**  
EECS and CSAIL, MIT, Cambridge, MA  
[ittair@mit.edu](mailto:ittair@mit.edu), [samhop@mit.edu](mailto:samhop@mit.edu)

This repository contains the reproducibility code and data for the paper _"Rescaled Influence Functions: Accurate Data Attribution in High Dimension"_ (under review). It includes all scripts and code required to reproduce the experiments, figures, and tables in the paper.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
- [Reproducibility Notes](#reproducibility-notes)

---

## 🔍 Overview

This project explores how to more accurately quantify the effect of individual training points on model predictions using **Rescaled Influence Functions (RIF)**, which improve upon classical influence functions (IF) especially in high-dimensional regimes. The repository includes:
- Implementations of IF and RIF.
- Scripts to embed datasets and run experiments.
- Code to produce the plots shown in the paper.

---

## ⚙️ Installation

We recommend using **conda** (or `venv` if preferred) to manage dependencies.

```bash
conda env create -f environment.yml
conda activate rif_env
```



You’ll also need:

- `wget`, `curl`, `unzip`
- [`kaggle` CLI](https://www.kaggle.com/docs/api) (for downloading some datasets)

### Setting up Kaggle CLI

Some datasets are downloaded using Kaggle's API. To use it:

1. Visit: [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Click “Create API Token”
3. Move the downloaded `kaggle.json` to the correct location:

```bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
---

## 📦 Dataset Setup

To download and prepare all datasets used in the paper, run the following command:

```bash
bash scripts/download_all_datasets.sh imdb spam dogfish esc50
bash scripts/generate_embeddings.sh
```

## Running the Experiments

To run all the experiments in our paper, simply call the following bash scripts

```bash
bash scripts/influence_experiment.sh # This will generate figure 1
bash scripts/vary_n_and_lambda_experiment.sh # This will generate figure 2
bash scripts/data_poisoning_experiment.sh # This will generate figure 3
```

We ran this experiments on a server equipped with 64GB RAM, 2 IBM POWER9 CPU cores, and 4 NVIDIA Tesla V100 SXM2 GPUs (each with 32GB memory).
The total runtime was only a few hours and required around 2GB of storage.

---

## 🗂️ Project Structure

Each directory plays a specific role in the reproducibility workflow:

- **scripts/**: Entry-point shell scripts for downloading data and running experiments.
- **datasets/**: Code for converting raw datasets into a standardized logistic regression format.
- **datasets/katl19/**: Code and data related to benchmarks reproduced from Koh et al. (2019).
- **src/**: Core algorithm implementations, including influence functions and our proposed rescaled variant.
- **experiments/**: Utility functions used by the main experiment scripts (e.g., metrics, evaluation).
- **main_files/**: Top-level Python scripts invoked by the bash scripts to run training and attribution.
- **paper_plots/**: Scripts to generate figures and tables for the paper from the stored results.

---
---

## 🔁 Reproducibility Notes

- Whenever possible, random seeds are fixed in the code to support reproducibility, but complete determinism is not guaranteed.
- Experiments are designed to be modular and relatively efficient:
  - Total runtime for reproducing all results is estimated at **10–100 hours** on a machine with **4 GPUs**.
  - Most individual experiments can be run in isolation and are less resource-intensive.
