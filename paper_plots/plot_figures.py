from experiments.experimental_pipeline import DEFAULT_RESULTS
from run_all_experiments import DATASETS
from run_dogfish_experiment import DATASET

# We only want to plot some of the regularization values to fit in a two-column submission format.
REGULARIZATIONS = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1.0]
REG_NAMES = ["1E-5", "1E-4", "1E-3", "1E-2", "1E-1", "1"]

if __name__ == "__main__":
    pass
