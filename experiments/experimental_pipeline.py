import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
import os

from datasets.common import SplitDataset, DatasetName
from datasets.frozen_embeddings.loader import EmbeddedDataset
from experiments.experiment import run_experiment_on_removal_set, ExperimentResult
from src.influence_functions import InfluenceFunctions, compute_model_influences
from datasets.load_datasets import load_dataset
from experiments.train_base_model import train_logistic_regression
from src.logistic_regression import LogisticRegression, RegularizationType
from experiments.removal_strategies import (
    RandomRemovalSetGenerator,
    ClusterRemovalSetGenerator,
    TopPercentileRemovalSetGenerator, RemovalSetGenerator,
)

DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", None)
if DATA_DIRECTORY is None:
    EXP_DIRECTORY = Path(__file__).parent
else:
    EXP_DIRECTORY = Path(DATA_DIRECTORY) / "experiments"

DEFAULT_CACHE = EXP_DIRECTORY / ".cache"
DEFAULT_RESULTS = EXP_DIRECTORY / "results"


def select_test_samples(
        model: LogisticRegression,
        experiment: SplitDataset,
        dataset_id: str,
        use_fixed_set: bool = False,
        num_test_samples: int = 100
) -> Tuple[np.ndarray, float, float]:
    """
    Select test samples based on fixed or computed logic, and calculate losses.

    Parameters:
        model (LogisticRegression): The trained logistic regression model.
        experiment (SplitDataSet): The dataset with train and test splits.
        dataset_id (str): Identifier for the dataset (e.g., "hospital", "spam", "dogfish").
        use_fixed_set (bool): If True, use fixed indices for test samples (default: False).
        num_test_samples (int): Number of test samples to select. If -1, use all test samples.

    Returns:
        Tuple[np.ndarray, float, float]:
            - Selected test sample indices (fixed_test).
            - Total self-loss across all training samples.
            - Total test loss for the selected test samples.
    """
    fixed_test: Optional[np.ndarray] = None
    test_losses = model.model.get_model_losses(experiment.test.features, experiment.test.labels)
    self_losses = model.get_self_losses()

    # Fixed test samples for specific datasets
    fixed_indices_map = {
        "hospital": [2267, 54826, 66678, 41567, 485, 25286],
        "spam": [386, 842, 326, 766, 65, 642],
        "mnist_small": [6172, 2044, 2293, 5305, 324, 3761],
        "mnist": [9009, 1790, 2293, 5844, 8977, 9433],
        "dogfish": [300, 339, 222, 520, 323, 182],
    }

    # Use fixed indices if the flag is set
    if use_fixed_set and dataset_id in fixed_indices_map:
        fixed_test = np.array(fixed_indices_map[dataset_id], dtype=int)
    else:
        # Predict probabilities and calculate test losses
        test_losses = model.model.get_model_losses(experiment.test.features, experiment.test.labels)
        self_losses = model.get_self_losses()

        # Use all test samples if num_test_samples is -1
        if num_test_samples < 0:
            fixed_test = np.arange(len(experiment.test.features))
        else:
            # Identify high-loss and random-loss samples
            np.random.seed(1)
            argsort = np.argsort(test_losses)
            high_loss = argsort[
                        -min(num_test_samples // 2, len(argsort)):]  # Up to num_test_samples // 2 high-loss samples
            random_loss = np.random.choice(
                argsort[:-len(high_loss)],
                min(len(argsort) - len(high_loss), num_test_samples - len(high_loss)),
                replace=False
            )
            fixed_test = np.concatenate((high_loss, random_loss))

    # Calculate losses
    test_loss = np.sum(test_losses[fixed_test])
    self_loss = np.sum(self_losses)

    return fixed_test, self_loss, test_loss


# Enum for specifying force refresh levels
class ForceRefreshLevel(Enum):
    MODEL_WEIGHTS = 0
    HESSIAN_INVERSE = 1
    INFLUENCES = 2
    DEFAULT = 3


def get_default_regularization(dataset_name: Union[str, DatasetName, EmbeddedDataset]) -> Tuple[float, str]:
    """
    Determine default regularization values based on the dataset name.

    Returns:
        Tuple[float, str]: Regularization strength (lambda) and type (e.g., L2).
    """
    if dataset_name == "DogFish":
        return 2.2e-2, "L2"  # λ/n = 2.2 × 10^-2
    elif dataset_name == "Enron":
        return 1.0e-3, "L2"  # λ/n = 1.0 × 10^-3
    elif dataset_name == "Diabetes":
        return 2.2e-4, "L2"  # λ/n = 2.2 × 10^-4
    elif dataset_name == "CDR":
        return 1.0e-4, "L2"  # λ/n = 1.0 × 10^-4
    elif dataset_name == "MNIST":
        return 1.0e-3, "L2"  # λ/n = 1.0 × 10^-3
    elif dataset_name == "MultiNLI":
        return 1.0e-4, "L2"  # λ/n = 1.0 × 10^-4
    elif isinstance(dataset_name, EmbeddedDataset):
        return 1.0e-5, "L2"  # Give embedded datasets a small default regularization
    else:
        raise ValueError(f"Default regularization not defined for dataset: {dataset_name}")


@dataclass
class ExperimentalPipeline:
    dataset_name: Union[str, DatasetName, EmbeddedDataset]
    force_refresh: ForceRefreshLevel = ForceRefreshLevel.MODEL_WEIGHTS  # Default to load the largest step
    regularization: Optional[float] = None
    reg_type: Optional[Union[str, RegularizationType]] = None
    num_test_samples: int = 100
    num_sizes: int = 40
    verbosity: int = 3
    regression: LogisticRegression = None
    influence_scores: InfluenceFunctions = None
    fixed_test: np.ndarray = None
    self_loss: float = None
    test_loss: float = None
    removal_strategies: List[RemovalSetGenerator] = None
    experiment_results: List[List[ExperimentResult]] = None
    dataloader_flags: Optional[Dict[str, Any]] = field(default_factory=dict)
    cache_dir: Path = DEFAULT_CACHE
    results_dir: Path = DEFAULT_RESULTS
    storage_allowance_mb: float = 25.0
    experiment_name: str = None
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    def __post_init__(self):
        # Set up cache directory
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(self.results_dir)
        if self.experiment_name is None:
            self.experiment_name = f"{self.dataset_name}"
        self.cache_file = self.cache_dir / self._get_filename()
        self.results_file = self.results_dir / self._get_filename()

        # Load dataset
        self.dataset = self._load_dataset()

        # Determine regularization and reg_type if not provided
        if self.regularization is None or self.reg_type is None:
            self.regularization, self.reg_type = get_default_regularization(self.dataset_name)

        if self.verbosity:
            print(f"Initialized experiment on {self.dataset_name=}:")
            print(f"{self.cache_file=}, {self.storage_allowance_mb=}, {self.reg_type=}, {self.regularization=}")
        # Load cache
        self.cache_data = self._load_cache()

    def _get_filename(self) -> str:
        return self.experiment_name + ".npz"

    def _load_dataset(self):
        """Load the dataset."""
        dataset = load_dataset(self.dataset_name, verbose=self.verbosity >= 2, **self.dataloader_flags)
        if self.max_train_samples is not None or self.max_test_samples is not None:
            dataset = dataset.subsample(self.max_train_samples, self.max_test_samples)
        return dataset

    def _load_cache(self):
        """
        Load cached data based on the specified force refresh level.

        Returns:
            dict: Loaded cache data, or an empty dict if no cache is loaded.
        """
        if not self.cache_file.exists():
            return {}

        cache_data = np.load(self.cache_file, allow_pickle=True)
        cache_fields = set(cache_data.keys())
        if self.verbosity >= 2:
            print(f"Loaded the following fields: {cache_fields}")
        loaded_data = {}

        # Load fields based on the force refresh level
        if self.force_refresh.value > ForceRefreshLevel.MODEL_WEIGHTS.value and "model_weights" in cache_fields:
            if self.verbosity >= 2:
                print("Loading model weights...")
            loaded_data["model_weights"] = cache_data["model_weights"]

        if self.force_refresh.value > ForceRefreshLevel.HESSIAN_INVERSE.value and "hessian_inverse" in cache_fields:
            if self.verbosity >= 2:
                print("Loading inverse hessian...")
            loaded_data["hessian_inverse"] = cache_data["hessian_inverse"]

        return loaded_data

    def _save_to_cache(self, new_data):
        """
        Save specified data to the cache file if it does not exceed the storage limit.

        Parameters:
            new_data (Dict[str, np.ndarray]): Key-value pairs to save in the cache.
        """

        # Estimate storage of current cache data
        def estimate_storage(array: np.ndarray) -> int:
            """Estimate storage size of a numpy array in bytes."""
            return array.nbytes  # Total bytes used by the array

        current_storage = sum(estimate_storage(array) for array in self.cache_data.values())

        # Estimate storage of new fields
        new_storage = sum(
            estimate_storage(array) for key, array in new_data.items() if key not in self.cache_data
        )

        # Check if the total storage exceeds the allowance
        total_storage_mb = (current_storage + new_storage) / (1024 ** 2)  # Convert to MB
        if total_storage_mb > self.storage_allowance_mb:
            return

        # Merge new data into cache and save
        cache_data = dict(self.cache_data, **new_data)
        np.savez(self.cache_file, **cache_data)

        # Update in-memory cache
        self.cache_data.update(new_data)

    def train_model(self):
        """Train logistic regression."""
        extended_weights = self.cache_data.get("model_weights")
        self.regression = train_logistic_regression(
            experiment=self.dataset,
            regularization=self.regularization,
            fit_intercept=False,
            reg_type=self.reg_type,
            verbose=self.verbosity >= 1,
            extended_model_weights=extended_weights,
        )

        # Cache the model weights
        self._save_to_cache({"model_weights": self.regression.model.extended_weights})

    def compute_influences(self):
        """Compute influence scores."""
        # Check if Hessian inverse is already in the cache
        hessian_inverse = self.cache_data.get("hessian_inverse")
        if hessian_inverse is None:
            # Compute and cache the Hessian inverse
            hessian_inverse = self.regression.compute_hessian_inv()
            self._save_to_cache({"hessian_inverse": hessian_inverse})

        # Compute influence scores
        self.influence_scores = compute_model_influences(
            regression=self.regression,
            experiment=self.dataset,
            verbose=self.verbosity >= 2,
        )

    def select_test_samples(self):
        """Select fixed test samples."""
        self.fixed_test, self.self_loss, self.test_loss = select_test_samples(
            model=self.regression,
            experiment=self.dataset,
            dataset_id=self.dataset_name,
            use_fixed_set=True,
            num_test_samples=self.num_test_samples,
        )

    def init_removal_strategies(self):
        """Initialize removal strategies."""
        model_influences = self.influence_scores.influence_scores
        self_losses = self.regression.get_self_losses()

        # Compute test loss gradients
        test_loss_gradients = self.regression.compute_gradients(
            self.dataset.test.features[self.fixed_test],
            self.dataset.test.labels[self.fixed_test],
        )
        test_loss_gradient = np.sum(test_loss_gradients, axis=0)

        # Compute test prediction gradients
        y_pred = self.regression.model.get_model_predictions(self.dataset.test.features[self.fixed_test])
        test_prediction_gradients = (
                self.dataset.test.features[self.fixed_test].T * y_pred * (1 - y_pred)
        ).T
        test_prediction_gradient = np.sum(test_prediction_gradients, axis=0)

        # Define removal strategies
        self.removal_strategies = [
            RandomRemovalSetGenerator(self.regression.n, num_sizes=self.num_sizes),
            ClusterRemovalSetGenerator(
                self.regression.n,
                features=self.regression.features,
                subset_selection_method="Random Feature",
                num_sizes=self.num_sizes,
            ),
            ClusterRemovalSetGenerator(
                self.regression.n,
                features=self.regression.features,
                subset_selection_method="L2 Distance",
                num_sizes=self.num_sizes,
            ),
            TopPercentileRemovalSetGenerator(
                self.regression.n,
                self_losses,
                metric_name="Loss",
                num_sizes=self.num_sizes,
            ),
            TopPercentileRemovalSetGenerator(
                self.regression.n,
                test_loss_gradient @ model_influences.T,
                metric_name="Positive Test-Loss Inf",
                num_sizes=self.num_sizes,
            ),
            TopPercentileRemovalSetGenerator(
                self.regression.n,
                -(test_loss_gradient @ model_influences.T),
                metric_name="Negative Test-Loss Inf",
                num_sizes=self.num_sizes,
            ),
            TopPercentileRemovalSetGenerator(
                self.regression.n,
                test_prediction_gradient @ model_influences.T,
                metric_name="Positive Test Inf",
                num_sizes=self.num_sizes,
            ),
            TopPercentileRemovalSetGenerator(
                self.regression.n,
                -(test_prediction_gradient @ model_influences.T),
                metric_name="Negative Test Inf",
                num_sizes=self.num_sizes,
            ),
        ]

    def run_experiments(self):
        """
        Run experiments on all removal strategies and removal sets.
        Results are stored in self.experiment_results and saved to a single npz file.

        This method loops over all removal strategies and, for each removal set,
        computes influence scores, Newton steps, and retrained weights.
        """
        if not self.removal_strategies:
            raise ValueError("Removal strategies are not initialized. Run init_removal_strategies first.")

        self.experiment_results = []  # Store results as a list of lists, one per strategy

        for strategy in self.removal_strategies:
            strategy_results = []  # Store results for this strategy
            self.experiment_results.append(strategy_results)

            if self.verbosity > 0:
                print(f"Running experiments for strategy: {strategy.name}")

            # Generate all removal sets
            removal_sets = strategy.generate_all_removal_sets()

            # Loop through removal sets with tqdm if verbosity is positive
            removal_iter = tqdm(removal_sets, desc=f"Processing {strategy.name}", disable=(self.verbosity == 0))

            for removal_set in removal_iter:
                # Run the experiment for this removal set
                result = run_experiment_on_removal_set(
                    original_regression=self.regression,
                    influence_scores=self.influence_scores,
                    indices_to_remove=removal_set,
                    removal_strategy=strategy.name,
                    experiment=self.dataset,
                    fixed_test_indices=self.fixed_test
                )
                strategy_results.append(result)  # Append result to strategy's list

            if self.verbosity > 0:
                print(f"Finished strategy: {strategy.name}")

            # Save all results to a single npz file
            self._save_all_results_to_npz()

    def _save_all_results_to_npz(self):
        """
        Save all experiment results to a single npz file.

        This method saves `self.experiment_results` to `self.results_file`.
        """
        # Ensure the results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Prepare a dictionary where each key is a strategy name and value is its results as dictionaries
        results_dict = {}
        for strategy, strategy_results in zip(self.removal_strategies, self.experiment_results):
            results_dict[strategy.name] = [
                result.to_dict() for result in strategy_results
            ]

        results_dict["original_model"] = self.regression.model.extended_weights

        # Save the dictionary to an npz file
        np.savez_compressed(self.results_file, **results_dict)

        if self.verbosity > 2:
            print(f"All results saved to {self.results_file}")

    def run(self):
        """Run the experimental pipeline."""
        steps = [
            (self._load_cache, "Load Cache"),
            (self.train_model, "Train Model"),
            (self.compute_influences, "Compute Influences"),
            (self.select_test_samples, "Select Test Samples"),
            (self.init_removal_strategies, "Init Removal Strategies"),
            (self.run_experiments, "Running Experiments")
        ]
        for step, name in steps:
            if self.verbosity:
                print(f"Running {name=}...", end=" ")
            t0 = time.time()
            step()
            if self.verbosity:
                print(f"Completed in {time.time() - t0:.1f} Seconds.")
