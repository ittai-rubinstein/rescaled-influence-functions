from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from datasets.common import SplitDataset
from src.influence_functions import InfluenceFunctions
from experiments.metrics import MetricResult, InfluenceOnTestPredictions, InfluenceOnTestFixedLoss, \
    InfluenceOnTestTotalLoss, InfluenceOnTotalLoss
from src.logistic_regression import LogisticRegression, LogisticRegressionModel
from src.newton_step import straightforward_newton_step, woodbury_newton_step
from src.removal_effects import refine_logistic_regression_model


@dataclass
class ExperimentResult:
    """
    Data class to store results of experiments on removal sets.
    """
    removal_strategy: str
    removal_set_size: int
    influence_sum: np.ndarray
    rescaled_influence_sum: np.ndarray
    newton_step: np.ndarray
    retrained_weights: np.ndarray
    removed_indices: np.ndarray  # Indices of removed train samples
    fixed_test_indices: np.ndarray  # Fixed test indices
    train_loss_retained: float  # Total loss on retained train samples
    train_loss_removed: float  # Total loss on removed train samples
    test_loss_fixed: float  # Total loss on fixed test samples
    metrics: Optional[Dict[str, MetricResult]] = None  # Dict from metric name to MetricResult

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Convert the ExperimentResult to a dictionary for saving to an npz file.

        Returns:
            Dict[str, np.ndarray]: Dictionary representation.
        """
        result_dict = {
            "removal_strategy": np.array([self.removal_strategy], dtype="S"),  # Store string as bytes
            "removal_set_size": np.array([self.removal_set_size]),
            "influence_sum": self.influence_sum,
            "rescaled_influence_sum": self.rescaled_influence_sum,
            "newton_step": self.newton_step,
            "retrained_weights": self.retrained_weights,
            "removed_indices": self.removed_indices,
            "fixed_test_indices": self.fixed_test_indices,
            "train_loss_retained": np.array([self.train_loss_retained]),
            "train_loss_removed": np.array([self.train_loss_removed]),
            "test_loss_fixed": np.array([self.test_loss_fixed]),
        }
        if self.metrics:
            # Add metrics as sub-dictionaries
            for metric_name, metric_result in self.metrics.items():
                metric_key = f"metric_{metric_name}"
                result_dict[metric_key] = metric_result.to_dict()
        return result_dict

    @staticmethod
    def from_dict(data: Dict[str, np.ndarray]) -> "ExperimentResult":
        """
        Create an ExperimentResult from a dictionary.

        Parameters:
            data (Dict[str, np.ndarray]): Dictionary representation of the ExperimentResult.

        Returns:
            ExperimentResult: Reconstructed ExperimentResult object.
        """
        data = {key: value for key, value in data}
        # Extract basic fields
        metrics = {}
        for key, value in data.items():
            if key.startswith("metric_"):
                metric_name = key[len("metric_"):]
                metrics[metric_name] = MetricResult.from_dict(value)

        return ExperimentResult(
            removal_strategy=data["removal_strategy"][0].decode("utf-8"),  # Convert bytes back to string
            removal_set_size=int(data["removal_set_size"][0]),
            influence_sum=data["influence_sum"],
            rescaled_influence_sum=data["rescaled_influence_sum"],
            newton_step=data["newton_step"],
            retrained_weights=data["retrained_weights"],
            removed_indices=data["removed_indices"],
            fixed_test_indices=data["fixed_test_indices"],
            train_loss_retained=float(data["train_loss_retained"][0]),
            train_loss_removed=float(data["train_loss_removed"][0]),
            test_loss_fixed=float(data["test_loss_fixed"][0]),
            metrics=metrics if metrics else None,
        )


def run_experiment_on_removal_set(
        original_regression: LogisticRegression,
        influence_scores: InfluenceFunctions,
        indices_to_remove: np.ndarray,
        removal_strategy: str,
        fixed_test_indices: np.ndarray,
        experiment: SplitDataset
) -> ExperimentResult:
    """
    Run the experiment for a given removal set, computing influence scores, newton steps,
    retrained weights, and metrics.

    Parameters:
        original_regression (LogisticRegression): The original model.
        influence_scores (InfluenceFunctions): Precomputed influence-related scores.
        indices_to_remove (np.ndarray): Indices of samples to remove.
        removal_strategy (str): The name of the removal strategy.
        fixed_test_indices (np.ndarray): Indices of fixed test samples.
        experiment (SplitDataset): The dataset with train/test splits.

    Returns:
        ExperimentResult: The result of the experiment.
    """
    # Compute influence sums
    influence_sum = np.sum(influence_scores.influence_scores[indices_to_remove], axis=0)
    rescaled_influence_sum = np.sum(
        influence_scores.rescaled_influence_scores[indices_to_remove], axis=0
    )

    # Compute the Newton step
    k = len(indices_to_remove)
    d = original_regression.d  # Dimensionality of the model

    if k > d:
        updated_model = straightforward_newton_step(original_regression, indices_to_remove)
    else:
        updated_model = woodbury_newton_step(
            original_regression, indices_to_remove,
            gram_matrix=influence_scores.gram_matrix
        )

    newton_step = updated_model.extended_weights - original_regression.model.extended_weights

    # Retrain the model
    retrained_model = refine_logistic_regression_model(
        original_regression,
        indices_to_remove=indices_to_remove,
        approximate_updated_weights=updated_model.extended_weights,
    )

    retrained_weights = retrained_model.model.extended_weights

    # Compute losses after retraining
    retained_indices = np.setdiff1d(np.arange(original_regression.n), indices_to_remove)

    train_loss_retained = np.sum(
        retrained_model.model.get_model_losses(original_regression.features[retained_indices], original_regression.labels[retained_indices])
    )
    train_loss_removed = np.sum(
        retrained_model.model.get_model_losses(original_regression.features[indices_to_remove], original_regression.labels[indices_to_remove])
    )
    test_loss_fixed = np.sum(
        retrained_model.model.get_model_losses(experiment.test.features[fixed_test_indices],
                                               experiment.test.labels[fixed_test_indices])
    )

    newton_model = updated_model
    influence_function_model = LogisticRegressionModel(
        weights=original_regression.model.weights + influence_sum,
        regularization=original_regression.model.regularization,
        reg_type=original_regression.model.reg_type
    )
    rescaled_influence_function_model = LogisticRegressionModel(
        weights=original_regression.model.weights + rescaled_influence_sum,
        regularization=original_regression.model.regularization,
        reg_type=original_regression.model.reg_type
    )
    # Compute metrics
    metrics = {}
    for metric_class in [InfluenceOnTestPredictions, InfluenceOnTestFixedLoss, InfluenceOnTestTotalLoss,
                         InfluenceOnTotalLoss]:
        metric = metric_class()

        # Compute exact and linear approximation using retrained model
        ground_truth_linear, ground_truth = metric.compute_approximate_and_exact(
            original_regression, retrained_model.model,
            experiment=experiment, fixed_test_indices=fixed_test_indices,
            train_loss_retained=train_loss_retained, train_loss_removed=train_loss_removed
        )

        # Influence function estimate
        influence_function_estimate_linear, influence_function_estimate = metric.compute_approximate_and_exact(
            original_regression, influence_function_model,
            experiment=experiment, fixed_test_indices=fixed_test_indices,
            train_loss_retained=train_loss_retained, train_loss_removed=train_loss_removed
        )

        # Rescaled influence function estimate
        rescaled_influence_function_estimate_linear, rescaled_influence_function_estimate = metric.compute_approximate_and_exact(
            original_regression, rescaled_influence_function_model,
            experiment=experiment, fixed_test_indices=fixed_test_indices,
            train_loss_retained=train_loss_retained, train_loss_removed=train_loss_removed
        )

        # Newton step estimate
        newton_step_estimate_linear, newton_step_estimate = metric.compute_approximate_and_exact(
            original_regression, newton_model,
            experiment=experiment, fixed_test_indices=fixed_test_indices,
            train_loss_retained=train_loss_retained, train_loss_removed=train_loss_removed
        )

        metrics[metric.name] = MetricResult(
            ground_truth=ground_truth,
            ground_truth_linear=ground_truth_linear,
            influence_function_estimate=influence_function_estimate,
            influence_function_estimate_linear=influence_function_estimate_linear,
            rescaled_influence_function_estimate=rescaled_influence_function_estimate,
            rescaled_influence_function_estimate_linear=rescaled_influence_function_estimate_linear,
            newton_step_estimate=newton_step_estimate,
            newton_step_estimate_linear=newton_step_estimate_linear,
        )

    return ExperimentResult(
        removal_strategy=removal_strategy,
        removal_set_size=k,
        influence_sum=influence_sum,
        rescaled_influence_sum=rescaled_influence_sum,
        newton_step=newton_step,
        retrained_weights=retrained_weights,
        removed_indices=indices_to_remove,
        fixed_test_indices=fixed_test_indices,
        train_loss_retained=train_loss_retained,
        train_loss_removed=train_loss_removed,
        test_loss_fixed=test_loss_fixed,
        metrics=metrics,
    )
