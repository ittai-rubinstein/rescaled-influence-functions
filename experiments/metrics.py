from dataclasses import dataclass
from typing import Dict
from typing import Protocol, Tuple

import numpy as np

from datasets.common import SplitDataset
from src.logistic_regression import LogisticRegression, LogisticRegressionModel


@dataclass
class MetricResult:
    """
    Data class to store results for a computed metric.
    """
    ground_truth: float
    ground_truth_linear: float
    influence_function_estimate: float
    influence_function_estimate_linear: float
    rescaled_influence_function_estimate: float
    rescaled_influence_function_estimate_linear: float
    newton_step_estimate: float
    newton_step_estimate_linear: float

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Convert the MetricResult to a dictionary for saving to an npz file.
        """
        return {
            "ground_truth": np.array([self.ground_truth]),
            "ground_truth_linear": np.array([self.ground_truth_linear]),
            "influence_function_estimate": np.array([self.influence_function_estimate]),
            "influence_function_estimate_linear": np.array([self.influence_function_estimate_linear]),
            "rescaled_influence_function_estimate": np.array([self.rescaled_influence_function_estimate]),
            "rescaled_influence_function_estimate_linear": np.array([self.rescaled_influence_function_estimate_linear]),
            "newton_step_estimate": np.array([self.newton_step_estimate]),
            "newton_step_estimate_linear": np.array([self.newton_step_estimate_linear]),
        }

    @staticmethod
    def from_dict(data: Dict[str, np.ndarray]) -> "MetricResult":
        """
        Create a MetricResult from a dictionary.
        """
        return MetricResult(
            ground_truth=float(data["ground_truth"][0]),
            ground_truth_linear=float(data["ground_truth_linear"][0]),
            influence_function_estimate=float(data["influence_function_estimate"][0]),
            influence_function_estimate_linear=float(data["influence_function_estimate_linear"][0]),
            rescaled_influence_function_estimate=float(data["rescaled_influence_function_estimate"][0]),
            rescaled_influence_function_estimate_linear=float(data["rescaled_influence_function_estimate_linear"][0]),
            newton_step_estimate=float(data["newton_step_estimate"][0]),
            newton_step_estimate_linear=float(data["newton_step_estimate_linear"][0]),
        )



class Metric(Protocol):
    """
    Interface for a metric to be computed in the experiment.
    Each metric must implement exact and approximate computation methods.
    """

    name: str

    def compute_exact(
            self, original_regression: LogisticRegression, new_model: LogisticRegressionModel, **kwargs
    ) -> float:
        """
        Compute the exact change in this metric.

        Parameters:
            original_regression (LogisticRegression): The original model before retraining.
            new_model (LogisticRegressionModel): The retrained model.
            kwargs: Additional inputs for specific metrics.

        Returns:
            float: The exact metric value.
        """
        raise NotImplementedError

    def compute_approximate(
            self, delta_model_weights: np.ndarray, original_regression: LogisticRegression, **kwargs
    ) -> float:
        """
        Compute the approximate change in this metric using a 1st order approximation.

        Parameters:
            delta_model_weights (np.ndarray): The estimated change in model weights.
            original_regression (LogisticRegression): The original model before retraining.
            kwargs: Additional inputs for specific metrics.

        Returns:
            float: The approximate metric value.
        """
        raise NotImplementedError

    def compute_approximate_and_exact(
            self,
            original_regression: LogisticRegression,
            new_model: LogisticRegressionModel,
            **kwargs
    ) -> Tuple[float, float]:
        """
        Computes both the approximate and exact values of the metric.

        Args:
            original_regression (LogisticRegression): The original model wrapper.
            new_model (LogisticRegressionModel): The retrained model instance.
            kwargs: Additional arguments for the specific metric.

        Returns:
            Tuple[float, float]: A tuple where the first element is the approximate value
                                 and the second is the exact value.
        """
        delta_model_weights = new_model.weights - original_regression.model.weights
        approximate = self.compute_approximate(delta_model_weights, original_regression, **kwargs)
        exact = self.compute_exact(original_regression, new_model, **kwargs)
        return approximate, exact


class InfluenceOnTotalLoss(Metric):
    name = "influence_on_total_loss"

    def compute_exact(
            self, original_regression: LogisticRegression, new_model: LogisticRegressionModel, **kwargs
    ) -> float:
        original_loss = np.sum(original_regression.get_self_losses())
        new_loss = np.sum(new_model.get_model_losses(original_regression.features, original_regression.labels))
        return new_loss - original_loss

    def compute_approximate(
            self, delta_model_weights: np.ndarray, original_regression: LogisticRegression, **kwargs
    ) -> float:
        hessian = original_regression.compute_hessian() * original_regression.n
        return 0.5 * float(delta_model_weights.T @ hessian @ delta_model_weights)


class InfluenceOnTestPredictions(Metric):
    name = "influence_on_test_predictions"

    def compute_exact(
            self, original_regression: LogisticRegression, new_model: LogisticRegressionModel, experiment: SplitDataset,
            fixed_test_indices: np.ndarray, **kwargs
    ) -> float:
        original_predictions = original_regression.model.get_model_predictions(
            experiment.test.features[fixed_test_indices]
        )
        retrained_predictions = new_model.get_model_predictions(
            experiment.test.features[fixed_test_indices]
        )
        return np.sum(retrained_predictions) - np.sum(original_predictions)

    def compute_approximate(
            self, delta_model_weights: np.ndarray, original_regression: LogisticRegression, experiment: SplitDataset,
            fixed_test_indices: np.ndarray, **kwargs
    ) -> float:
        y_pred = original_regression.model.get_model_predictions(experiment.test.features[fixed_test_indices])
        test_prediction_gradients = (
                experiment.test.features[fixed_test_indices].T * y_pred * (1 - y_pred)
        ).T
        test_prediction_gradient = np.sum(test_prediction_gradients, axis=0)
        return test_prediction_gradient @ delta_model_weights


class InfluenceOnTestFixedLoss(Metric):
    name = "influence_on_test_fixed_loss"

    def compute_exact(
            self, original_regression: LogisticRegression, new_model: LogisticRegressionModel, experiment: SplitDataset,
            fixed_test_indices: np.ndarray, **kwargs
    ) -> float:
        original_losses = original_regression.model.get_model_losses(
            experiment.test.features[fixed_test_indices], experiment.test.labels[fixed_test_indices]
        )
        retrained_losses = new_model.get_model_losses(
            experiment.test.features[fixed_test_indices], experiment.test.labels[fixed_test_indices]
        )
        return np.sum(retrained_losses) - np.sum(original_losses)

    def compute_approximate(
            self, delta_model_weights: np.ndarray, original_regression: LogisticRegression, experiment: SplitDataset,
            fixed_test_indices: np.ndarray, **kwargs
    ) -> float:
        test_loss_gradients = original_regression.compute_gradients(
            experiment.test.features[fixed_test_indices], experiment.test.labels[fixed_test_indices]
        )
        test_loss_gradient = np.sum(test_loss_gradients, axis=0)
        return test_loss_gradient @ delta_model_weights


class InfluenceOnTestTotalLoss(Metric):
    name = "influence_on_test_total_loss"

    def compute_exact(
            self, original_regression: LogisticRegression, new_model: LogisticRegressionModel, experiment: SplitDataset,
            **kwargs
    ) -> float:
        original_losses = original_regression.model.get_model_losses(
            experiment.test.features, experiment.test.labels
        )
        retrained_losses = new_model.get_model_losses(
            experiment.test.features, experiment.test.labels
        )
        return np.sum(retrained_losses) - np.sum(original_losses)

    def compute_approximate(
            self, delta_model_weights: np.ndarray, original_regression: LogisticRegression, experiment: SplitDataset, **kwargs
    ) -> float:
        test_loss_gradients = original_regression.compute_gradients(
            experiment.test.features, experiment.test.labels
        )
        test_loss_gradient = np.sum(test_loss_gradients, axis=0)
        return test_loss_gradient @ delta_model_weights
