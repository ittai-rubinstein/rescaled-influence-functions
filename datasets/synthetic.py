from dataclasses import dataclass
from typing import Optional
import numpy as np

from datasets.common import SplitDataset, Dataset


def create_synthetic_logistic_regression_data(
        n: int,
        d: int,
        num_test: Optional[int] = None,
        ground_truth_model: Optional[np.ndarray] = None
) -> SplitDataset:
    """
    Generate a synthetic logistic regression dataset.

    The function creates a training dataset of `n` samples and a test dataset
    of `num_test` samples (defaulting to int(0.2 * n)). Each sample's feature vector
    is drawn from a standard multivariate normal distribution N(0, I_d). Labels are
    generated using a logistic model defined by a ground truth weight vector.

    Args:
        n (int): Number of training samples.
        d (int): Dimensionality of each sample.
        num_test (Optional[int]): Number of test samples. Defaults to int(0.2 * n).
        ground_truth_model (Optional[np.ndarray]): Ground truth weight vector. If not provided,
            it is drawn uniformly from the unit sphere in ℝᵈ.

    Returns:
        SplitDataset: A dataclass containing the training and test datasets.
    """
    if num_test is None:
        num_test = int(0.2 * n)

    # If no ground truth model is provided, generate one uniformly from the unit sphere.
    if ground_truth_model is None:
        ground_truth_model = np.random.randn(d)
        ground_truth_model = ground_truth_model / np.linalg.norm(ground_truth_model)
    else:
        ground_truth_model = np.array(ground_truth_model)

    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    # Generate training data
    train_features = np.random.randn(n, d)
    train_logits = train_features.dot(ground_truth_model)
    train_prob = sigmoid(train_logits)
    train_labels = (np.random.rand(n) < train_prob).astype(int)

    # Generate test data
    test_features = np.random.randn(num_test, d)
    test_logits = test_features.dot(ground_truth_model)
    test_prob = sigmoid(test_logits)
    test_labels = (np.random.rand(num_test) < test_prob).astype(int)

    train_dataset = Dataset(features=train_features, labels=train_labels)
    test_dataset = Dataset(features=test_features, labels=test_labels)

    return SplitDataset(train=train_dataset, test=test_dataset)
