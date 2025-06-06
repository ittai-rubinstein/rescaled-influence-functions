import time
from typing import Union, Optional

import numpy as np

from datasets.common import SplitDataset
from src.logistic_regression import RegularizationType, LogisticRegression


def train_logistic_regression(
        experiment: SplitDataset,
        regularization: float,
        fit_intercept: bool = False,
        reg_type: Union[RegularizationType, str] = "L2",
        verbose: bool = True,
        extended_model_weights: Optional[np.ndarray] = None
) -> LogisticRegression:
    """
    Train a logistic regression model on a given experiment and evaluate its accuracy on the test set.

    Parameters:
        experiment (SplitDataSet): Dataset object with train and test splits.
        regularization (float): Regularization strength (default: 2.2E-4).
        fit_intercept (bool): Whether to fit an intercept (default: False).
        reg_type (str): Regularization type (default: "L2").
        verbose (bool): If True, print training and evaluation details.

    Returns:
        Tuple[LogisticRegression, float, float]: A tuple containing:
            - Trained LogisticRegression model.
            - Test accuracy (float) as a percentage.
            - Training time (float) in seconds.
    """
    t0 = time.time()

    # Initialize and fit the model
    model = LogisticRegression(
        experiment.train.features,
        experiment.train.labels,
        regularization=regularization,
        fit_intercept=fit_intercept,
        reg_type=reg_type
    )
    if extended_model_weights is not None:
        model.model.extended_weights = extended_model_weights
        return model
    model.fit()

    # Calculate test accuracy
    test_accuracy = model.model.get_accuracy(experiment.test.features, experiment.test.labels)

    # Measure training time
    training_time = time.time() - t0

    # Verbose output
    if verbose:
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Training took {training_time:.2f} seconds")

    return model
