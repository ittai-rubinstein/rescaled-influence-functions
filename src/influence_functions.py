import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from datasets.common import SplitDataset
from src.logistic_regression import LogisticRegression


@dataclass
class InfluenceFunctions:
    """
    Data class to store influence-related scores and intermediate computations.
    """
    H: np.ndarray  # Hessian matrix
    H_inv: np.ndarray  # Inverse of the Hessian matrix
    y: np.ndarray  # Predicted probabilities (y_i for each sample)
    alpha: np.ndarray  # Predicted model variance (y_i * (1 - y_i))
    gradients: np.ndarray  # Gradients for the training set
    influence_scores: np.ndarray  # Model influence scores
    rescaled_influence_scores: np.ndarray  # Rescaled model influence scores
    gram_matrix: Optional[np.ndarray] = None  # where G[j, l] = x_j^T H^{-1} x_l
    # scaled_feature_matrix: np.ndarray #V := H^{-1} X, where X is the feature matrix


def compute_model_influences(
    regression: LogisticRegression,
    experiment: SplitDataset,
    verbose: bool = True,
    compute_gram_matrix: bool = True
) -> InfluenceFunctions:
    """
    Compute the Hessian, its inverse, model influences, and rescaled influences.

    Parameters:
        regression (LogisticRegression): The trained logistic regression model.
        experiment (SplitDataSet): The dataset with train features and labels.
        verbose (bool): If True, prints progress and timing for each step.

    Returns:
        InfluenceFunctions: A data class containing influence scores and intermediate results.
    """
    # Initialize timers
    start_time = time.time()

    # Compute Hessian
    if verbose:
        print("Computing the Hessian...", end=" ")
    H = regression.n * regression.compute_hessian()
    if verbose:
        print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    # Compute Hessian inverse
    start_time = time.time()
    if verbose:
        print("Inverting the Hessian...", end=" ")
    H_inv = regression.compute_hessian_inv() / regression.n
    if verbose:
        print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    # Compute gradients for the training set
    start_time = time.time()
    if verbose:
        print("Computing trainset gradients...", end=" ")
    trainset_gradients = regression.compute_gradients(experiment.train.features, experiment.train.labels)
    if verbose:
        print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    # Compute model influence scores
    start_time = time.time()
    if verbose:
        print("Computing model influences...", end=" ")
    influence_scores = trainset_gradients @ H_inv
    if verbose:
        print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    # Compute predictions on the trainset (y_i)
    y = regression.model.get_model_predictions(experiment.train.features)  # Predicted probabilities

    # Compute alpha_i = y_i * (1 - y_i)
    alpha = y * (1 - y)

    # Compute rescaled model influences using numpy
    start_time = time.time()
    if verbose:
        print("Computing rescaled model influences...", end=" ")

    # Compute x_i^T H_inv x_i for all training points in one operation
    X = experiment.train.features  # Feature matrix (n_samples x n_features)
    x_H_inv = X @ H_inv  # Precompute H_inv X (n_samples x n_features)
    x_H_inv_x = np.einsum('ij,ij->i', x_H_inv, X)  # Compute x_i^T H_inv x_i for each sample

    # Compute scaling factors and rescale the influence scores
    scale_factors = 1 - (x_H_inv_x * alpha)
    rescaled_influence_scores = influence_scores / scale_factors[:, np.newaxis]

    if verbose:
        print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    # Compute Gram matrix (G[j, l] = x_j^T H_inv x_l) and scaled feature matrix (V = H_inv X)
    start_time = time.time()
    if verbose:
        print("Computing Gram matrix and scaled feature matrix...", end=" ")

    gram_matrix = None
    if compute_gram_matrix:
        gram_matrix = X @ x_H_inv.T  # Compute G
    # scaled_feature_matrix = x_H_inv  # V = H_inv X

    if verbose:
        print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    # Return results in a data class
    return InfluenceFunctions(
        H=H,
        H_inv=H_inv,
        y=y,
        alpha=alpha,
        gradients=trainset_gradients,
        influence_scores=influence_scores,
        rescaled_influence_scores=rescaled_influence_scores,
        gram_matrix=gram_matrix,
        # scaled_feature_matrix=scaled_feature_matrix
    )
