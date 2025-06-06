from collections import namedtuple
from typing import List, Tuple
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.logistic_regression import LogisticRegression


def refine_logistic_regression_model(
        original_regression: LogisticRegression,
        indices_to_remove: np.ndarray,
        approximate_updated_weights: Optional[np.ndarray] = None
) -> LogisticRegression:
    """
    Refine a logistic regression model by removing specified samples, performing a Newton step,
    and fitting the updated model with warm start.

    Parameters:
        original_regression (LogisticRegression): The original logistic regression object.
        indices_to_remove (np.ndarray): Array of integers specifying the indices of samples to remove.

    Returns:
        LogisticRegression: The refined logistic regression object.
    """
    # Step 1: Perform a Newton step
    if approximate_updated_weights is None:
        approximate_updated_weights = original_regression.model.extended_weights

    # Step 2: Create a new LogisticRegression object with retained samples
    retained_indices = np.setdiff1d(np.arange(original_regression.features.shape[0]), indices_to_remove)
    retained_features = original_regression.features[retained_indices]
    retained_labels = original_regression.labels[retained_indices]
    refined_model = LogisticRegression(
        features=retained_features,
        labels=retained_labels,
        regularization=original_regression.model.regularization,
        reg_type=original_regression.model.reg_type,
        fit_intercept=original_regression.fit_intercept
    )

    # Step 3: Set the updated weights and fit with warm start
    refined_model.model.extended_weights = approximate_updated_weights
    refined_model.fit(warm_start=True)

    # Step 4: Return the refined model
    return refined_model


def compute_adjusted_metrics(
        H: np.ndarray,
        predictions: np.ndarray,
        actuals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute adjusted metrics for visualization: scale and residual scatter.

    Parameters:
        H (np.ndarray): Hessian matrix of shape (d, d), where d is the number of features.
        predictions (np.ndarray): Array of predicted changes, shape (n, d), where n is the number of pairs.
        actuals (np.ndarray): Array of actual changes, shape (n, d), where n is the number of pairs.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays of shape (n, 2):
            - Scale plot points: (<actual, H actual>, <actual, H predicted>).
            - Residual scatter points: (<actual, H actual>, residual).
    """
    n, d = predictions.shape
    assert actuals.shape == (n, d), "Predictions and actuals must have the same shape."
    assert H.shape == (d, d), "Hessian matrix must have shape (d, d)."

    # Compute Hessian-adjusted inner products
    pred_H_actual = np.einsum('nd,df,nf->n', predictions, H, actuals)
    pred_H_pred = np.einsum('nd,df,nf->n', predictions, H, predictions)
    actual_H_actual = np.einsum('nd,df,nf->n', actuals, H, actuals)

    # Scale plot points
    scale_points = np.vstack((actual_H_actual, pred_H_actual)).T

    # Residual scatter points
    residual = pred_H_pred - (pred_H_actual ** 2 / actual_H_actual)
    residual_points = np.vstack((actual_H_actual, residual)).T

    return scale_points, residual_points


Experiment = namedtuple("Experiment", ["predictions", "actuals", "name"])

#
# def plot_metrics(
#         H_matrices: List[Tuple[str, np.ndarray]],
#         experiments: List[Experiment]
# ) -> None:
#     """
#     Plot metrics for a list of H matrices and experiments.
#
#     Parameters:
#         H_matrices (List[Tuple[str, np.ndarray]]): List of (name, H matrix) pairs.
#         experiments (List[Experiment]): List of named tuples containing predictions, actuals, and experiment names.
#
#     Returns:
#         None
#     """
#     num_H = len(H_matrices)
#     num_experiments = len(experiments)
#     colors = plt.cm.get_cmap("tab10", num_experiments)  # Unique colors for experiments
#
#     # Set up the figure and axes
#     fig, axes = plt.subplots(num_H, 2, figsize=(12, 4 * num_H), constrained_layout=True)
#     if num_H == 1:
#         axes = [axes]  # Ensure axes is always a list for consistent iteration
#
#     # Iterate over each H matrix
#     for h_idx, (H_name, H) in enumerate(H_matrices):
#         for exp_idx, experiment in enumerate(experiments):
#             # Compute metrics
#             scale_points, residual_points = compute_adjusted_metrics(
#                 H=H,
#                 predictions=experiment.predictions,
#                 actuals=experiment.actuals
#             )
#
#             # Scale Plot (left subplot for this H)
#             axes[h_idx][0].scatter(
#                 scale_points[:, 0],
#                 scale_points[:, 1],
#                 label=experiment.name,
#                 color=colors(exp_idx),
#                 alpha=0.6
#             )
#
#             # Residual Plot (right subplot for this H)
#             axes[h_idx][1].scatter(
#                 residual_points[:, 0],
#                 residual_points[:, 1],
#                 label=experiment.name,
#                 color=colors(exp_idx),
#                 alpha=0.6
#             )
#
#         # Add titles and labels
#         axes[h_idx][0].set_title(f"Scale Plot for {H_name}")
#         axes[h_idx][0].set_xlabel("<actual, H actual>")
#         axes[h_idx][0].set_ylabel("<actual, H predicted>")
#         axes[h_idx][0].legend()
#
#         axes[h_idx][1].set_title(f"Residual Plot for {H_name}")
#         axes[h_idx][1].set_xlabel("<actual, H actual>")
#         axes[h_idx][1].set_ylabel("Residual")
#         axes[h_idx][1].legend()
#
#     plt.show()
