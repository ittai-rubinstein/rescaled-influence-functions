from typing import Optional

import numpy as np

from src.logistic_regression import LogisticRegressionModel, LogisticRegression, RegularizationType


def update_gram_matrix(
        G: np.ndarray,
        alpha_i: float,
        i: int
) -> np.ndarray:
    """
    Updates the Gram matrix G when the matrix H is updated to H - alpha_i * x_i x_i^T.

    Args:
        G (np.ndarray): The original Gram matrix of shape (n, n), where G[j, l] = x_j^T H^{-1} x_l.
        alpha_i (float): Scalar alpha_i defining the rank-1 update H_i = alpha_i * x_i x_i^T.
        i (int): Index of the vector x_i involved in the update.

    Returns:
        np.ndarray: The updated Gram matrix of shape (n, n).
    """
    n = G.shape[0]
    assert G.shape == (n, n), "Gram matrix G must be square."
    assert 0 <= i < n, "Index i must be valid for the Gram matrix."

    G_i = G[:, i]  # Extract the i-th column (or row, since G is symmetric)
    G_ii = G[i, i]  # Extract the diagonal element G[i, i]

    # Compute the denominator 1 - alpha_i * G[i, i]
    denominator = 1 - alpha_i * G_ii
    if abs(denominator) < 1e-10:
        raise ValueError("Update results in numerical instability (denominator close to zero).")

    # Compute the outer product G_i * G_i^T
    correction = (alpha_i / denominator) * np.outer(G_i, G_i)

    # Update the Gram matrix
    G_updated = G + correction

    return G_updated


def update_scaled_feature_matrix(
        V: np.ndarray,
        G: np.ndarray,
        alpha_i: float,
        i: int
) -> np.ndarray:
    """
    Updates the scaled feature matrix V := H^{-1} X when H -> H - alpha_i * x_i x_i^T.

    Args:
        V (np.ndarray): The original scaled feature matrix of shape (d, n), where d is the feature dimension.
        G (np.ndarray): The original Gram matrix of shape (n, n), where G[j, l] = x_j^T H^{-1} x_l.
        alpha_i (float): Scalar alpha_i defining the rank-1 update H_i = alpha_i * x_i x_i^T.
        i (int): Index of the vector x_i involved in the update.

    Returns:
        np.ndarray: The updated scaled feature matrix of shape (d, n).
    """
    n = G.shape[0]
    d = V.shape[0]
    assert G.shape == (n, n), "Gram matrix G must be square."
    assert V.shape[1] == n, "Matrix V must have n columns matching the Gram matrix."
    assert 0 <= i < n, "Index i must be valid for the Gram matrix."

    # Extract the i-th column of G and the corresponding feature in V
    G_i = G[:, i]  # Column vector (n,)
    G_ii = G[i, i]  # Diagonal element
    V_i = V[:, i]  # Column vector (d,)

    # Compute the denominator: 1 - alpha_i * G[i, i]
    denominator = 1 - alpha_i * G_ii
    if abs(denominator) < 1e-10:
        raise ValueError("Update results in numerical instability (denominator close to zero).")

    # Compute the correction term for each column of V
    correction = (alpha_i / denominator) * np.outer(V_i, G_i)  # Shape: (d, n)

    # Update V
    V_updated = V + correction

    return V_updated


def straightforward_newton_step(
        model: LogisticRegression,
        indices: np.ndarray
) -> LogisticRegressionModel:
    """
    Perform a straightforward Newton step on the logistic regression model,
    removing a set of samples.

    Parameters:
        model (LogisticRegression): The logistic regression object containing data, weights, and Hessian information.
        indices (np.ndarray): Array of integers specifying the indices of samples to remove.

    Returns:
        LogisticRegressionModel: The updated LogisticRegressionModel with modified weights and intercept.
    """
    # Step 1: Compute the Hessian of the full model
    H = model.compute_hessian() * model.n

    # Step 2: Create a smaller model with only the samples to be removed
    features_to_remove = model.features[indices]
    labels_to_remove = model.labels[indices]

    smaller_model = LogisticRegression(
        features=features_to_remove,
        labels=labels_to_remove,
        regularization=model.model.regularization,
        reg_type=model.model.reg_type,
        fit_intercept=model.fit_intercept
    )
    smaller_model.model.extended_weights = model.model.extended_weights

    # Compute DeltaH = Hessian of the smaller model
    DeltaH = smaller_model.compute_hessian() * smaller_model.n

    # Step 3: Compute the sum of gradients for the removed samples
    gradients = smaller_model.compute_gradients()
    gradient_sum = np.sum(gradients, axis=0).reshape(-1, 1)  # Make it a column vector

    # Step 4: Compute the Newton step
    H_minus_DeltaH = H - DeltaH
    H_minus_DeltaH_inv = np.linalg.inv(H_minus_DeltaH)  # Invert the matrix
    newton_step = (H_minus_DeltaH_inv @ gradient_sum).ravel()

    # Step 5: Update the weights of the original model
    extended_weights = model.model.extended_weights  # Get extended weights (including intercept, if any)
    updated_extended_weights = extended_weights + newton_step

    # Create a new LogisticRegressionModel with updated weights and intercept
    updated_model = LogisticRegressionModel(
        weights=updated_extended_weights,
        regularization=model.model.regularization,
        reg_type=model.model.reg_type
    )

    return updated_model


def approximate_newton_step(
    model: LogisticRegression,
    indices: np.ndarray,
    gram_matrix: Optional[np.ndarray] = None,
    scaled_feature_matrix: Optional[np.ndarray] = None
) -> LogisticRegressionModel:
    """
    Perform an approximate Newton step on the logistic regression model, avoiding matrix inversion.

    Parameters:
        model (LogisticRegression): The logistic regression object containing data, weights, and Hessian information.
        indices (np.ndarray): Array of integers specifying the indices of samples to remove.
        gram_matrix (Optional[np.ndarray]): Precomputed Gram matrix. Must be either n x n or k x k,
                                            where n is the total number of samples and k is the number of samples to remove.
        scaled_feature_matrix (Optional[np.ndarray]): Precomputed scaled feature matrix V = H^{-1} X.
                                                      Must be either n x d or k x d,
                                                      where n is the total number of samples and d is the number of features.

    Returns:
        LogisticRegressionModel: The updated LogisticRegressionModel with modified weights.

    Runtime:
        When both `gram_matrix` and `scaled_feature_matrix` are provided:
            - Total runtime is O(k^3 + d * k^2), where:
                * k = number of samples to remove.
                * d = number of features.
            - This includes iterative updates to the Gram matrix and the scaled feature matrix.

        When `gram_matrix` and `scaled_feature_matrix` are not provided:
            - Precomputation adds:
                * O(d^2 * n + d * n^2) to compute the Gram matrix (G) and scaled feature matrix (V).
            - Total runtime becomes O(d^2 * n + d * n^2 + k^3 + d * k^2), where:
                * n = total number of samples.
                * k = number of samples to remove.
                * d = number of features.

    Note:
        - This implementation is optimized for scenarios where k << n (number of removed samples is much smaller than the total number of samples).
        - Precomputing and providing `gram_matrix` and `scaled_feature_matrix` significantly reduces runtime for large datasets.
    """

    # Step 1: Check for constraints
    if model.model.reg_type != RegularizationType.L2 and model.model.regularization != 0:
        raise NotImplementedError("Only L2 regularization or zero regularization is supported.")
    if model.fit_intercept:
        raise NotImplementedError("fit_intercept=True is not implemented.")

    # Step 2: Compute H inverse
    H_inv = model.compute_hessian_inv() * model.n  # Apply unit consistency adjustment

    # Step 3: Validate and process the Gram matrix
    k = len(indices)
    if gram_matrix is not None:
        if gram_matrix.shape == (model.n, model.n):  # Full Gram matrix
            gram_matrix = gram_matrix[np.ix_(indices, indices)]  # Submatrix for removal set
        elif gram_matrix.shape != (k, k):  # Ensure dimensions match removal set
            raise ValueError(f"Gram matrix dimensions must be {model.n} x {model.n} or {k} x {k}.")
    else:
        # Compute the Gram matrix for the removal set
        features_to_remove = model.features[indices]
        gram_matrix = features_to_remove @ H_inv @ features_to_remove.T

    # Step 4: Validate and process the scaled feature matrix (V)
    if scaled_feature_matrix is not None:
        if scaled_feature_matrix.shape == (model.d, model.n):  # Full scaled feature matrix
            scaled_feature_matrix = scaled_feature_matrix[:, indices]  # Limit to removal set
        elif scaled_feature_matrix.shape != (model.d, k):  # Ensure dimensions match removal set
            raise ValueError(f"Scaled feature matrix dimensions must be {model.d} x {model.n} or {model.d} x {k}.")
    else:
        # Compute the scaled feature matrix for the removal set
        features_to_remove = model.features[indices]
        scaled_feature_matrix = H_inv @ features_to_remove.T  # Shape: d x k
        scaled_feature_matrix = scaled_feature_matrix

    # Step 5: Precompute values for the removal set
    beta = np.zeros(k)
    alpha = np.zeros(k)
    for idx, i in enumerate(indices):
        x_i = model.features[i]
        y_i = model.model.get_model_predictions(x_i.reshape(1, -1))[0]
        beta[idx] = y_i - model.labels[i]
        alpha[idx] = y_i * (1 - y_i)

    # Step 6: Iteratively update the Gram matrix and scaled feature matrix
    updated_gram_matrix = gram_matrix.copy()
    updated_scaled_feature_matrix = scaled_feature_matrix.copy()
    for idx, i in enumerate(indices):
        # Update scaled feature matrix
        updated_scaled_feature_matrix = update_scaled_feature_matrix(
            updated_scaled_feature_matrix,
            updated_gram_matrix,
            alpha[idx],
            idx
        )

        # Update Gram matrix
        updated_gram_matrix = update_gram_matrix(
            updated_gram_matrix,
            alpha[idx],
            idx
        )

    # Step 7: Compute the approximate Newton step
    newton_step = updated_scaled_feature_matrix @ beta  # Sum_i beta_i * updated v_i

    # Step 8: Update the weights
    weights = model.model.weights
    updated_weights = weights + newton_step

    # Create a new LogisticRegressionModel with updated weights and intercept
    updated_model = LogisticRegressionModel(
        weights=updated_weights,
        intercept=0.0,
        regularization=model.model.regularization,
        reg_type=model.model.reg_type
    )

    return updated_model


def woodbury_newton_step(
    original_regression: LogisticRegression,
    indices: np.ndarray,
    gram_matrix: Optional[np.ndarray] = None
) -> LogisticRegressionModel:
    """
    Perform an exact rank-k Woodbury update to the inverse Hessian in a logistic regression model
    to estimate the effect of removing a subset of samples, using an unnormalized Gram matrix.

    Parameters:
        original_regression (LogisticRegression): The logistic regression object with features, labels, weights, etc.
        indices (np.ndarray): Array of sample indices (shape: (k,)) to remove.
        gram_matrix (Optional[np.ndarray]): Optional unnormalized Gram matrix (shape: (n, n) or (k, k)),
                                            defined as G = X H^{-1} X^T (not yet scaled by beta).

    Returns:
        LogisticRegressionModel: A new LogisticRegressionModel with updated weights after approximate Newton step.

    Notes:
        - Uses Woodbury identity to efficiently compute the change in weights due to removing k samples.
        - Assumes L2 or no regularization.
        - Requires `fit_intercept=False` (not implemented for bias terms).
        - Efficient when k << n.

    Runtime:
        With precomputed full Gram matrix: O(k^3 + d * k^2)
        Without Gram matrix: O(d^2 * n + d * k^2 + k^3)
    """

    if original_regression.model.reg_type != RegularizationType.L2 and original_regression.model.regularization != 0:
        raise NotImplementedError("Only L2 or zero regularization is supported.")
    if original_regression.fit_intercept:
        raise NotImplementedError("fit_intercept=True is not implemented.")

    n, d = original_regression.n, original_regression.d
    k = len(indices)

    # Step 1: Compute inverse Hessian
    H_inv = original_regression.compute_hessian_inv() / original_regression.n  # Normalize appropriately

    # Step 2: Extract features and predictions for selected samples
    X_T = original_regression.features[indices]         # shape (k, d)
    y_preds = original_regression.model.get_model_predictions(X_T)  # shape (k,)
    beta = np.sqrt(y_preds * (1 - y_preds))           # shape (k,)

    # Step 3: Construct U = X_T^T * diag(beta)
    U = (X_T.T * beta)   # shape: (d, k)

    # Step 4: Prepare Gram submatrix and scale it by beta
    if gram_matrix is not None:
        if gram_matrix.shape == (n, n):
            G_sub = gram_matrix[np.ix_(indices, indices)]  # shape: (k, k)
        elif gram_matrix.shape == (k, k):
            G_sub = gram_matrix
        else:
            raise ValueError(f"Gram matrix must be of shape ({n}, {n}) or ({k}, {k})")
    else:
        # Compute Gram matrix from scratch
        G_sub = X_T @ H_inv @ X_T.T  # (k, k)

    # Now scale: G' = D G_sub D, where D = diag(beta)
    G_scaled = beta[:, None] * G_sub * beta[None, :]  # shape: (k, k)

    # Old debug code:
    # print(f"{(U.T @ H_inv @ U)[:3, :3]=}")
    # print(f"{G_scaled[:3, :3]=}")
    # assert np.allclose(U.T @ H_inv @ U, G_scaled)
    # H = (model.compute_hessian() * model.n)
    # assert np.allclose(H @ H_inv, np.eye(d))
    # delta_H = X_T.T @ np.diag(beta ** 2) @ X_T
    # assert delta_H.shape == (d, d)
    # H_delta_inv = np.linalg.pinv(H - delta_H)
    # H_delta_inv_woodbury = H_inv + (H_inv @ U @ np.linalg.inv(np.eye(k) - G_scaled) @ U.T @ H_inv)
    # print(f"{H_delta_inv[:3, :3]=}")
    # print(f"{H_delta_inv_woodbury[:3, :3]=}")
    # assert np.allclose(H_delta_inv_woodbury, H_delta_inv)

    # Step 5: Compute Woodbury update
    # weights = w + H_inv @ U @ (I - G_scaled)^(-1) @ U.T @ H_inv @ gradient
    # In Newton step, update = H_inv @ grad; grad = sum_i (y_i - t_i) x_i
    # So:
    residual = y_preds - original_regression.labels[indices]      # shape: (k,)
    update_direction = U @ np.linalg.solve(np.eye(k) - G_scaled, beta * (G_sub @ residual))  # shape: (d,)

    # Step 6: Update weights
    updated_weights = original_regression.model.weights + H_inv @ (update_direction + (X_T.T @ residual))

    # Step 7: Return updated model
    updated_model = LogisticRegressionModel(
        weights=updated_weights,
        regularization=original_regression.model.regularization,
        reg_type=original_regression.model.reg_type
    )

    return updated_model

