import unittest
import numpy as np

from src.logistic_regression import LogisticRegression, RegularizationType
from src.newton_step import update_gram_matrix, update_scaled_feature_matrix, approximate_newton_step, \
    straightforward_newton_step, woodbury_newton_step


class TestNewtonStep(unittest.TestCase):
    def setUp(self):
        """
        Set up a simple LogisticRegression model for testing.
        """
        np.random.seed(42)
        n, d, k = 100, 25, 5  # Number of samples, features, and removals

        # Generate random features
        self.features = np.random.randn(n, d)

        # Ground truth weights vector (unit vector)
        ground_truth_weights = np.ones(d)

        # Generate probabilities from the logistic model
        logits = self.features @ ground_truth_weights
        probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid function

        # Draw labels from a Bernoulli distribution
        self.labels = np.random.binomial(1, probabilities)

        # Set regularization and create the LogisticRegression model
        self.regularization = 1E-2
        self.model = LogisticRegression(
            features=self.features,
            labels=self.labels,
            regularization=self.regularization,
            reg_type=RegularizationType.L2,
            fit_intercept=False
        )
        self.model.fit()

    def test_update_gram_matrix_synthetic(self):
        """
        Test the update_gram_matrix function with synthetic data by comparing
        the Sherman-Morrison-based update with a direct recomputation.
        """
        # Parameters
        n = 5  # Number of vectors
        d = 4  # Dimensionality of vectors
        alpha_i = 0.9
        i = 2  # Index of the vector x_i

        # Generate random vectors x_i and construct PSD matrix H
        np.random.seed(42)
        X = np.random.randn(n, d)  # n vectors of dimension d
        C = 1.0  # Regularization parameter
        H = X.T @ X + C * np.eye(d)  # PSD matrix

        # Compute the Gram matrix G = X @ H^{-1} @ X.T
        H_inv = np.linalg.inv(H)
        G = X @ H_inv @ X.T

        # Apply the Sherman-Morrison-based update
        G_updated = update_gram_matrix(G, alpha_i, i)

        # Naively recompute the updated Gram matrix
        H_i = alpha_i * np.outer(X[i], X[i])  # Rank-1 update
        H_new = H - H_i  # Updated H
        H_new_inv = np.linalg.inv(H_new)  # Direct inversion
        G_naive = X @ H_new_inv @ X.T  # Direct recomputation of Gram matrix

        # Check that the updated G matches the naive recomputation
        np.testing.assert_allclose(G_updated, G_naive, rtol=1e-6, atol=1e-8)

    def test_update_scaled_feature_matrix_synthetic(self):
        """
        Test the update_scaled_feature_matrix function with synthetic data by comparing
        the Sherman-Morrison-based update with a direct recomputation.
        """
        # Parameters
        n = 5  # Number of vectors (samples)
        d = 4  # Dimensionality of vectors (features)
        alpha_i = 0.9
        i = 2  # Index of the vector x_i

        # Generate random feature matrix X and construct PSD matrix H
        np.random.seed(42)
        X = np.random.randn(d, n)  # d features, n samples (transposed compared to earlier tests)
        C = 1.0  # Regularization parameter
        H = X @ X.T + C * np.eye(d)  # PSD matrix

        # Compute the scaled feature matrix V = H^{-1} @ X
        H_inv = np.linalg.inv(H)
        V = H_inv @ X

        # Compute the Gram matrix G = X.T @ H^{-1} @ X
        G = X.T @ V  # Equivalent to X.T @ H_inv @ X

        # Apply the Sherman-Morrison-based update to V
        V_updated = update_scaled_feature_matrix(V, G, alpha_i, i)

        # Naively recompute V after applying the rank-1 update to H
        H_i = alpha_i * np.outer(X[:, i], X[:, i])  # Rank-1 update
        H_new = H - H_i  # Updated H
        H_new_inv = np.linalg.inv(H_new)  # Direct inversion
        V_naive = H_new_inv @ X  # Direct recomputation of V

        # Check that the updated V matches the naive recomputation
        np.testing.assert_allclose(V_updated, V_naive, rtol=1e-6, atol=1e-8)

    def test_approximate_newton_step_invalid_inputs(self):
        """
        Test that the approximate_newton_step function raises exceptions for invalid inputs.
        """
        indices = np.array([0, 1, 2])  # Indices of samples to remove

        # Case 1: Invalid regularization type
        self.model.model.reg_type = "invalid_type"
        with self.assertRaises(NotImplementedError):
            approximate_newton_step(self.model, indices)

        # Case 2: Invalid fit_intercept setting
        self.model.model.reg_type = RegularizationType.L2
        self.model.fit_intercept = True
        with self.assertRaises(NotImplementedError):
            approximate_newton_step(self.model, indices)

        # Case 3: Invalid Gram matrix dimensions
        self.model.fit_intercept = False
        gram_matrix = np.random.randn(5, 3)  # Invalid shape
        with self.assertRaises(ValueError):
            approximate_newton_step(self.model, indices, gram_matrix=gram_matrix)

        # Case 4: Invalid scaled feature matrix dimensions
        scaled_feature_matrix = np.random.randn(3, 4)  # Invalid shape
        with self.assertRaises(ValueError):
            approximate_newton_step(self.model, indices, scaled_feature_matrix=scaled_feature_matrix)

    def test_approximate_newton_step_valid_inputs(self):
        """
        Test that the approximate_newton_step function does not raise exceptions for valid inputs.
        """
        indices = np.array([0, 1, 2])  # Indices of samples to remove

        # Valid Gram matrix and scaled feature matrix
        H_inv = self.model.compute_hessian_inv() / self.model.n
        gram_matrix = self.features @ H_inv @ self.features.T
        scaled_feature_matrix = H_inv @ self.features.T

        try:
            approximate_newton_step(
                self.model,
                indices,
                gram_matrix=gram_matrix,
                scaled_feature_matrix=scaled_feature_matrix
            )
        except Exception as e:
            self.fail(f"approximate_newton_step raised an unexpected exception: {e}")

    def test_approximate_newton_step_vs_straightforward(self):
        """
        Test that the approximate Newton step produces the same model weights as the
        straightforward Newton step when the regularization parameter is set to 0.
        """
        # Step 1: Create a copy of the model with regularization set to 0
        reg_free_model = LogisticRegression(
            features=self.features,
            labels=self.labels,
            regularization=0.0,
            reg_type=self.model.model.reg_type,
            fit_intercept=self.model.fit_intercept
        )
        reg_free_model.model.weights = self.model.model.weights.copy()  # Copy weights from the original model
        reg_free_model.fit(warm_start=True)  # Fit the model with warm_start=True

        # Step 2: Initialize Gram matrix (G) and scaled feature matrix (V) on the regularization-free model
        H_inv = reg_free_model.compute_hessian_inv() / reg_free_model.n  # Scale H_inv for unit consistency
        gram_matrix = self.features @ H_inv @ self.features.T
        scaled_feature_matrix = H_inv @ self.features.T

        # Step 3: Select random removal indices
        np.random.seed(42)
        k = 5  # Number of samples to remove
        removal_indices = np.random.choice(self.features.shape[0], size=k, replace=False)

        # Step 4: Perform approximate and straightforward Newton steps
        approx_model = approximate_newton_step(
            model=reg_free_model,
            indices=removal_indices,
            gram_matrix=gram_matrix,
            scaled_feature_matrix=scaled_feature_matrix
        )

        straightforward_model = straightforward_newton_step(
            model=reg_free_model,
            indices=removal_indices
        )

        # Step 5: Verify that the weights produced by both methods are identical
        diff_approx = reg_free_model.model.weights - approx_model.weights
        diff_exact = reg_free_model.model.weights - straightforward_model.weights
        np.testing.assert_allclose(
            diff_approx,
            diff_exact,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Approximate Newton step weights differ from straightforward Newton step weights"
        )

    def test_woodbury_newton_step_vs_straightforward(self):
        """
        Test that the woodbury-based Newton step produces the same model weights as the
        straightforward Newton step when regularization is set to 0.
        """
        # Step 1: Create a copy of the model with regularization set to 0
        reg_free_model = LogisticRegression(
            features=self.features,
            labels=self.labels,
            regularization=0.0,
            reg_type=self.model.model.reg_type,
            fit_intercept=self.model.fit_intercept
        )
        reg_free_model.model.weights = self.model.model.weights.copy()
        reg_free_model.fit(warm_start=True)

        # Step 2: Precompute H_inv and Gram matrix
        H_inv = reg_free_model.compute_hessian_inv() / reg_free_model.n  # match unit convention
        gram_matrix = self.features @ H_inv @ self.features.T  # full unnormalized Gram matrix

        # Step 3: Select random subset of samples to remove
        np.random.seed(42)
        k = 5
        removal_indices = np.random.choice(self.features.shape[0], size=k, replace=False)

        # Step 4: Apply woodbury and straightforward updates
        woodbury_model = woodbury_newton_step(
            original_regression=reg_free_model,
            indices=removal_indices,
            gram_matrix=gram_matrix
        )
        straightforward_model = straightforward_newton_step(
            model=reg_free_model,
            indices=removal_indices
        )

        # Step 5: Compare weight differences
        diff_woodbury = reg_free_model.model.weights - woodbury_model.weights
        diff_exact = reg_free_model.model.weights - straightforward_model.weights

        np.testing.assert_allclose(
            diff_woodbury,
            diff_exact,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Woodbury-based Newton step does not match straightforward Newton step"
        )


if __name__ == "__main__":
    unittest.main()
