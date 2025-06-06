import unittest
import numpy as np
from src.logistic_regression import LogisticRegression, LogisticRegressionModel


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Data generation as before
        np.random.seed(0)
        self.n = 100  # Number of samples
        self.d = 5  # Number of dimensions

        # Generate structured dataset
        x_0 = np.random.choice([-1, 1], size=(self.n, 1))
        other_features = np.random.randn(self.n, self.d - 2)
        last_feature = np.zeros((self.n, 1))
        self.features = np.hstack([x_0, other_features, last_feature])
        self.labels = (x_0.squeeze() > 0).astype(int)

        self.regularization = 0.1
        self.model = LogisticRegression(self.features, self.labels, self.regularization)

    def test_hessian_psd_at_optimal(self):
        """Test that the Hessian at the optimal point is positive semi-definite (PSD)."""
        self.model.fit()
        hessian = self.model.compute_hessian()
        eigenvalues = np.linalg.eigvalsh(hessian)
        self.assertTrue(np.all(eigenvalues >= 0), "Hessian is not positive semi-definite at optimal point.")

    def test_taylor_expansion_approximation_optimal(self):
        """Test that a small perturbation causes loss to change by approximately Delta^T H Delta at optimal point."""
        self.model.fit()
        hessian = self.model.compute_hessian()
        delta = np.random.randn(self.d) * 0.01
        original_loss = self.model.model.get_model_losses(self.features, self.labels, self.regularization).mean()

        # Apply the perturbation
        perturbed_model = self.model.model.weights + delta
        self.model.model.weights = perturbed_model
        perturbed_loss = self.model.model.get_model_losses(self.features, self.labels, self.regularization).mean()

        # Compute the expected change in loss from Taylor expansion
        predicted_loss_change = 0.5 * delta.T @ hessian @ delta
        actual_loss_change = perturbed_loss - original_loss

        self.assertAlmostEqual(actual_loss_change, predicted_loss_change, places=3)

    def test_gradient_first_order_approximation(self):
        """Test that a small perturbation from a random initialization changes loss by approximately gradient^T Delta."""
        # Random model initialization
        random_weights = np.random.randn(self.d)
        self.model.model.weights = random_weights

        delta = np.random.randn(self.d) * 0.01
        original_loss = self.model.model.get_model_losses(self.features, self.labels, self.regularization).mean()

        # Apply the perturbation
        perturbed_model = random_weights + delta
        self.model.model.weights = perturbed_model
        perturbed_loss = self.model.model.get_model_losses(self.features, self.labels, self.regularization).mean()

        # Compute gradient at random initialization
        gradient = np.mean(self.model.compute_gradients(), axis=0)

        # First-order Taylor approximation of loss change
        predicted_loss_change = gradient @ delta
        actual_loss_change = perturbed_loss - original_loss

        self.assertAlmostEqual(actual_loss_change, predicted_loss_change, places=3)

    def test_hessian_inner_product_equivalence_for_d_greater_n(self):
        """Test that the hessian inner product method matches direct computation when d > n."""
        # Increase d beyond n to test with d > n
        self.d = 200  # Set d greater than n
        self.features = np.random.randn(self.n, self.d)
        self.labels = np.random.randint(0, 2, size=self.n)
        self.model = LogisticRegression(self.features, self.labels, self.regularization)
        self.model.fit()

        # Compute the Hessian inner product using the efficient method
        random_vector = np.random.randn(self.d)
        hessian_inner_prod = self.model.hessian_inner_prod(random_vector)

        # Compute the full Hessian and take the inner product directly
        full_hessian = self.model.compute_hessian()
        direct_inner_prod = full_hessian @ random_vector

        np.testing.assert_allclose(hessian_inner_prod, direct_inner_prod, atol=1e-3)

    def test_model_initialization(self):
        """Test that the model initializes correctly with default values."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model.weights.shape[0], self.features.shape[1])
        self.assertIsNone(self.model.model_predictions)


    def test_fit_function(self):
        """Test that the fit function optimizes the weights and makes correct predictions."""
        self.model.fit()
        predictions = self.model.model.get_model_predictions(self.features)
        predicted_labels = (predictions >= 0.5).astype(int)
        np.testing.assert_array_equal(predicted_labels, self.labels)

    def test_fit_with_regularization(self):
        """Test that the regularization affects the fit as expected."""
        model_no_reg = LogisticRegression(self.features, self.labels, regularization=0.0)
        model_with_reg = LogisticRegression(self.features, self.labels, regularization=0.1)
        model_no_reg.fit()
        model_with_reg.fit()
        weights_no_reg = model_no_reg.model.weights
        weights_with_reg = model_with_reg.model.weights
        self.assertTrue(np.linalg.norm(weights_with_reg) < np.linalg.norm(weights_no_reg))

    def test_gradient_close_to_zero_after_fit(self):
        """Test that the gradient is close to zero after optimization."""
        self.model.fit()
        gradients = np.mean(self.model.compute_gradients(), axis=0)
        gradient_norm = np.linalg.norm(gradients)
        self.assertAlmostEqual(gradient_norm, 0, places=3)

    def test_weight_on_last_dimension(self):
        """Test that the weight on the last dimension is close to zero due to regularization."""
        self.model.fit()
        self.assertAlmostEqual(self.model.model.weights[-1], 0, places=4)


# Run the tests
if __name__ == '__main__':
    unittest.main()
