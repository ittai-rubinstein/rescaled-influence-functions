import unittest
import numpy as np
from src.bounds.component_bounds import BetaFunction, GammaFunction, WtDeltaFunction, WtDeltaDerivativeFunction


class TestSingleVariableFunctions(unittest.TestCase):
    test_ranges = [
        # Small ranges near zero
        (0, 1e-6),
        (1e-6, 1e-3),
        (-1e-3, 0),

        # Larger ranges
        (0, 1),
        (1, 10),
        (-10, 0),

        # Very close pairs
        (1, 1.0001),
        (-1, -0.9999),

        # Mixed ranges
        (-1, 1),
    ]

    def evaluate_and_test(self, function_instance, a, b, epsilon=1e-3, num_points=1000, debug_message: str = ""):
        """
        Helper function to test range_mapping against a numerical grid-based approximation.
        """
        # Create a dense grid of points in the range [a, b]
        x_values = np.linspace(a, b, num_points)

        # Evaluate the function on the grid
        y_values = function_instance.f(x_values)

        # Compute actual observed min and max
        observed_min = np.min(y_values)
        observed_max = np.max(y_values)

        # Use the function's range_mapping method to compute the predicted min and max
        predicted_min, predicted_max = function_instance.range_mapping(a, b)

        # Assert the predictions with one-sided error tolerances
        self.assertGreaterEqual(
            predicted_min, observed_min - epsilon,
            f"Lower bound {predicted_min} is loose {observed_min} - {epsilon}. {debug_message=}."
        )
        self.assertGreaterEqual(
            observed_min, predicted_min,
            f"Lower bound {predicted_min} is wrong {observed_min}. {debug_message=}."
        )

        self.assertLessEqual(
            predicted_max, observed_max + epsilon,
            f"Upper bound {predicted_max} is loose {observed_max} + {epsilon}. {debug_message=}."
        )
        self.assertLessEqual(
            observed_max, predicted_max,
            f"Upper bound {predicted_max} is wrong {observed_max}. {debug_message=}."
        )

    def test_beta_function(self):
        # Create an instance of BetaFunction
        beta_function = BetaFunction()

        for a, b in self.test_ranges:
            with self.subTest(f"Testing BetaFunction on range [{a}, {b}]"):
                self.evaluate_and_test(beta_function, a, b)

    def test_gamma_function(self):
        # Create an instance of GammaFunction
        gamma_function = GammaFunction()

        for a, b in self.test_ranges:
            with self.subTest(f"Testing GammaFunction on range [{a}, {b}]"):
                self.evaluate_and_test(gamma_function, a, b)

    def test_wt_delta_function(self):
        parameter_pairs = [
            (0.25, 0.9999),  # Near upper limits
            (0.001, 1 / 0.001 - 1e-6),  # Near lower limits
            (0.1, 0.9),  # Random valid pair
            (0.05, 0.95),  # Random valid pair
        ]

        for beta_0, g_ii in parameter_pairs:
            wt_delta_function = WtDeltaFunction(beta_0, g_ii)
            for a, b in self.test_ranges:
                with self.subTest(f"Testing WtDeltaFunction with beta_0={beta_0}, g_ii={g_ii} on range [{a}, {b}]"):
                    self.evaluate_and_test(wt_delta_function, a, b)

    def test_wt_delta_derivative_function(self):
        parameter_pairs = [
            (0.25, 0.9999),  # Near upper limits
            (0.001, 1 / 0.001 - 1e-6),  # Near lower limits
            (0.1, 0.9),  # Random valid pair
            (0.05, 0.95),  # Random valid pair
        ]

        for beta_0, g_ii in parameter_pairs:
            wt_delta_derivative_function = WtDeltaDerivativeFunction(beta_0, g_ii)
            for a, b in self.test_ranges:
                with self.subTest(
                        f"Testing WtDeltaDerivativeFunction with beta_0={beta_0}, g_ii={g_ii} on range [{a}, {b}]"):
                    self.evaluate_and_test(wt_delta_derivative_function, a, b, debug_message=f"{beta_0=}, {g_ii=}, {a=}, {b=}")


if __name__ == "__main__":
    unittest.main()
