from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize
import time

LOGIT_MIN, LOGIT_MAX = -100, 100

class RegularizationType(Enum):
    L1 = "L1"
    L2 = "L2"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)


@dataclass
class LogisticRegressionModel:
    weights: np.ndarray
    regularization: float = 0.0
    reg_type: Union[RegularizationType, str] = RegularizationType.L2

    @property
    def extended_weights(self) -> np.ndarray:
        """
        Get the extended weights, which include the intercept if fit_intercept is True.
        """
        return self.weights

    @extended_weights.setter
    def extended_weights(self, values: np.ndarray):
        """
        Set the extended weights, updating the intercept and weights appropriately.
        """
        self.weights = values

    def get_model_predictions(self, features: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid (logistic) predictions for given features using the model.

        Parameters:
            features (np.ndarray): An n x d array where each row is a feature vector for one sample.

        Returns:
            np.ndarray: An n x 1 array of predicted probabilities (between 0 and 1) for each sample.
        """
        logits = np.clip(features @ self.weights, LOGIT_MIN, LOGIT_MAX)
        return 1 / (1 + np.exp(-logits))

    def get_model_losses(self, features: np.ndarray, labels: np.ndarray, regularization: Optional[float] = None,
                         reg_type: Optional[RegularizationType] = None) -> np.ndarray:
        """
        Compute the binary cross-entropy loss for each sample, with optional L1 or L2 regularization.

        Parameters:
            features (np.ndarray): An n x d array of features.
            labels (np.ndarray): An n x 1 array of binary labels (0 or 1).
            regularization (float): Regularization strength.
            reg_type (RegularizationType): Type of regularization (L1 or L2).

        Returns:
            np.ndarray: An n x 1 array of individual losses for each sample.
        """
        if regularization is None:
            regularization = self.regularization
        if reg_type is None:
            reg_type = self.reg_type
        predictions = self.get_model_predictions(features)
        losses = -labels * np.log(predictions + 1e-15) - (1 - labels) * np.log(1 - predictions + 1e-15)

        if regularization > 0:
            if reg_type == RegularizationType.L2:
                reg_term = 0.5 * regularization * np.sum(self.weights ** 2)
            elif reg_type == RegularizationType.L1:
                reg_term = regularization * np.sum(np.abs(self.weights))
            losses += reg_term

        return losses

    def get_sample_gradients(self, features: np.ndarray, labels: np.ndarray, regularization: Optional[float] = None,
                             reg_type: Optional[RegularizationType] = None) -> np.ndarray:
        """
        Compute the gradient of the loss function with respect to weights  for each sample,
        including regularization.

        Parameters:
            features (np.ndarray): An n x d array of features.
            labels (np.ndarray): An n x 1 array of binary labels.
            regularization (float): Regularization strength.
            reg_type (RegularizationType): Type of regularization (L1 or L2).

        Returns:
            np.ndarray: An n x (d) array where each row is the gradient of the loss for a sample.
        """
        if regularization is None:
            regularization = self.regularization
        if reg_type is None:
            reg_type = self.reg_type
        predictions = self.get_model_predictions(features)
        error = predictions - labels
        feature_gradients = features * error[:, None]  # Each sample's gradient for each feature

        if regularization > 0:
            if reg_type == RegularizationType.L2:
                regularization_grad = regularization * self.weights
            elif reg_type == RegularizationType.L1:
                regularization_grad = regularization * np.sign(self.weights)
            feature_gradients += regularization_grad[None, :]  # Apply regularization to feature gradients

        return feature_gradients

    def get_loss(self, features: np.ndarray, labels: np.ndarray, regularization: Optional[float] = None,
                 reg_type: Optional[RegularizationType] = None) -> float:
        """
        Compute the mean binary cross-entropy loss with regularization.

        Parameters:
            features (np.ndarray): An n x d array of features.
            labels (np.ndarray): An n x 1 array of binary labels (0 or 1).
            regularization (float): Regularization strength.
            reg_type (RegularizationType): Type of regularization (L1 or L2).

        Returns:
            float: The total mean loss for the dataset.
        """
        if regularization is None:
            regularization = self.regularization
        if reg_type is None:
            reg_type = self.reg_type
        individual_losses = self.get_model_losses(features, labels, regularization, reg_type)
        return np.mean(individual_losses)

    def get_gradient(self, features: np.ndarray, labels: np.ndarray, regularization: Optional[float] = None,
                     reg_type: Optional[RegularizationType] = None) -> np.ndarray:
        """
        Compute the mean gradient of the loss function with respect to weights, with regularization.

        Parameters:
            features (np.ndarray): An n x d array of features.
            labels (np.ndarray): An n x 1 array of binary labels.
            regularization (float): Regularization strength.
            reg_type (RegularizationType): Type of regularization (L1 or L2).

        Returns:
            np.ndarray: Mean gradient of the loss with respect to model parameters.
        """
        if regularization is None:
            regularization = self.regularization
        if reg_type is None:
            reg_type = self.reg_type
        individual_gradients = self.get_sample_gradients(features, labels, regularization, reg_type)
        return np.mean(individual_gradients, axis=0)  # Average over all samples

    def get_accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the accuracy of the model on the dataset.

        Returns:
            float: The accuracy as a percentage of correct predictions.
        """
        predictions = self.get_model_predictions(features) >= 0.5
        correct = np.sum(predictions == labels)
        return (correct / len(labels)) * 100


class LogisticRegression:
    self_losses: Optional[np.ndarray] = None
    features: np.ndarray
    model: LogisticRegressionModel
    labels: np.ndarray

    def __init__(self, features: np.ndarray, labels: np.ndarray, regularization: float = 0.0,
                 reg_type: Union[RegularizationType, str] = RegularizationType.L2, fit_intercept: bool = False):
        """
        Initialize the logistic regression model.

        Parameters:
            features (np.ndarray): An n x d array of training samples in R^d.
            labels (np.ndarray): An n x 1 array of binary labels for each sample.
            regularization (float): Regularization strength.
            reg_type (RegularizationType): Type of regularization (L1 or L2).
            fit_intercept (bool): Whether to fit an intercept.
        """
        self.labels = labels
        self.fit_intercept = fit_intercept
        if fit_intercept:
            self.features = np.hstack((np.ones_like(features[:, :1]), features))
        else:
            self.features = features
        self.n, self.d = self.features.shape
        self.model = LogisticRegressionModel(weights=np.zeros(self.d), regularization=regularization,
                                             reg_type=reg_type)
        self.model_predictions = None
        # self.loss_gradients = None
        self.hessian = None
        self.hessian_inverse = None

    def fit(self, warm_start: bool = False, verbose: bool = False, max_evals: Optional[int] = None):
        """
        Fit the logistic regression model using the L-BFGS-B optimization method.

        Parameters:
            warm_start (bool): If True, start from the current model weights; otherwise, start from zero.
            verbose (bool): If True, adds a debug print to help track progress of the model train
            max_evals (int): If set, this caps the number of function evals in training.
                By default 500, or 50 when warm_start is set.
        """
        initial_weights = self.model.weights if warm_start \
            else np.zeros(self.d)
        reg_type = self.model.reg_type
        regularization = self.model.regularization

        state = {"loss": None, "gradient": None, "iteration": 0, "start_time": time.time()}

        def loss_fn(weights: np.ndarray) -> float:
            logits = np.clip(self.features @ weights, LOGIT_MIN, LOGIT_MAX)
            predictions = 1 / (1 + np.exp(-logits))
            losses = -self.labels * np.log(predictions + 1e-15) - (1 - self.labels) * np.log(1 - predictions + 1e-15)
            loss = np.mean(losses)

            if reg_type == RegularizationType.L2:
                reg_term = 0.5 * regularization * (np.linalg.norm(weights, 2) ** 2)
            elif reg_type == RegularizationType.L1:
                reg_term = regularization * np.linalg.norm(weights, 1)

            state["loss"] = loss + reg_term
            return state["loss"]

        def gradient_fn(weights: np.ndarray) -> np.ndarray:
            logits = np.clip(self.features @ weights, LOGIT_MIN, LOGIT_MAX)
            predictions = 1 / (1 + np.exp(-logits))
            error = predictions - self.labels
            feature_grad = self.features.T @ error / self.n  # Each sample's gradient for each feature

            if reg_type == RegularizationType.L2:
                regularization_grad = regularization * weights
            elif reg_type == RegularizationType.L1:
                regularization_grad = regularization * np.sign(weights)

            state["gradient"] = feature_grad + regularization_grad
            return state["gradient"]

        def callback(weights):
            state["iteration"] += 1
            elapsed_time = time.time() - state["start_time"]

            if verbose:
                print(f"Iteration {state['iteration']}")
                print(f"Elapsed Time: {elapsed_time:.4f} seconds")
                print(f"Current Loss: {state['loss']:.6f}")
                print(f"L2 Norm of Gradient: {np.linalg.norm(state['gradient']):.6f}")
                print("-" * 30)

        # Set aggressive stopping conditions to reduce dependence of final model on initialization
        if max_evals is None:
            max_evals = 50 if warm_start else 500
        options = {
            'maxiter': max_evals,
            'maxfun': max_evals,
            'gtol': 1e-5,  # Stop only when gradient norm is really small
            'ftol': 1e-6,  # Require function value to change very little
            'disp': verbose
        }

        result = minimize(
            fun=loss_fn,
            x0=initial_weights,
            jac=gradient_fn,
            method="L-BFGS-B",
            callback=callback,
            options=options
        )
        self.model.weights = result.x
        # Update model predictions
        self.model_predictions = self.model.get_model_predictions(self.features)

    def compute_gradients(self, features: Optional[np.ndarray] = None,
                          labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the gradients of the loss for each sample.

        Returns:
            np.ndarray: An n x (d + 1) array where each row is the gradient of the loss for a sample.
                        The first column is the gradient w.r.t. the intercept (if fit_intercept is True),
                        and the remaining columns correspond to the feature gradients.
        """
        if features is None:
            features = self.features
        if labels is None:
            labels = self.labels
        individual_gradients = self.model.get_sample_gradients(features, labels)
        return individual_gradients

    def compute_hessian(self) -> np.ndarray:
        """
        Compute the Hessian matrix of the loss function at the current model weights.

        Returns:
            np.ndarray: The Hessian matrix of size (d+1) x (d+1) if fit_intercept is True, else d x d.
        """
        predictions = self.model.get_model_predictions(self.features)
        diag = predictions * (1 - predictions)
        weighted_features = self.features.T * diag
        hessian = (weighted_features @ self.features) / self.n

        if self.model.reg_type == RegularizationType.L2 and self.model.regularization > 0:
            hessian += self.model.regularization * np.eye(self.d)

        self.hessian = hessian
        return hessian

    def hessian_inner_prod_d_large(self, v: np.ndarray) -> np.ndarray:
        """
        Compute the inner product of the Hessian with a given vector v without storing the full Hessian matrix.

        Parameters:
            v (np.ndarray): A vector of size d.

        Returns:
            np.ndarray: Result of Hessian * v.
        """
        predictions = self.model.get_model_predictions(self.features)
        diag = predictions * (1 - predictions)
        weighted_features = self.features.T * diag

        hessian_v_product = weighted_features @ (self.features @ v) / self.n

        if self.model.regularization > 0 and self.model.reg_type == RegularizationType.L2:
            hessian_v_product += self.model.regularization * v
        return hessian_v_product

    def hessian_inner_prod(self, v: np.ndarray) -> np.ndarray:
        """
        Compute the inner product of the Hessian with a given vector v based on dimensionality.

        Parameters:
            v (np.ndarray): A vector of size d+1 if fit_intercept is True, else size d.

        Returns:
            np.ndarray: Result of Hessian * v.
        """
        if self.n > self.d and self.hessian is None:
            self.compute_hessian()
        return self.hessian @ v if self.hessian is not None else self.hessian_inner_prod_d_large(v)

    def compute_hessian_inv(self) -> np.ndarray:
        if self.hessian_inverse is None:
            self.hessian_inverse = np.linalg.pinv(self.compute_hessian())
        return self.hessian_inverse

    def hessian_inv_inner_prod(self, v: np.ndarray) -> np.ndarray:
        """
        Compute the product of the inverse Hessian with a given vector v.

        Parameters:
            v (np.ndarray): A vector of size d+1 if fit_intercept is True, else size d.

        Returns:
            np.ndarray: Result of inverse(Hessian) * v.
        """
        if self.n > self.d:
            if self.hessian is None:
                self.compute_hessian()
            if self.hessian_inverse is None:
                self.hessian_inverse = np.linalg.pinv(self.hessian)
            return self.hessian_inverse @ v
        else:
            if self.hessian is None:
                self.compute_hessian()

            EPSILON = 1E-6
            pseudo_inv_correction = np.eye(self.d) * EPSILON
            return np.linalg.solve(self.hessian + pseudo_inv_correction, v)

    def get_self_losses(self):
        if self.self_losses is None:
            self.self_losses = self.model.get_model_losses(self.features, self.labels)
        return self.self_losses
