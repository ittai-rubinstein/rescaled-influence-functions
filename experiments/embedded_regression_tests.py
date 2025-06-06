import time

from datasets.common import SplitDataset
from datasets.frozen_embeddings.loader  import EmbeddingDataset, load_embedding_dataset
import numpy as np

from src.bounds.dynamic_programming_algs import rti
from src.logistic_regression import LogisticRegression
from copy import deepcopy
from matplotlib import pyplot as plt

def compute_weighted_tensor_contraction_matrix(
        feature_matrix: np.ndarray,
        gram_matrix: np.ndarray,
        gamma: np.ndarray
) -> np.ndarray:
    """
    Computes M = X * ((gamma gamma^T) ⊙ (P ⊙ P)) * X^T
    where:
      - X is the feature matrix of shape (d, n)
      - P = X^T X is the Gram matrix of shape (n, n)
      - gamma is a weight vector of shape (n,)

    Parameters:
    - feature_matrix (np.ndarray): The feature matrix X of shape (d, n)
    - gram_matrix (np.ndarray): The Gram matrix P of shape (n, n)
    - gamma (np.ndarray): The weight vector gamma of shape (n,)

    Returns:
    - np.ndarray: The resulting matrix M of shape (d, d)
    """
    hadamard_squared = gram_matrix * gram_matrix # P ⊙ P
    weighted_component = gamma[:, np.newaxis] * hadamard_squared * gamma[np.newaxis, :]
    M = feature_matrix @ weighted_component @ feature_matrix.T
    return M


if __name__ == "__main__":

    # Step 1: load the dataset
    dataset_name = EmbeddingDataset.SST2

    print(f"Loading Dataset {dataset_name}")

    experiment = load_embedding_dataset(dataset_name, max_samples=5000)

    print("Train set dimensions:")
    X = experiment.train.features
    Y = experiment.train.labels
    print(X.shape, Y.shape)
    n, d = X.shape

    # Step 2: train a logistic regression model on the entire dataset
    t0 = time.time()
    model = LogisticRegression(experiment.train.features, experiment.train.labels, regularization=1E-3,
                               fit_intercept=False, reg_type="L2")
    model.fit(verbose=True)

    test_accuracy = model.model.get_accuracy(experiment.test.features, experiment.test.labels)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print(f"This took {time.time() - t0:.2f} seconds")

    # Step 3: compute raw matrices

    from src.bounds.utils import compute_symmetric_matrix_sqrt
    import time
    import numpy as np

    # Measure and print elapsed time for each step
    start_time = time.time()
    print("Computing trainset gradients...", end=" ")
    trainset_gradients = model.compute_gradients(
        experiment.train.features, experiment.train.labels
    )
    print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    print("Computing the Hessian...", end=" ")
    H = model.n * model.compute_hessian()
    print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    print("Inverting the Hessian...", end=" ")
    H_inv = model.compute_hessian_inv() / model.n
    print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    print("Normalizing the Features...", end=" ")
    H_inv_sqrt = compute_symmetric_matrix_sqrt(H_inv)
    X_norm = X @ H_inv_sqrt
    print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    print("Computing the model influences...", end=" ")
    model_influences = trainset_gradients @ H_inv
    print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    print("Computing the test set gradients...", end=" ")
    # test_loss_gradient = np.sum(model.compute_gradients(
    #     experiment.test.features[fixed_test, :], experiment.test.labels[fixed_test]
    # ), axis=0)
    print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    print("Computing test loss infs...", end=" ")
    # sample_influence_on_test_loss = model_influences @ test_loss_gradient
    print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    print("Computing self-loss infs...", end=" ")
    self_loss_infs = np.einsum('ij,ij->i', model_influences, trainset_gradients)
    print(f"Done. Took {time.time() - start_time:.2f} seconds.")

    # Step 4: compute softmax gradients at initial model
    y_hat = model.model_predictions
    alpha = experiment.train.labels - y_hat
    beta = np.sqrt(y_hat * (1 - y_hat))
    gamma = (beta ** 2) * (1 - (2 * y_hat))

    # Step 5: compute the Gram matrix
    print("Computing Gram matrices...")

    Gu = experiment.train.features @ H_inv @ experiment.train.features.T
    assert np.allclose(Gu, X_norm @ X_norm.T)

    # G = Gu * (beta[:, np.newaxis]) * (beta[np.newaxis, :])

    # Step 6: compute 1st order update matrix
    print("Computing M...")
    R = 1 / (1 - (Gu.diagonal() * (beta ** 2)))
    M = compute_weighted_tensor_contraction_matrix(X_norm.T * R[np.newaxis, :], Gu - np.diag(Gu.diagonal()), gamma)
    k_prime_thresh = 1 / np.max(np.linalg.eigvalsh(M))
    print("Computing spectrum of M...")
    print(f"k prime thresh = {k_prime_thresh}")

    G = beta[:, np.newaxis] * Gu * beta[np.newaxis, :]
    rti_alpha_G_alpha = rti((alpha / beta)[np.newaxis, :] * (alpha / beta)[:, np.newaxis] * G)
    print("RTI alpha G alpha")
    print('\n'.join(f'{i+1}: {rti_alpha_G_alpha[i]}' for i in range(max(20, int(2*k_prime_thresh)))))

