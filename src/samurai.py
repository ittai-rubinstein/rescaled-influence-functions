import numpy as np


def get_V_matrix(M: np.ndarray) -> np.ndarray:
    """
    Given the gram matrix, computes the V matrix = maximal abs of sum of k elements from ith row.
    Args:
        M: a matrix representing the Gram matrix times residuals.

    Returns: V the V matrix
    """
    M = np.copy(M)

    np.sort(M, axis=1)
    V1 = np.abs(np.cumsum(M, axis=1))
    V2 = np.abs(np.cumsum(M[:, ::-1], axis=1))

    return np.maximum(V1, V2)


def get_W_matrix(V: np.ndarray) -> np.ndarray:
    """
    Returns the W matrix = sum over k' largest elements of the V matrix
    Args:
        V: The V matrix defined above

    Returns: The W matrix defined above.
    """

    V = np.copy(V)
    np.sort(V, axis=0)
    return np.cumsum(V[::-1, :])


def get_G_hat(G: np.ndarray) -> np.ndarray:
    """
    Removes the diagonal elements from a given Gram matrix
    Args:
        G: The gram matrix.

    Returns: G hat = G - diagonal elements.
    """
    return G - np.diag(G.diagonal())


def bound_I_GD(G: np.ndarray) -> np.ndarray:
    """
    Given G, returns a mapping from k to sigmax((I - GD)^{-1}) over D of norm k.
    Args:
        G: gram matrix

    Returns: bound on sigmax((I - GD)^{-1}) over D of norm k.
    """

    G_hat = get_G_hat(G)
    R = 1 / (1 - G.diagonal())
    RG2R = R[:, np.newaxis] * (G_hat @ G_hat) * R[np.newaxis, :]
    RG2R_hat = get_G_hat(RG2R)**2
    np.sort(RG2R_hat, axis=1)
    M = np.cumsum(RG2R_hat, axis=1)
    np.sort(M, axis=0)
    diag_max = np.max(RG2R_hat.diagonal())
    return np.max(R) * (1 / (1 - diag_max - np.cumsum(M, axis=0).diagonal()))

