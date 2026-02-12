import numpy as np
from sklearn import metrics


def compute_l(mask: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.sum(A * (mask.reshape(-1, 1) * mask.reshape(1, -1)))

def compute_symm_l(mask: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.sum(A[mask.reshape(-1, 1) * mask.reshape(1, -1)]) + np.sum(A[(~mask).reshape(-1, 1) * (~mask).reshape(1, -1)])

def mmd_lin(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    return delta.dot(delta.T)

def mmd_gaussian(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
