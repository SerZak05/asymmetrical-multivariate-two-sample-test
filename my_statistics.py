import numpy as np
from sklearn import metrics


def compute_l(mask: np.ndarray, A: np.ndarray) -> float:
    """Computes L_n statistic (see def. before Prop.2.1).
    Args:
        mask (np.ndarray): Bool mask for points in the first class.
        A (np.ndarray): Adjacency matrix of 1NN graph.
    
    Returns:
        float: Computed statistic.
    """
    return np.sum(A * (mask.reshape(-1, 1) * mask.reshape(1, -1)))

def compute_symm_l(mask: np.ndarray, A: np.ndarray) -> float:
    """Computes L_nk statistic with k = 1.
    Args:
        mask (np.ndarray): Bool mask for points in the first class.
        A (np.ndarray): Adjacency matrix of 1NN graph.
    
    Returns:
        float: Computed statistic.
    """
    return np.sum(A[mask.reshape(-1, 1) * mask.reshape(1, -1)]) + np.sum(A[(~mask).reshape(-1, 1) * (~mask).reshape(1, -1)])

def mmd_lin(X: np.ndarray, Y: np.ndarray) -> float:
    """Computes MMD^2_u with linear kernel.
    Args:
        X (np.ndarray): First class points.
        Y (np.ndarray): Second class points.
    
    Returns:
        float: Computed statistic.
    """
    delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
    return delta.dot(delta.T)

def mmd_gaussian(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Computes MMD^2_u with gaussian kernel.
    Args:
        X (np.ndarray): First class points.
        Y (np.ndarray): Second class points.
        gamma (float): Parameter of the kernel.
    
    Returns:
        float: Computed statistic.
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
