import numpy as np
from sklearn import metrics
import torch


def compute_l(mask: np.ndarray, A: np.ndarray) -> float:
    """Computes L_n statistic (see def. before Prop.2.1).
    Args:
        mask (np.ndarray): Bool mask for points in the first class.
        A (np.ndarray): Adjacency matrix of 1NN graph.
    
    Returns:
        float: Computed statistic.
    """
    return np.sum(A * (mask.reshape(-1, 1) * mask.reshape(1, -1)))

def compute_l_torch(mask: torch.Tensor, A: torch.Tensor) -> float:
    """Computes L_n statistic (see def. before Prop.2.1).
    Args:
        mask (torch.Tensor): Bool mask for points in the first class.
        A (torch.Tensor): Adjacency matrix of 1NN graph.
    
    Returns:
        float: Computed statistic.
    """
    return torch.sum(A * (mask.reshape(-1, 1) * mask.reshape(1, -1))).item()

def compute_symm_l(mask: np.ndarray, A: np.ndarray) -> float:
    """Computes L_nk statistic with k = 1.
    Args:
        mask (np.ndarray): Bool mask for points in the first class.
        A (np.ndarray): Adjacency matrix of 1NN graph.
    
    Returns:
        float: Computed statistic.
    """
    return np.sum(A[mask.reshape(-1, 1) * mask.reshape(1, -1)]) + np.sum(A[(~mask).reshape(-1, 1) * (~mask).reshape(1, -1)])

def compute_symm_l_torch(mask: torch.Tensor, A: torch.Tensor) -> float:
    """Computes L_nk statistic with k = 1.
    Args:
        mask (torch.Tensor): Bool mask for points in the first class.
        A (torch.Tensor): Adjacency matrix of 1NN graph.
    
    Returns:
        float: Computed statistic.
    """
    return torch.sum(A[mask.reshape(-1, 1) * mask.reshape(1, -1)]).item() + torch.sum(A[(~mask).reshape(-1, 1) * (~mask).reshape(1, -1)]).item()

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

def rbf_kernel(x, y, sigma_list):
    """
    Computes the RBF kernel matrix between x and y.
    
    Args:
        x (torch.Tensor): First set of samples (batch_size_x, features).
        y (torch.Tensor): Second set of samples (batch_size_y, features).
        sigma_list (list or torch.Tensor): List of bandwidths for a multi-scale kernel.

    Returns:
        torch.Tensor: Kernel matrix of shape (batch_size_x, batch_size_y).
    """
    # Use broadcasting to compute pairwise squared Euclidean distances
    # shape (batch_size_x, batch_size_y, features)
    diff = x.unsqueeze(1) - y.unsqueeze(0) 
    dist_sq = torch.sum(diff * diff, dim=2) # shape (batch_size_x, batch_size_y)
    
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2.0 * sigma**2)
        K += torch.exp(-gamma * dist_sq)
    return K

def mmd_gaussian_torch(points: torch.Tensor, labels: torch.Tensor, sigma_list=[1.0]) -> torch.Tensor:
    """
    Computes the unbiased MMD^2 estimate with an RBF kernel.
    
    Args:
        points (torch.Tensor): All points.
        labels (torch.Tensor): Labels of points (0 or 1).
        sigma_list (list): Bandwidths for the RBF kernel.
        
    Returns:
        torch.Tensor: The MMD^2 statistic.
    """
    k_xx = rbf_kernel(points[~labels], points[~labels], sigma_list)
    k_yy = rbf_kernel(points[labels], points[labels], sigma_list)
    k_xy = rbf_kernel(points[~labels], points[labels], sigma_list)

    # Compute the MMD statistic
    # Unbiased estimate can be used if desired, but here's a standard formulation:
    mmd_sq = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
    
    # The result should be non-negative
    return torch.clamp(mmd_sq, min=0)
