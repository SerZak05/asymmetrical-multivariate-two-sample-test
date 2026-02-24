import numpy as np
import torch, torch_geometric
import scipy.stats as ss
import typing as tp
from sklearn.neighbors import kneighbors_graph
from my_statistics import compute_l, compute_l_torch, compute_symm_l, compute_symm_l_torch, mmd_lin, mmd_gaussian, mmd_gaussian_torch


def get_params(n: int, n1: int, A: np.ndarray) -> tuple[float, float]:
    """Returns parameters of asymptotic normal distribution of asymmetrical test.

    Args:
        n (int): Number of points.
        n1 (int): Number of points in the first class.
        A (np.ndarray): Adjacency matrix of 1-NN graph
    
    Returns:
        Tuple of mean and variation of normal distribution.
    """
    mu = n1*(n1 - 1) / (n-1)
    d = A.sum(axis=0).ravel()
    cn = np.mean((d - 1)**2)
    vn = np.sum(A * A.T) / n / 2
    var = 4*n1*(n1-1)*(n-n1)*n / (n*(n-1)*(n-2)*(n-3)) * ((n1 - 2) * cn + (n - n1 - 1) * (1 + vn - 2/(n-1)))
    var /= 4
    return mu, var

def get_params_torch(n: int, n1: int, A: torch.Tensor) -> tuple[float, float]:
    """Returns parameters of asymptotic normal distribution of asymmetrical test.

    Args:
        n (int): Number of points.
        n1 (int): Number of points in the first class.
        A (torch.Tensor): Adjacency matrix of 1-NN graph
    
    Returns:
        Tuple of mean and variation of normal distribution.
    """
    mu = n1*(n1 - 1) / (n-1)
    d = A.sum(dim=0).ravel()
    cn = torch.mean((d - 1)**2).cpu().item()
    vn = torch.sum(A * A.T).cpu().item() / n / 2
    var = 4*n1*(n1-1)*(n-n1)*n / (n*(n-1)*(n-2)*(n-3)) * ((n1 - 2) * cn + (n - n1 - 1) * (1 + vn - 2/(n-1)))
    var /= 4
    return mu, var


def get_symm_params(n: int, n1: int, A: np.ndarray) -> tuple[float, float]:
    """Returns parameters of asymptotic normal distribution of symmetrical (original) test
    Args:
        n (int): Number of points.
        n1 (int): Number of points in the first class.
        A (np.ndarray): Adjacency matrix of 1-NN graph
    
    Returns:
        Tuple of mean and variation of normal distribution.
    """
    n2 = n - n1
    mu = (n1*(n1 - 1) + n2*(n2 - 1)) / (n-1)
    d = A.sum(axis=0).ravel()
    cn = np.mean((d - 1)**2)
    vn = np.sum(A * A.T) / n / 2
    q = 4 * (n1 - 1) * (n2 - 1) / ((n - 2) * (n - 3))
    var = n1 * n2 / (n - 1) * (q * (1 + vn - 2 / (n - 1)) + (1-q)*cn)

    return mu, var

def get_symm_params_torch(n: int, n1: int, A: torch.Tensor) -> tuple[float, float]:
    """Returns parameters of asymptotic normal distribution of symmetrical (original) test
    Args:
        n (int): Number of points.
        n1 (int): Number of points in the first class.
        A (torch.Tensor): Adjacency matrix of 1-NN graph
    
    Returns:
        Tuple of mean and variation of normal distribution.
    """
    n2 = n - n1
    mu = (n1*(n1 - 1) + n2*(n2 - 1)) / (n-1)
    d = A.sum(dim=0).ravel()
    cn = torch.mean((d - 1)**2).item()
    vn = torch.sum(A * A.T).item() / n / 2
    q = 4 * (n1 - 1) * (n2 - 1) / ((n - 2) * (n - 3))
    var = n1 * n2 / (n - 1) * (q * (1 + vn - 2 / (n - 1)) + (1-q)*cn)

    return mu, var


def reject_asymm_hypo(points1: np.ndarray, points2: np.ndarray, alpha: float = 0.05) -> bool:
    """Asymptotical test based on asymmetric nearest neighbour type coincidences
    Args:
        points1 (np.ndarray): First class points.
        points2 (np.ndarray): Second class points.
        a (float): Significance level.
    
    Returns:
        bool: True, if H0 is rejected.
    """
    n1 = len(points1)
    n = n1 + len(points2)
    points = np.vstack((points1, points2))
    A = kneighbors_graph(points, 1, mode='connectivity', include_self=False)
    A = np.array(A.todense(), dtype=np.int32)

    mask = np.zeros(n, dtype=np.bool)
    mask[:n1] = True
    l = compute_l(mask, A)
    mu, var = get_params(n, n1, A)
    c = mu + np.sqrt(var) * ss.norm.ppf(1 - alpha)

    return l >= c

@torch.no_grad
def reject_asymm_hypo_torch(points1: torch.Tensor, points2: torch.Tensor, alpha: float = 0.05, device="cuda") -> bool:
    """Asymptotical test based on asymmetric nearest neighbour type coincidences
    Args:
        points1 (torch.Tensor): First class points.
        points2 (torch.Tensor): Second class points.
        a (float): Significance level.
    
    Returns:
        bool: True, if H0 is rejected.
    """
    if points1 is not torch.Tensor or points2 is not torch.Tensor:
        points1 = torch.tensor(points1, device=device)
        points2 = torch.tensor(points2, device=device)
    n1 = len(points1)
    n = n1 + len(points2)
    points = torch.vstack((points1, points2))
    # A = kneighbors_graph(points, 1, mode='connectivity', include_self=False)
    # A = np.array(A.todense(), dtype=np.int32)
    A = torch_geometric.utils.to_dense_adj(torch_geometric.nn.knn_graph(points, k=1, loop=False)).squeeze(0)
    # print(A.shape)

    mask = torch.zeros(n, device=device)
    mask[:n1] = 1
    l = compute_l_torch(mask, A)
    mu, var = get_params_torch(n, n1, A)
    c = mu + np.sqrt(var) * ss.norm.ppf(1 - alpha)

    return l >= c



def reject_asymm_rev_hypo_torch(points1: torch.Tensor, points2: torch.Tensor, alpha: float = 0.05) -> bool:
    """
    Asymptotic test based on asymmetric nearest neighbour type coincidences. Reverses the order of classes.
    Args:
        points1 (torch.Tensor): Second class points.
        points2 (torch.Tensor): First class points.
        alpha (float): Significance level.
    
    Returns:
        bool: True, if H0 is rejected.
    """
    return reject_asymm_hypo_torch(points2, points1, alpha)


def reject_symm_hypo(points1: np.ndarray, points2: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Asymptotic test based on nearest neighbour type coincidences (original paper).
    Args:
        points1 (np.ndarray): First class points.
        points2 (np.ndarray): Second class points.
        alpha (float): Significance level.
    
    Returns:
        bool: True, if H0 is rejected.
    """
    n1 = len(points1)
    n = n1 + len(points2)
    points = np.vstack([points1, points2])
    A = kneighbors_graph(points, 1, mode='connectivity', include_self=False)
    A = np.array(A.todense())
    mask = np.zeros(n, dtype=bool)
    mask[:n1] = True
    l = compute_symm_l(mask, A)
    mu, var = get_symm_params(n, n1, A)
    c = mu + np.sqrt(var) * ss.norm.ppf(1 - alpha)
    return l >= c

def reject_symm_hypo_torch(points1: torch.Tensor, points2: torch.Tensor, alpha: float = 0.05, device="cuda") -> bool:
    """
    Asymptotic test based on nearest neighbour type coincidences (original paper).
    Args:
        points1 (torch.Tensor): First class points.
        points2 (torch.Tensor): Second class points.
        alpha (float): Significance level.
    
    Returns:
        bool: True, if H0 is rejected.
    """
    if points1 is not torch.Tensor or points2 is not torch.Tensor:
        points1 = torch.tensor(points1, device=device)
        points2 = torch.tensor(points2, device=device)
    n1 = len(points1)
    n = n1 + len(points2)
    points = torch.vstack([points1, points2])
    A = torch_geometric.utils.to_dense_adj(torch_geometric.nn.knn_graph(points, k=1, loop=False)).squeeze(0)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[:n1] = True
    l = compute_symm_l_torch(mask, A)
    mu, var = get_symm_params_torch(n, n1, A)
    c = mu + np.sqrt(var) * ss.norm.ppf(1 - alpha)
    return l >= c


def reject_mmd2_u_hypo(points1: np.ndarray, points2: np.ndarray, mmd_func, alpha: float = 0.05, bootstrap_iters: int = 50) -> bool:
    """
    Asymptotic test based on MMD^2_u statistics. Uses bootstrap to approximate the distribution.
    Args:
        points1 (np.ndarray): First class points.
        points2 (np.ndarray): Second class points.
        mmd_func (function): Function that computes mmd statistics.
        alpha (float): Significance level.
        bootstrap_iters (int): Number of bootstrap samples.

    Returns:
        bool: True, if H0 is rejected.
    """
    m = len(points1)
    mmd_obs = mmd_func(points1, points2)
    points = np.vstack([points1, points2])
    bootstrap_samples = np.zeros(bootstrap_iters)
    for i in range(bootstrap_iters):
        np.random.shuffle(points)
        bootstrap_samples[i] = mmd_func(points[:m], points[m:])
    
    threshold = np.quantile(bootstrap_samples, 1 - alpha)
    return mmd_obs >= threshold

def reject_mmd2_u_lin_hypo(points1, points2, alpha = 0.05):
    """
    Asymptotic test based on MMD^2_u statistics with linear kernel. Calls `reject_mmd2_u_hypo` with `mmd_lin`.
    Args:
        points1 (np.ndarray): First class points.
        points2 (np.ndarray): Second class points.
        alpha (float): Significance level.
    
    Returns:
        bool: True, if H0 is rejected.
    """
    return reject_mmd2_u_hypo(points1, points2, mmd_lin, alpha)

def reject_mmd2_u_gaussian_hypo(points1, points2, alpha = 0.05):
    """
    Asymptotic test based on MMD^2_u statistics with gaussian kernel. Calls `reject_mmd2_u_hypo` with `mmd_gaussian`.
    Args:
        points1 (np.ndarray): First class points.
        points2 (np.ndarray): Second class points.
        alpha (float): Significance level.
    
    Returns:
        bool: True, if H0 is rejected.
    """
    return reject_mmd2_u_hypo(points1, points2, mmd_gaussian, alpha)

def reject_mmd2_l_hypo(points1: np.ndarray, points2: np.ndarray, alpha = 0.05) -> bool:
    """
    Asymptotic test based on MMD^2_l statistics.
    Args:
        points1 (np.ndarray): First class points.
        points2 (np.ndarray): Second class points.
        alpha (float): Significance level.
    
    Returns:
        bool: True, if H0 is rejected.
    """
    assert points1.shape == points2.shape
    m = len(points1)
    assert m % 2 == 0
    d = points1.shape[1]
    pairs1 = points1.reshape(-1, 2, d)
    pairs2 = points2.reshape(-1, 2, d)
    diffs = pairs1 - pairs2
    h_vals = np.sum(diffs[:, 0, :] * diffs[:, 1, :], axis=1)
    mmd2_lin = np.mean(h_vals)
    sigma2_hat = 2 * np.mean(h_vals**2)
    z_test = mmd2_lin / np.sqrt(sigma2_hat / m)
    z_crit = ss.norm.ppf(1 - alpha / 2)
    return np.abs(z_test) >= z_crit

def reject_mmd2_u_gaussian_hypo_torch(points1: torch.Tensor, points2: torch.Tensor, alpha: float = 0.05, bootstrap_iters: int = 10, device = "cuda") -> bool:
    """
    Asymptotic test based on MMD^2_u statistics. Uses bootstrap to approximate the distribution.
    Args:
        points1 (torch.Tensor): First class points.
        points2 (torch.Tensor): Second class points.
        alpha (float): Significance level.
        bootstrap_iters (int): Number of bootstrap samples.

    Returns:
        bool: True, if H0 is rejected.
    """
    if points1 is not torch.Tensor or points2 is not torch.Tensor:
        points1 = torch.tensor(points1, device=device)
        points2 = torch.tensor(points2, device=device)
    m = len(points1)
    n = len(points1) + len(points2)
    labels = torch.zeros(n, dtype=torch.bool, device=device)
    labels[m:] = True
    points = torch.vstack([points1, points2])

    mmd_obs = mmd_gaussian_torch(points, labels)

    bootstrap_samples = torch.zeros(bootstrap_iters, device=device)
    perm = torch.randperm(n, device=device)
    for i in range(bootstrap_iters):
        bootstrap_samples[i] = mmd_gaussian_torch(points, labels[perm])
        perm = torch.randperm(n)
    
    threshold = torch.quantile(bootstrap_samples, 1 - alpha)
    return mmd_obs.cpu().item() >= threshold.cpu().item()