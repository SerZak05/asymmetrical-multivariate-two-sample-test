import numpy as np
import tqdm
import scipy.stats as ss


def test_significance_level(reject_fn, n_iter=1000, n=500, d=200):
    """Computes observed significance level of given test.
    Args:
        reject_fn (function(points1, points2)): Function (given test) that rejects H0.
        n_iter (int): Number of samples to test on.
        n (int): Number of points in each sample.
        d (int): Dimentionality of points.
    
    Returns:
        float: Observed significance level.
    """
    n1 = n // 2
    n2 = n - n1
    # H0 is true
    rej_cnt = 0
    for _ in tqdm.tqdm(range(n_iter)):
        points1 = ss.norm.rvs(size=(n1, d))
        points2 = ss.norm.rvs(size=(n2, d))
        rej_cnt += reject_fn(points1, points2)
    return rej_cnt / n_iter


def power_mean_experiment(tests: list, n_iter=100, n1=250, n2=250):
    """Experiment with two normal distributions with different means and unit variance. Plots the results.
    Args:
        tests (tuple[function]): Tests to experiment on.
        n_iter (int): Number of experiments per graph point (per dimentionality).
        n1 (int): Number of first class points
        n2 (int): Number of second class points
    
    Returns:
        None
    """
    means = np.logspace(np.log10(0.05), np.log10(50), 20, dtype=np.float32)
    x = np.logspace(0.5, 2.0, 20, dtype=np.float32)
    plot_points = [[] for test in tests]
    for d in tqdm.tqdm(x):
        d = int(d)
        powers = np.zeros(len(tests))
        for mean in means:
            rej_cnts = np.zeros(len(tests))
            for _ in range(n_iter):
                points1 = ss.norm.rvs(scale=np.sqrt(d), size=(n1, d))
                points2 = ss.norm.rvs(loc=mean, scale=np.sqrt(d), size=(n2, d))
                for i, test in enumerate(tests):
                    rej_cnts[i] += test(points1, points2)
            powers += rej_cnts / n_iter
        powers /= len(means)
        for i in range(len(tests)):
            plot_points[i].append(powers[i])
    return x, plot_points


def power_var_experiment(tests: list, n_iter=100, n1=250, n2=250):
    """Experiment with two normal distributions with mean = 0 and different variance. Plots the results.
    Args:
        tests (tuple[function]): Tests to experiment on.
        n_iter (int): Number of experiments per graph point (per dimentionality).
        n1 (int): Number of first class points
        n2 (int): Number of second class points
    
    Returns:
        None
    """
    sigmas = np.logspace(0.01, 1.0, 20, base=10)
    x = np.logspace(0.5, 2.0, 20)
    plot_points = [[] for test in tests]
    for d in tqdm.tqdm(x):
        d = int(d)
        powers = np.zeros(len(tests))
        for sigma in sigmas:
            rej_cnts = np.zeros(len(tests))
            for _ in range(n_iter):
                points1 = ss.norm.rvs(size=(n1, d))
                points2 = ss.norm.rvs(scale=sigma, size=(n2, d))
                for i, test in enumerate(tests):
                    rej_cnts[i] += test(points1, points2)
            powers += rej_cnts / n_iter
        powers /= len(sigmas)
        for i in range(len(tests)):
            plot_points[i].append(powers[i])
    return x, plot_points