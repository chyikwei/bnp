"""Utilities for distribution expections"""

import numpy as np
from scipy.special import psi


def log_dirichlet_expectation(alpha):
    """Expectation log dirichlet distribution

    For a vector theta ~ Dirichlet(alpha):
    E[log(theta_{i})] = psi(alpha_{i}) - psi(sum(alpha))


    Parameters
    ----------
    alpha : array [n,] or [n, m]
    

    Returns
    -------
    array: same shape as `aplha`

    """
    if (len(alpha.shape) == 1):
        # 1-dim
        return psi(alpha) - psi(np.sum(alpha))
    else:
        # 2-dim
        return (psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


def log_stick_expectation(sticks):
    """Expectation of log stick-breaking process

    Parameters
    ----------
    sticks : array, [2 ,k]
        Each column is a pair of parameter of Beta distribution a_{k}, b_{k}

    Returns
    -------
    Elogsticks: array, [k+1,]
        this is E[log(sticks)]

    """

    # psi(a_{k} + b_{k}) for k = {1,2,..., K-1}
    stick_sum = psi(np.sum(sticks, 0))
    # E[log(V_{k})] = psi(a_{k}) - psi(a_{k} + b_{k}) for k = {1,2,...,K-1}
    expectation_log_v = psi(sticks[0]) - stick_sum
    # E[log(1 - V_{k})] = psi(b_{k}) - psi(a_{k} + b_{k}) for k = {1,2,...,K-1}
    expacetaion_log_1_minus_v = psi(sticks[1]) - stick_sum 

    size = sticks.shape[1] + 1
    Elogsticks = np.zeros(size)
    ## E[log(sigma_{k}(V))] =  E[log(V_{k})] + sum_{1 to k-1}(E[log(1 - V_{l})])
    Elogsticks[0: (size - 1)] = expectation_log_v
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(expacetaion_log_1_minus_v)
    return Elogsticks
