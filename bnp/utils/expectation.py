"""Utilities for distribution expections"""

import numpy as np
from scipy.special import psi

from sklearn.externals.six.moves import xrange


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


def stick_expectation(sticks):
    """Expectation of stick-breaking process

    Parameters
    ----------
    sticks : array, [2 ,k]
        Each column is a pair of parameter of Beta distribution a_{k}, b_{k}

    Returns
    -------
    Elogsticks: array, [k+1,]
        this is E[sticks]
    """
    size = sticks.shape[1] + 1
    probs = np.zeros(size)
    expectations = sticks[0] / np.sum(sticks, axis=0)

    rest_stick = 1.
    for i in xrange(0, size-1):
        probs[i] = (rest_stick * expectations[i])
        rest_stick -= probs[i]
    probs[size-1] = rest_stick
    return probs
