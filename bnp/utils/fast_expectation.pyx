cimport cython
cimport numpy as np
import numpy as np

np.import_array()

from libc.math cimport exp, fabs, log
from numpy.math cimport EULER


@cython.boundscheck(False)
@cython.wraparound(False)
def log_stick_expectation(np.ndarray[ndim=2, dtype=np.float64_t, mode="c"] sticks):
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

    cdef np.float64_t cum_elog_1_v, col_sum
    cdef np.ndarray[ndim=1, dtype=np.float64_t] elogsticks
    cdef np.npy_intp i, j, n_cols

    if sticks.shape[0] != 2:
        raise ValueError("sticks.shape[0] != 2")

    n_cols = sticks.shape[1]
    elogsticks = np.empty(n_cols + 1)

    cum_elog_1_v = 0.
    for i in range(n_cols):
        col_sum = psi(sticks[0][i] + sticks[1][i])
        elogsticks[i] = (psi(sticks[0][i]) - col_sum)
        elogsticks[i] += cum_elog_1_v
        cum_elog_1_v += (psi(sticks[1][i]) - col_sum)
    elogsticks[n_cols] = cum_elog_1_v
    return elogsticks


# Psi function for positive arguments. Optimized for speed, not accuracy.
#
# After: J. Bernardo (1976). Algorithm AS 103: Psi (Digamma) Function.
# http://www.uv.es/~bernardo/1976AppStatist.pdf
@cython.cdivision(True)
cdef double psi(double x) nogil:
    if x <= 1e-6:
        # psi(x) = -EULER - 1/x + O(x)
        return -EULER - 1. / x

    cdef double r, result = 0

    # psi(x + 1) = psi(x) + 1/x
    while x < 6:
        result -= 1. / x
        x += 1

    # psi(x) = log(x) - 1/(2x) - 1/(12x**2) + 1/(120x**4) - 1/(252x**6)
    #          + O(1/x**8)
    r = 1. / x
    result += log(x) - .5 * r
    r = r * r
    result -= r * ((1./12.) - r * ((1./120.) - r * (1./252.)))
    return result;
