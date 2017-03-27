cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport exp, fabs, log

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def row_log_normalize_exp(np.ndarray[ndim=2, dtype=np.float64_t, mode="c"] arr):
    """
    This will normalize each row for np.exp(arr), take log, and
    do is in-place update.

    Equivalent to:
        arr -= logsumexp(arr, axis=1)[:, np.newaxis]"

    """
    cdef np.float64_t v_max, v_sum
    cdef np.npy_intp i, j, n_rows, n_cols

    n_rows = arr.shape[0]
    n_cols = arr.shape[1]
    for i in range(n_rows):
        v_max = arr[i, 0]
        for j in range(n_cols):
            if arr[i, j] > v_max:
                v_max = arr[i, j]

        #sum of exp value
        v_sum = 0.0
        for j in range(n_cols):
            v_sum += exp(arr[i, j] - v_max)
        # logsumexp value
        v_sum = log(v_sum) + v_max

        # update row value
        for j in range(n_cols):
            arr[i, j] -= v_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def mean_change_2d(np.ndarray[ndim=2, dtype=np.float64_t] arr_1,
                np.ndarray[ndim=2, dtype=np.float64_t] arr_2):
    """Calculate the mean difference between two arrays.

    This is equivalent to np.abs(arr_1 - arr2).mean().
    """

    cdef np.float64_t total, diff
    cdef np.npy_intp i, j, n_rows, n_cols

    n_rows = arr_1.shape[0]
    n_cols = arr_1.shape[1]

    if arr_2.shape[0] != n_rows or arr_2.shape[1] != n_cols:
        raise ValueError("arr_1 and arr_2 shape mismatch")

    total = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            diff = fabs(arr_1[i, j] - arr_2[i, j])
            total += diff

    return total / (n_rows * n_cols)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def beta_param_update(double alpha,
                   np.ndarray[ndim=1, dtype=np.float64_t] row_stats,
                   np.ndarray[ndim=2, dtype=np.float64_t, mode="c"] out):
    """In-place update beta distribution params

    Equivalent to:
        T = row_stats.shape[0]
        # out.shape = (2, T-1)
        out[0] = 1.0 + row_stats[:T-1]
        out[1] = alpha + np.flipud(np.cumsum(np.flipud(row_stats[1:])))
    """
    cdef np.float64_t temp_sum
    cdef np.npy_intp i, j, n_cols

    n_cols = row_stats.shape[0]

    if out.shape[0] != 2:
        raise ValueError("out.shape[0] != 2")

    if out.shape[1] != (n_cols - 1):
        raise ValueError("row_stats and out shape mismatch")

    # param 0: 1 + sum(row sum of sstats)
    for i in range(n_cols - 1):
        out[0, i] = 1. + row_stats[i]

    # reverse sum
    temp_sum = 0.0
    for i from n_cols > i > 0:
        temp_sum += row_stats[i]
        out[1, i-1] = alpha + temp_sum
