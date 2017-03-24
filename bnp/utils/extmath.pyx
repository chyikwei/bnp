cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport exp, log

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
