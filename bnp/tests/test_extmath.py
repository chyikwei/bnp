import unittest

import numpy as np
from numpy.random import RandomState
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.extmath import logsumexp

from bnp.utils.extmath import row_log_normalize_exp, mean_change_2d


class TestExtMathUtils(unittest.TestCase):
    """Test utils.extmath"""

    def setUp(self):
        self.rand = RandomState(0)

    def test_row_normalize_exp(self):
        arr = self.rand.random_sample((400, 200))
        arr2 = arr.copy()
        # in-place update
        row_log_normalize_exp(arr)
        arr2 -= logsumexp(arr2, axis=1)[:, np.newaxis]
        assert_almost_equal(arr, arr2)

    def test_mean_change_2d(self):
        arr1 = self.rand.random_sample((1000, 200))
        arr2 = self.rand.random_sample((1000, 200))
        ret1 = mean_change_2d(arr1, arr2)
        ret2 = np.abs(arr1 - arr2).mean()
        assert_almost_equal(ret1, ret2)
