import unittest

import numpy as np
from numpy.random import RandomState
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.extmath import logsumexp

from bnp.utils.extmath import row_log_normalize_exp


class TestExtMathUtils(unittest.TestCase):
    """Test utils.extmath"""

    def setUp(self):
        self.rand = RandomState(0)

    def test_row_log_normalize_exp(self):
        arr = self.rand.random_sample((400, 200))
        arr2 = arr.copy()
        row_log_normalize_exp(arr)
        arr2 -= logsumexp(arr2, axis=1)[:, np.newaxis]
        assert_almost_equal(arr, arr2)
