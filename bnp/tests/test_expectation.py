import unittest

import numpy as np
from numpy.random import RandomState
from scipy.special import psi

from sklearn.utils.testing import assert_almost_equal, assert_equal
from bnp.utils.expectation import (log_dirichlet_expectation,
                                   log_stick_expectation,
                                   stick_expectation)


class TestDirichletExpectation(unittest.TestCase):
    """Test log_dirichlet_expectation"""

    def setUp(self):
        self.rand = RandomState(0)

    def test_dirichlet_expectation_with_sampling(self):
        alpha = np.ones((10))
        samples = int(1e5)
        expectation = log_dirichlet_expectation(alpha)
        sample_mean = np.mean(np.log(self.rand.dirichlet(alpha, samples)), 0)
        assert_almost_equal(expectation, sample_mean, decimal=2)

    def test_2d_dirichlet_expectation(self):
        alpha = self.rand.choice(range(1, 10), 20).reshape(2, 10)
        exp1 = log_dirichlet_expectation(alpha)
        exp2 = (psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])
        assert_almost_equal(exp1, exp2)


class TestLogStickExpectation(unittest.TestCase):
    """Test log_stick_expectation"""

    def setUp(self):
        self.rand = RandomState(0)

    def test_log_stick_expectation_shape(self):
        num_stick = self.rand.randint(100, 200)
        var_sticks = np.ones((2, num_stick-1))
        expectation_stick = log_stick_expectation(var_sticks)
        assert_equal(expectation_stick.shape, (num_stick,))

    def test_uniform_log_stick_expectation(self):
        num_stick = 100
        var_sticks = np.array([np.ones(num_stick-1), np.arange(num_stick-1, 0, -1)])
        expectation_stick = log_stick_expectation(var_sticks)
        all_equal_stick = np.ones(expectation_stick.shape) * expectation_stick[0]
        assert_almost_equal(expectation_stick, all_equal_stick)


class TestStickExpectation(unittest.TestCase):
    """Test stick_expectation"""

    def setUp(self):
        self.rand = RandomState(0)

    def test_stick_expectation_shape(self):
        num_stick = self.rand.randint(100, 200)
        var_sticks = np.ones((2, num_stick-1))
        expectation_stick = stick_expectation(var_sticks)
        assert_equal(expectation_stick.shape, (num_stick,))
        assert_almost_equal(np.sum(expectation_stick), 1.0)

    def test_uniform_stick_expectation(self):
        num_stick = 100
        var_sticks = np.array([np.ones(num_stick-1), np.arange(num_stick-1, 0, -1)])
        expectation_stick = stick_expectation(var_sticks)
        all_equal_stick = np.ones(expectation_stick.shape) * expectation_stick[0]
        assert_almost_equal(expectation_stick, all_equal_stick)
        assert_almost_equal(np.sum(expectation_stick), 1.0)

