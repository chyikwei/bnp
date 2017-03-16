import unittest

import numpy as np
from numpy.random import RandomState
from sklearn.externals.six.moves import xrange
from sklearn.utils.testing import assert_equal, assert_array_equal

from bnp.utils.sample_generator import make_doc_word_matrix


class TestSampleGenerator(unittest.TestCase):
    """Test sample_generator"""

    def setUp(self):
        self.rand = RandomState(0)

    def test_simple_matrix(self):
        """Test diag matrix
        """
        n_topics = self.rand.randint(100, 200)
        params = {
            'n_topics': n_topics,
            'words_per_topic': 1,
            'docs_per_topic': 1,
            'words_per_doc': 1,
            'random_state': self.rand
        }
        matrix = make_doc_word_matrix(**params)
        dense = matrix.toarray()
        assert_array_equal(dense, np.eye(n_topics))

    def test_simple_matrix(self):
        """Test words per document
        """
        n_topics = self.rand.randint(100, 200)
        words_per_topic = 30
        words_per_doc = self.rand.randint(10, 20)

        params = {
            'n_topics': n_topics,
            'words_per_topic': words_per_topic,
            'docs_per_topic': 1,
            'words_per_doc': words_per_doc,
            'random_state': self.rand
        }
        matrix = make_doc_word_matrix(**params)
        dense = matrix.toarray()
        assert_equal(dense.shape[0], n_topics)
        assert_equal(dense.shape[1], words_per_topic * n_topics)
        row_sum = np.sum(dense, axis=1)
        assert_array_equal(row_sum, np.repeat(words_per_doc, dense.shape[0]))
