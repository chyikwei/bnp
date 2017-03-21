import unittest

import numpy as np
from numpy.random import RandomState
from sklearn.utils.testing import assert_equal, assert_array_equal, assert_true, assert_less

from bnp.utils import make_doc_word_matrix, make_uniform_doc_word_matrix


class TestSampleGenerator(unittest.TestCase):
    """Test sample_generator"""

    def setUp(self):
        self.rand = RandomState(0)

    def test_diag_matrix(self):
        """Test diag matrix"""
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

    def test_uniform_diag_matrix(self):
        """Test diag matrix with make uniform matrix"""
        n_topics = self.rand.randint(100, 200)
        params = {
            'n_topics': n_topics,
            'words_per_topic': 1,
            'docs_per_topic': 1,
        }
        matrix = make_uniform_doc_word_matrix(**params)
        dense = matrix.toarray()
        assert_array_equal(dense, np.eye(n_topics))

    def test_shuffle_uniform_diag_matrix(self):
        """Test suffle diag matrix with make uniform matrix"""
        n_topics = self.rand.randint(100, 200)
        params = {
            'n_topics': n_topics,
            'words_per_topic': 1,
            'docs_per_topic': 1,
            'shuffle': True
        }
        matrix = make_uniform_doc_word_matrix(**params)
        dense = matrix.toarray()
        diag_shift = False
        for idx in range(n_topics):
            if dense[idx, idx] < 1.:
                diag_shift = True
                break
        assert_true(diag_shift)

    def test_make_matrix_simple(self):
        """Test words per document
        """
        n_topics = self.rand.randint(100, 200)
        words_per_topic = 30
        words_per_doc = self.rand.randint(10, 20)

        params = {
            'n_topics': n_topics,
            'words_per_topic': words_per_topic,
            'docs_per_topic': 1,
            'words_per_doc': words_per_doc
        }
        matrix = make_doc_word_matrix(**params)
        dense = matrix.toarray()
        assert_equal(dense.shape[0], n_topics)
        assert_equal(dense.shape[1], words_per_topic * n_topics)
        row_sum = np.sum(dense, axis=1)
        assert_array_equal(row_sum, np.repeat(words_per_doc, dense.shape[0]))

    def test_make_matrix_words(self):
        """Test words in each doc are in the same topic
        """
        n_topics = self.rand.randint(100, 200)
        words_per_topic = 30
        words_per_doc = self.rand.randint(10, 20)

        params = {
            'n_topics': n_topics,
            'words_per_topic': words_per_topic,
            'docs_per_topic': 100,
            'words_per_doc': words_per_doc
        }
        matrix = make_doc_word_matrix(**params)
        dense = matrix.toarray()

        for i in xrange(dense.shape[0]):
            col_idx = np.where(dense[i, :] > 0)[0]
            max_idx = np.max(col_idx)
            min_idx = np.min(col_idx)
            assert_less(max_idx - min_idx, words_per_topic)

    def test_make_uniform_matrix(self):
        """Test words per document
        """
        n_topics = self.rand.randint(100, 200)
        words_per_topic = self.rand.randint(10, 20)
        docs_per_topic = self.rand.randint(100, 2000)

        params = {
            'n_topics': n_topics,
            'words_per_topic': words_per_topic,
            'docs_per_topic': docs_per_topic,
        }
        matrix = make_uniform_doc_word_matrix(**params)
        dense = matrix.toarray()
        assert_equal(dense.shape[0], n_topics * docs_per_topic)
        assert_equal(dense.shape[1], n_topics * words_per_topic)
        row_sum = np.sum(dense, axis=1)
        assert_array_equal(row_sum, np.repeat(words_per_topic, n_topics * docs_per_topic))
