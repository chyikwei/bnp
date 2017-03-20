import numpy as np
from scipy.special import psi
from sklearn.externals.six.moves import xrange
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import (assert_almost_equal, assert_raises_regexp,
                                   assert_equal)

from bnp.online_hdp import HierarchicalDirichletProcess
from bnp.utils.sample_generator import make_uniform_doc_word_matrix


def _build_sparse_mtx(n_topics=10):
    X = make_uniform_doc_word_matrix(
        n_topics=n_topics, words_per_topic=3, docs_per_topic=3)
    return (n_topics, X)


def test_hdp_fit():
    """Test HDP fit"""

    _, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
    }
    hdp = HierarchicalDirichletProcess(**params)
    hdp.fit(X)


def test_hdp_partial_fit():
    """Test HDP partial_fit"""

    _, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'online',
    }
    hdp = HierarchicalDirichletProcess(**params)
    for _ in xrange(10):
        hdp.partial_fit(X)


def test_hdp_transform():
    """Test HDP transform"""

    _, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
    }
    hdp = HierarchicalDirichletProcess(**params)

    assert_raises_regexp(NotFittedError, r"^no 'lambda_' attribute",
                         hdp.transform, X)
    hdp.fit(X)
    transformed = hdp.transform(X)
    assert_equal(transformed.shape[0], X.shape[0])
    assert_equal(transformed.shape[1], 20)


def test_hdp_fit_transform():
    """Test HDP fit_transform"""

    _, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
        'random_state': 1,
    }
    hdp1 = HierarchicalDirichletProcess(**params)
    hdp1.fit(X)
    transformed_1 = hdp1.transform(X)

    hdp2 = HierarchicalDirichletProcess(**params)
    transformed_2 = hdp2.fit_transform(X)
    assert_almost_equal(transformed_1, transformed_2)


def test_hdp_topic_distribution():
    """Test HDP topic_distribution"""
    _, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
        'random_state': 1,
    }
    hdp = HierarchicalDirichletProcess(**params)

    assert_raises_regexp(NotFittedError, r"^no 'lambda_' attribute",
                         hdp.transform, X)

    hdp.fit(X)
    topic_distr = hdp.topic_distribution()
    assert_almost_equal(np.sum(topic_distr), 1.0)


def test_partial_fit_after_fit():
    """Test run partial_fit after fit

    partial_fit should reset global parameters
    """
    _, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
        'random_state': 1,
    }
    hdp1 = HierarchicalDirichletProcess(**params)
    hdp1.fit(X)
    hdp1.partial_fit(X)
    hdp2 = HierarchicalDirichletProcess(**params)
    hdp2.partial_fit(X)
    assert_almost_equal(hdp1.transform(X), hdp2.transform(X))


def test_enable_likelihood():
    """Test enable doc_likelihood check

    The result should be the same no matter it
    is True or False.
    """
    _, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
        'random_state': 1,
        'check_doc_likelihood': True
    }
    hdp1 = HierarchicalDirichletProcess(**params)
    ret1 = hdp1.fit_transform(X)

    params['check_doc_likelihood'] = False
    hdp2 = HierarchicalDirichletProcess(**params)
    ret2 = hdp2.fit_transform(X)
    assert_almost_equal(ret1, ret2)
