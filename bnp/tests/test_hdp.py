import numpy as np

from sklearn.externals.six.moves import xrange
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import (assert_almost_equal, assert_raises_regexp,
                                   assert_equal, assert_true, assert_greater_equal)
from sklearn.utils import shuffle

from bnp.online_hdp import HierarchicalDirichletProcess
from bnp.utils import make_doc_word_matrix, make_uniform_doc_word_matrix


def _hdp_topic_check(hdp_model, n_topics, words_per_topic,
                     topics_threshold=0.1):
    """Check large topic inside HDP is grouped correctly"""

    topic_distr = hdp_model.topic_distribution()
    topic_covers = np.zeros(n_topics)
    for idx in xrange(hdp_model.lambda_.shape[0]):
        if topic_distr[idx] < topics_threshold:
            continue
        topic = hdp_model.lambda_[idx, :]
        top_word_idxs = topic.argsort()[:-words_per_topic - 1:-1]
        max_idx = np.max(top_word_idxs)
        min_idx = np.min(top_word_idxs)
        assert_equal(max_idx - min_idx, words_per_topic - 1)
        topic_covers[int(min_idx / words_per_topic)] = 1
    assert_true((topic_covers > 0.).all())


def test_hdp_fit_transform():
    """Test HDP fit_transform"""

    X = make_uniform_doc_word_matrix(
        n_topics=10, words_per_topic=3, docs_per_topic=3)

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


def test_hdp_dense_input():
    """Dense and sparse input should be the same"""

    X = make_uniform_doc_word_matrix(
        n_topics=10, words_per_topic=3, docs_per_topic=3)
    dense_X = X.todense()

    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
        'random_state': 1,
    }
    hdp1 = HierarchicalDirichletProcess(**params)
    transformed_1 = hdp1.fit_transform(dense_X)

    hdp2 = HierarchicalDirichletProcess(**params)
    transformed_2 = hdp2.fit_transform(X)
    assert_almost_equal(transformed_1, transformed_2)


def test_hdp_transform():
    """Test HDP transform"""

    X = make_uniform_doc_word_matrix(
        n_topics=10, words_per_topic=3, docs_per_topic=3)

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


def test_hdp_topic_distribution():
    """Test HDP topic_distribution"""

    X = make_uniform_doc_word_matrix(
        n_topics=10, words_per_topic=3, docs_per_topic=3)

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

    X = make_uniform_doc_word_matrix(
        n_topics=10, words_per_topic=3, docs_per_topic=3)

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


def test_likelihood_check():
    """Test enable doc_likelihood check

    The result should be the same no matter it
    is True or False.
    """
    X = make_uniform_doc_word_matrix(
        n_topics=10, words_per_topic=3, docs_per_topic=3)

    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
        'random_state': 1,
        'check_doc_likelihood': True,
        'evaluate_every': 1,
    }
    hdp1 = HierarchicalDirichletProcess(**params)
    ret1 = hdp1.fit_transform(X)

    params['check_doc_likelihood'] = False
    hdp2 = HierarchicalDirichletProcess(**params)
    ret2 = hdp2.fit_transform(X)
    assert_almost_equal(ret1, ret2)


def test_hdp_fit_topics_with_fake_data():
    """Test HDP fit with fake data

    Top words in large topics should be grouped correctly
    (small topic can be ignored.)
    """
    n_topics = 3
    n_topic_truncate = 10
    topics_threshold = 0.1
    words_per_topic = 10
    tf = make_doc_word_matrix(n_topics=n_topics,
                              words_per_topic=words_per_topic,
                              docs_per_topic=100,
                              words_per_doc=50,
                              shuffle=True,
                              random_state=0)

    hdp = HierarchicalDirichletProcess(n_topic_truncate=n_topic_truncate,
                                       n_doc_truncate=3,
                                       max_iter=5,
                                       random_state=0)

    hdp.fit(tf)
    _hdp_topic_check(hdp, n_topics, words_per_topic, topics_threshold)


def test_hdp_partial_fit_with_fake_data():
    """Test HDP partial_fit with fake data
    
    Same as `test_hdp_fit_topics_with_fake_data` but
    use `partial_fit` to replace `fit`
    """

    n_topics = 3
    n_topic_truncate = 10
    topics_threshold = 0.1
    words_per_topic = 10
    tf = make_doc_word_matrix(n_topics=n_topics,
                              words_per_topic=words_per_topic,
                              docs_per_topic=100,
                              words_per_doc=50,
                              shuffle=True,
                              random_state=0)

    hdp = HierarchicalDirichletProcess(n_topic_truncate=n_topic_truncate,
                                       n_doc_truncate=3,
                                       random_state=0)
    for _ in xrange(5):
        hdp.partial_fit(tf)
    _hdp_topic_check(hdp, n_topics, words_per_topic, topics_threshold)


def test_hdp_score():
    """Test HDP score function
    """
    n_topics = 10
    n_topic_truncate = 3
    words_per_topic = 10
    tf = make_doc_word_matrix(n_topics=n_topics,
                              words_per_topic=words_per_topic,
                              docs_per_topic=100,
                              words_per_doc=50,
                              shuffle=True,
                              random_state=0)

    hdp1 = HierarchicalDirichletProcess(n_topic_truncate=n_topic_truncate,
                                        n_doc_truncate=3,
                                        max_iter=1,
                                        random_state=0)

    hdp2 = HierarchicalDirichletProcess(n_topic_truncate=n_topic_truncate,
                                        n_doc_truncate=3,
                                        max_iter=5,
                                        random_state=0)
    hdp1.fit(tf)
    hdp2.fit(tf)
    score_1 = hdp1.score(tf)
    score_2 = hdp2.score(tf)
    assert_greater_equal(score_2, score_1)
