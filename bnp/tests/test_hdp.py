import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.special import psi
from sklearn.externals.six.moves import xrange

from sklearn.utils.testing import assert_almost_equal

from bnp.online_hdp import HierarchicalDirichletProcess

def _build_sparse_mtx(n_topics=10):
    # Create n_topics and each topic has 3 distinct words.
    # (Each word only belongs to a single topic.)
    block = n_topics * np.ones((3, 3))
    blocks = [block] * n_topics
    X = block_diag(*blocks)
    X = csr_matrix(X)
    return (n_topics, X)


def dumb_hdp_fit_test():
    """Test HDP fit"""

    n_topics, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'batch',
        'max_iter': 10,
    }
    hdp = HierarchicalDirichletProcess(**params)
    hdp.fit(X)


def dumb_hdp_partial_fit_test():
    """Test HDP partial_fit"""

    n_topics, X = _build_sparse_mtx()
    params = {
        'n_topic_truncate': 20,
        'n_doc_truncate': 5,
        'learning_method': 'online',
    }
    hdp = HierarchicalDirichletProcess(**params)
    for _ in xrange(10):
        hdp.partial_fit(X)
