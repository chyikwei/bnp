"""Generate samples dataset
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state
from sklearn.externals.six.moves import xrange


def make_doc_word_matrix(n_topics, words_per_topic, docs_per_topic,
                         words_per_doc, shuffle=False, random_state=None):
    """Generate document-word matrix with hidden topics

    Each row is a document with `words_per_doc` words from
    hidden topic and each word is sampled from its hidden
    topic.

    Parameters
    ----------
    n_topics : int
        number of topics

    words_per_topic : int
        number of words per topic

    docs_per_topic : int
        number of documents per topic

    words_per_doc : int
        number of words per document

    shuffle : boolean, optional (default=False)
        shuffle returned matrix or not

    random_state : int or RandomState instance or None, optional (default=None)
        Pseudo-random number generator seed control.

    Return
    ------
    matrix: doc-words matrix. It is a sparse matrix of shape
            [n_topics * docs_per_topic, n_topics * words_per_topic]

    """
    random_state_ = check_random_state(random_state)

    word_cnt = n_topics * words_per_topic
    doc_cnt = n_topics * docs_per_topic
    total_topic_words = docs_per_topic * words_per_doc

    topic_word_ids = np.arange(word_cnt).reshape(n_topics, words_per_topic)
    topic_words = []
    for topic_idx in xrange(n_topics):
        doc_samples = random_state_.choice(
            topic_word_ids[topic_idx, :], total_topic_words, replace=True).\
            reshape(docs_per_topic, words_per_doc)
        topic_words.append(doc_samples)
    doc_words = np.vstack(topic_words)

    row_idx = []
    col_idx = []
    data = []
    for doc_idx in xrange(doc_words.shape[0]):
        word_idx, cnts = np.unique(doc_words[doc_idx, :], return_counts=True)
        row_idx.append(np.repeat(doc_idx, word_idx.shape[0]))
        col_idx.append(word_idx)
        data.append(cnts)
    row_idx = np.hstack(row_idx)
    col_idx = np.hstack(col_idx)
    data = np.hstack(data)

    matrix = csr_matrix((data, (row_idx, col_idx)), shape=(doc_cnt, word_cnt))
    if shuffle:
        new_idx = np.arange(doc_cnt)
        random_state_.shuffle(new_idx)
        matrix = matrix[new_idx, :]
    return matrix


def make_uniform_doc_word_matrix(n_topics, words_per_topic, docs_per_topic,
                                 shuffle=False, random_state=None):
    """Generate uniform document-word matrix with hidden topics

    Similar to `make_doc_word_matrix`. The only difference is we set all
    word count in the hidden topic to 1.

    Parameters
    ----------
    n_topics : int
        number of topics

    words_per_topic: int
        number of words per topic

    docs_per_topic: int
        number of documents per topic

    shuffle : boolean, optional (default=False)
        shuffle returned matrix or not

    random_state : int or RandomState instance or None, optional (default=None)
        Pseudo-random number generator seed control.

    Return
    ------
    matrix: doc-words matrix. It is a sparse matrix of shape
            [n_topics * docs_per_topic, n_topics * words_per_topic]

    """

    block = np.ones((docs_per_topic, words_per_topic))
    blocks = [block] * n_topics
    matrix = block_diag(*blocks)
    if shuffle:
        random_state_ = check_random_state(random_state)
        new_idx = np.arange(matrix.shape[0])
        random_state_.shuffle(new_idx)
        matrix = matrix[new_idx]
    return csr_matrix(matrix)
