from __future__ import print_function
from time import time

import numpy as np
from numpy.random import RandomState
from sklearn.utils import shuffle

from sklearn.decomposition import LatentDirichletAllocation
from bnp.online_hdp import HierarchicalDirichletProcess
from bnp.utils.sample_generator import make_doc_word_matrix

n_iter = 5
n_topic_truncate = 10
n_doc_truncate = 5
n_top_words = 10


def print_top_words(model, n_top_words):
    topic_distr = model.topic_distribution()
    topic_order = np.argsort(topic_distr)[::-1]
    for topic_idx in xrange(model.lambda_.shape[0]):
        topic = model.lambda_[topic_idx, :]
        message = "Topic #%d (%.3f): " % (topic_idx, topic_distr[topic_idx])
        message += " ".join([str(i)
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


rs = RandomState(100)

tf = make_doc_word_matrix(n_topics=5,
                          words_per_topic=10,
                          docs_per_topic=1000,
                          words_per_doc=50,
                          random_state=rs)


hdp = HierarchicalDirichletProcess(n_topic_truncate=n_topic_truncate,
                                   n_doc_truncate=n_doc_truncate,
                                   omega=2.0,
                                   alpha=1.0,
                                   kappa=0.7,
                                   tau=64.,
                                   max_iter=10,
                                   learning_method='online',
                                   batch_size=250,
                                   total_samples=1e5,
                                   max_doc_update_iter=200,
                                   verbose=1,
                                   mean_change_tol=1e-3,
                                   random_state=100)

#for i in range(5):
#    t0 = time()
#    print("iter %d" % i)
#    suffled_tf = shuffle(tf, random_state=rs)
#    hdp.partial_fit(suffled_tf)
#    print("done in %0.3fs." % (time() - t0))


#print("\nTopics in HDP model:")
#print_top_words(hdp, n_top_words)

print("\nTopics in LDA model:")
lda = LatentDirichletAllocation(n_topics=n_topic_truncate, max_iter=5,
                                learning_method='online',
                                learning_offset=64.,
                                random_state=0)

for i in range(5):
    t0 = time()
    print("iter %d" % i)
    suffled_tf = shuffle(tf, random_state=rs)
    lda.partial_fit(suffled_tf)
    print("done in %0.3fs." % (time() - t0))

for topic_idx in xrange(lda.components_.shape[0]):
        topic = lda.components_[topic_idx, :]
        message = "Topic #%d: " % (topic_idx)
        message += " ".join([str(i)
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
