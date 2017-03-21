from __future__ import print_function

import numpy as np
from numpy.random import RandomState

from bnp.online_hdp import HierarchicalDirichletProcess
from bnp.utils.sample_generator import make_doc_word_matrix

n_iter = 5
n_topic_truncate = 10
n_doc_truncate = 5
n_top_words = 10


def print_top_words(model, n_words):
    topic_distr = model.topic_distribution()
    for topic_idx in range(model.lambda_.shape[0]):
        topic = model.lambda_[topic_idx, :]
        message = "Topic #%d (%.3f): " % (topic_idx, topic_distr[topic_idx])
        message += " ".join([str(i)
                             for i in topic.argsort()[:-n_words - 1:-1]])
        print(message)
    print()


rs = RandomState(100)

tf = make_doc_word_matrix(n_topics=5,
                          words_per_topic=10,
                          docs_per_topic=500,
                          words_per_doc=50,
                          shuffle=True,
                          random_state=rs)

hdp = HierarchicalDirichletProcess(n_topic_truncate=n_topic_truncate,
                                   n_doc_truncate=n_doc_truncate,
                                   omega=2.0,
                                   alpha=1.0,
                                   max_iter=5,
                                   max_doc_update_iter=200,
                                   random_state=100)

hdp.fit(tf)

print("\nTopics in HDP model:")
print_top_words(hdp, n_top_words)
