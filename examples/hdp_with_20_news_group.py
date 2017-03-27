"""HDP with 20 news group data

Modified form scikit-learn's "Topic extraction with NMF and LDA"
"""

# Author: Chyi-Kwei Yau <chyikwei.yau@gmail.com>

from __future__ import print_function
from time import time

import numpy as np
from numpy.random import RandomState
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import shuffle

from bnp.online_hdp import HierarchicalDirichletProcess

n_iter = 5
n_features = 1000
n_topic_truncate = 50
n_doc_truncate = 10
n_top_words = 10
n_top_topics = 5
n_inference_docs = 20

rs = RandomState(100)

def print_top_words(model, feature_names, n_top_words):
    topic_distr = model.topic_distribution()
    for topic_idx in range(model.lambda_.shape[0]):
        topic = model.lambda_[topic_idx, :]
        message = "Topic #%d (%.3f): " % (topic_idx, topic_distr[topic_idx])
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
target_names = dataset.target_names
train_samples = dataset.data[:-n_inference_docs]
train_targets = dataset.target[:-n_inference_docs]
inference_samples = dataset.data[-n_inference_docs:]
inference_targets = dataset.target[-n_inference_docs:]
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for HDP.
print("Extracting tf features for HDP...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(train_samples)
print("done in %0.3fs." % (time() - t0))
print()

print("Fitting HDP models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (tf.shape[0], n_features))
hdp = HierarchicalDirichletProcess(n_topic_truncate=n_topic_truncate,
                                   n_doc_truncate=n_doc_truncate,
                                   omega=2.0,
                                   alpha=1.0,
                                   kappa=0.7,
                                   tau=64.,
                                   max_iter=10,
                                   learning_method='online',
                                   batch_size=250,
                                   total_samples=1e6,
                                   max_doc_update_iter=200,
                                   verbose=1,
                                   mean_change_tol=1e-3,
                                   random_state=100)

for i in range(5):
    t0 = time()
    print("iter %d" % i)
    suffled_tf = shuffle(tf, random_state=rs)
    hdp.partial_fit(suffled_tf)
    print("done in %0.3fs." % (time() - t0))

print("\nTopics in HDP model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(hdp, tf_feature_names, n_top_words)

# top topics in each group
print("\nTop topics in each group:")
train_topics = hdp.transform(tf)
# normalize
train_topics =  train_topics / np.sum(train_topics, axis=1)[:, np.newaxis]
for grp_idx, group_name in enumerate(target_names):
    doc_idx = np.where(train_targets == grp_idx)[0]
    mean_doc_topics = np.mean(train_topics[doc_idx, :], axis=0)
    top_idx = mean_doc_topics.argsort()[:-n_top_topics - 1:-1]
    print("group: %s:" % group_name)
    print("top topics: %s" % (", ".join(["#%d (%.3f)" % (idx, mean_doc_topics[idx]) for idx in top_idx])))
    print()
