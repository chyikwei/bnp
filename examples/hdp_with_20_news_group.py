"""HDP with 20 news group data

Modified form scikit-learn's "Topic extraction with NMF and LDA"
"""

# Author: Chyi-Kwei Yau <chyikwei.yau@gmail.com>

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

from bnp.online_hdp import HierarchicalDirichletProcess

n_samples = 4000
n_features = 1000
n_topic_truncate = 20
n_doc_truncate = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    topic_distr = model.topic_distribution()
    for topic_idx, topic in enumerate(model.lambda_):
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
data_samples = dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for HDP.
print("Extracting tf features for HDP...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

print("Fitting HDP models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
hdp = HierarchicalDirichletProcess(n_topic_truncate=n_topic_truncate,
                                   n_doc_truncate=n_doc_truncate,
                                   omega=10.0,
                                   max_iter=10,
                                   learning_method='batch',
                                   n_jobs=4,
                                   verbose=2,
                                   random_state=0)
t0 = time()
hdp.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in HDP model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(hdp, tf_feature_names, n_top_words)
