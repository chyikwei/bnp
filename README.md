[![Build Status](https://circleci.com/gh/chyikwei/bnp.png?&style=shield)](https://circleci.com/gh/gh/chyikwei/bnp)
[![Coverage Status](https://coveralls.io/repos/github/chyikwei/bnp/badge.svg?branch=master)](https://coveralls.io/github/chyikwei/bnp?branch=master)

# Bayesian Nonparametric

Bayesian Nonparametric models with Python. Models will follow scikit-learn's interface.

Current model:
--------------
- Hierarchical Dirichlet Process (working on performance optimization now)

Reference:
----------
- "Stochastic Variational Inference", Matthew D. Hoffman, David M. Blei, Chong Wang, John Paisley, 2013
- "Online Variational Inference for the Hierarchical Dirichlet Process", Chong Wang, John Paisley, David M. Blei, 2011
- Chong Wang's [online-hdp code](https://github.com/blei-lab/online-hdp).

Install:
--------
```
# clone repoisitory
git clone git@github.com:chyikwei/bnp.git
cd bnp

# install dependencies (numpy, scipy, scikit-learn)
pip install -r requirements.txt
pip install .
```

Getting started:
----------------
```python
>>> from __future__ import print_function
>>> from numpy.random import RandomState
>>> from bnp import HierarchicalDirichletProcess
>>> from bnp.utils import make_doc_word_matrix
>>> rs = RandomState(100)

# Generate document-word matrix with 5 hidden topics (each topic has 10 uniuque words),
# and each topic has 100 docs.
>>> tf = make_doc_word_matrix(n_topics=5,
...                           words_per_topic=10,
...                           docs_per_topic=100,
...                           words_per_doc=50,
...                           shuffle=True,
...                           random_state=rs)
>>> tf.shape
(500, 50)

# check a few sample documents
>>> tf[0,:].todense()
matrix([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4, 11,  5,  3,
          2,  3,  3,  7,  6,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
>>> tf[1,:].todense()
matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3,
         6, 7, 6, 7, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0]])

# fit model
>>> hdp = HierarchicalDirichletProcess(n_topic_truncate=10,
...                                    n_doc_truncate=5,
...                                    max_iter=5,
...                                    verbose=1,
...                                    random_state=rs)
>>> hdp.fit(tf)
iteration: 1
iteration: 2
iteration: 3
iteration: 4
iteration: 5

# print topic function
>>> def print_top_words(model, n_words):
...     topic_distr = model.topic_distribution()
...     for topic_idx in range(model.lambda_.shape[0]):
...         topic = model.lambda_[topic_idx, :]
...         message = "Topic %d (proportion: %.2f): " % (topic_idx, topic_distr[topic_idx])
...         message += " ".join([str(i)
...                              for i in topic.argsort()[:-n_words - 1:-1]])
...         print(message)
...

# we can see hdp find 6 large topics map to the hidden topics we generate.
# Only topic 3 and 9 are are splited from the original hidden topic)
>>> print_top_words(hdp, 10)
Topic 0 (proportion: 0.00): 49 12 22 21 20 19 18 17 16 15
Topic 1 (proportion: 0.00): 49 12 22 21 20 19 18 17 16 15
Topic 2 (proportion: 0.00): 49 12 22 21 20 19 18 17 16 15
Topic 3 (proportion: 0.11): 34 30 37 39 31 32 36 35 33 38
Topic 4 (proportion: 0.20): 20 26 22 28 21 24 23 27 25 29
Topic 5 (proportion: 0.20): 46 48 44 41 47 43 40 49 42 45
Topic 6 (proportion: 0.20): 5 8 1 2 4 3 6 9 7 0
Topic 7 (proportion: 0.20): 15 14 17 19 10 18 13 12 16 11
Topic 8 (proportion: 0.00): 43 49 48 44 41 47 45 40 42 46
Topic 9 (proportion: 0.09): 38 33 35 36 32 31 37 30 34 39
```

Examples
--------
In `bnp/examples` folder. (Will add ipython notebook soon)


Running Test:
-------------
```
python setup.py test
```

Uninstall:
----------
```
pip uninstall bnp
```
