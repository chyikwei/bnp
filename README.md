[![Build Status](https://travis-ci.org/chyikwei/bnp.svg?branch=master)](https://travis-ci.org/chyikwei/bnp)
[![Build Status](https://circleci.com/gh/chyikwei/bnp.png?&style=shield)](https://circleci.com/gh/gh/chyikwei/bnp)
[![Coverage Status](https://coveralls.io/repos/github/chyikwei/bnp/badge.svg?branch=master)](https://coveralls.io/github/chyikwei/bnp?branch=master)

# Bayesian Nonparametric
Bayesian Nonparametric models with Python.

Models follow scikit-learn's API and can be used as its extension.

Current model:
--------------
- **Hierarchical Dirichlet Process**

   HDP is similar to LDA (Latent Direchlet Allocation) but assumes an "infinite" number of topics. This implementation is based on Chong Wang's online-hdp and optimized with cython.
  

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

# install dependencies (cython, numpy, scipy, scikit-learn)
pip install -r requirements.txt
pip install .
```

Getting started:
----------------
In `bnp.utils` we proivde a function to generate fake document-word matrix with hidden topics. We will run our HDP model with it.

First, we can generate a document-word matrix with 5 hidden topics. (each topic has 10 uniuque words and each topic has 100 docs.)

```python
>>> from __future__ import print_function
>>> from bnp.online_hdp import HierarchicalDirichletProcess
>>> from bnp.utils import make_doc_word_matrix

>>> tf = make_doc_word_matrix(n_topics=5,
...                           words_per_topic=10,
...                           docs_per_topic=100,
...                           words_per_doc=20,
...                           shuffle=True,
...                           random_state=0)
>>> tf.shape
(500, 50)
```

For samples in the matrix, each row(document) only contains words from a specific topic (word 0 to 9: topic 1, 10 to 19: topic 2,...)

```python
>>> tf[0].toarray()
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 4, 1, 2, 3, 3, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0]])
>>> tf[1].toarray()
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 3, 1, 3, 2, 1, 2, 0, 3, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0]])
```

Next we fit a HDP model with this matrix

```python
>>> hdp = HierarchicalDirichletProcess(n_topic_truncate=10,
...                                    n_doc_truncate=3,
...                                    max_iter=5,
...                                    random_state=0)
>>> hdp.fit(tf)
```

Then we can print out topic proportion and top topic words in HDP model.

```python
# print topic function
>>> def print_top_words(model, n_words):
...     topic_distr = model.topic_distribution()
...     for topic_idx in range(model.lambda_.shape[0]):
...         topic = model.lambda_[topic_idx, :]
...         message = "Topic %d (proportion: %.2f): " % (topic_idx, topic_distr[topic_idx])
...         message += " ".join([str(i) for i in topic.argsort()[:-n_words - 1:-1]])
...         print(message)

>>> print_top_words(hdp, 10)
Topic 0 (proportion: 0.20): 3 1 7 5 8 4 0 2 9 6
Topic 1 (proportion: 0.00): 49 12 22 21 20 19 18 17 16 15
Topic 2 (proportion: 0.04): 43 49 44 45 47 40 46 48 41 42
Topic 3 (proportion: 0.13): 14 18 10 15 16 12 17 19 11 13
Topic 4 (proportion: 0.07): 19 16 10 15 11 17 12 13 18 14
Topic 5 (proportion: 0.01): 23 29 28 20 21 25 26 24 27 22
Topic 6 (proportion: 0.01): 31 38 35 39 30 33 34 37 32 36
Topic 7 (proportion: 0.19): 35 31 39 30 33 38 32 34 36 37
Topic 8 (proportion: 0.16): 48 42 46 49 45 47 41 44 40 43
Topic 9 (proportion: 0.19): 21 29 28 23 20 24 26 27 25 22
```

Here HDP find 7 large topics (> 1%) and those can map to the hidden topics we generated before.


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
