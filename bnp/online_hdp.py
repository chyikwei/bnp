"""Online HDP with variational inference

This implementation is modified from Chong Wang's online-hdp code
(https://github.com/blei-lab/online-hdp)

Also, some code are from scikit-learn's online_lda implementation and the code structure is
the same.
(https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/online_lda.py)
"""

# Author: Chyi-Kwei Yau
# Original implementation: Chong Wang


import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import (check_random_state, check_array,
                           gen_batches, gen_even_slices, _get_n_jobs)
from sklearn.utils.validation import check_non_negative
from sklearn.exceptions import NotFittedError
from sklearn.utils.extmath import logsumexp
from sklearn.externals.joblib import Parallel, delayed

from .utils.expectation import (log_dirichlet_expectation,
                                log_stick_expectation,
                                stick_expectation)

EPS = np.finfo(np.float).eps


def mean_change(arr1, arr2):
    return np.abs(arr1 - arr2).mean()


def _update_local_variational_parameters(X, elog_beta, elog_stick, n_doc_truncate,
                                         alpha, max_iters, mean_change_tol,
                                         cal_sstats, cal_doc_distr):
    """Update local variational parameter

    This is step 3~10 in reference [1]

    Paramters
    ---------
    TODO

    Returns
    -------
    (doc_topic_distr, suff_stats) :
        `doc_topic_distr` is unnormalized topic distribution for each document.
        In the literature, this is `gamma`. we can calculate `E[log(theta)]`
        from it.
        `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, this will be None.
    """
    is_sparse_x = sp.issparse(X)
    n_samples, n_features = X.shape
    n_topic_truncate = elog_beta.shape[0]

    if cal_sstats:
        suff_stats = {
            # ss_lambda, shape = (K, n_features)
            'lambda': np.zeros(elog_beta.shape),
            # ss_v_stick, shape = (K,)
            'v_stick': np.zeros(elog_beta.shape[0])
        }
    else:
        suff_stats = None

    if cal_doc_distr:
        doc_topic_distr = np.empty((n_samples, n_topic_truncate))
    else:
        doc_topic_distr = None

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    for idx_d in xrange(n_samples):
        # get word_id and count in each document
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        # elog_beta_d, shape = (K, N_unique)
        elog_beta_d = elog_beta[:, ids]
        elog_beta_d_weighted = elog_beta_d * cnts
        # exp_elog_beta_sum, shape = (K,)
        exp_elog_beta_sum = np.exp(np.sum(elog_beta_d_weighted, axis=1)) + EPS
        # normalize
        exp_elog_beta_sum /= np.sum(exp_elog_beta_sum)
        # initialize `zeta` (step 5. in ref [1]), shape = (T, K)
        zeta_d = np.repeat(exp_elog_beta_sum[np.newaxis, :], n_doc_truncate, axis=0)

        # initialize `phi` step 6 in ref [1]), shape = (N, T)
        log_phi_d = np.dot(zeta_d, elog_beta_d).T
        # norm_phi, shape = (N, 1)
        norm_phi = logsumexp(log_phi_d, axis=1) + EPS
        phi_d = np.exp(log_phi_d - norm_phi[:, np.newaxis])

        # initialized gamma_d, shape = (2, T-1)
        gamma_d = np.empty((2, n_doc_truncate-1))

        # update variables
        for n_iter in xrange(0, max_iters):
            last_phi_d = phi_d

            # phi_all, shape = (N, T)
            phi_all = phi_d * cnts[:, np.newaxis]
            # update gamma_d, zeta_d (step 8. in ref [1])
            # gamma_d
            gamma_d[0] = 1.0 + np.sum(phi_all[:, :n_doc_truncate-1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:, 1:], 0))
            gamma_d[1] = alpha + np.flipud(np.cumsum(phi_cum))
            # E[log(pi_{d})], shape = (T,)
            elog_local_stick = log_stick_expectation(gamma_d)
            # normalized zeta_d
            log_zeta_d = np.dot(phi_d.T, elog_beta_d_weighted.T) + elog_stick
            norm_zeta = logsumexp(log_zeta_d, axis=1) + EPS
            zeta_d = np.exp(log_zeta_d - norm_zeta[:, np.newaxis])
            # phi_d (step 9. in ref [1])
            log_phi_d = np.dot(zeta_d, elog_beta_d).T + elog_local_stick
            norm_phi = logsumexp(log_phi_d, axis=1) + EPS
            phi_d = np.exp(log_phi_d - norm_phi[:, np.newaxis])

            # check convergence
            m_change = mean_change(last_phi_d, phi_d)
            if m_change < mean_change_tol:
                #print "converged iter: %d" % (n_iter)
                break
            #else:
                # DEBUG. delete later
                #print "iter %d mean_change: %.5f" % (n_iter, m_change)

        # update doc topic distribution
        if cal_doc_distr:
            # doc_topics, shape = (K,)
            doc_topic_distr[idx_d, :] = np.dot(np.sum(phi_all, axis=0), zeta_d)

        # update sstats
        if cal_sstats:
            suff_stats['v_stick'] += np.sum(zeta_d, axis=0)
            suff_stats['lambda'][:, ids] += np.dot(zeta_d.T, phi_all.T)
    return doc_topic_distr, suff_stats


class HierarchicalDirichletProcess(BaseEstimator, TransformerMixin):
    """Hierarchical Dirichlet Process with Stochastic Variational Inference
    
    HDP is the nonparametric version of LDA. The algorithm determines
    the number of topics based on data it fits instead of using a fixed
    number.

    Note: This implementation is described in Fig. 9 in [1]. A lot 
          of greek letter are used as variable names. Check Fig. 8
          and Fig. 9 in [1] to know more about the alogrithm and each
          variable. Some of notation are different from original
          implementation in [3].


    Parameters
    ----------
    n_topic_truncate : int, optional (default=150)
        topics level truncation. In literature, it is `K`
    
    n_doc_truncate: int, optional (default=20)
        document level truncation. In literature, it is `T`

    omega: float, optional (default=1.0)
        topic level concentration

    alpha: float, optional (default=1.0)
        document level concentration

    eta: float, optional (default=1e-2)
        the topic Dirichlet prior
    
    learning_method : 'batch' | 'online', default='online'
        Method used to update latent variable. Only used in `fit` method.

    kappa : float, optional (default=0.5)
        This is forgetting rate in the online learning
        method. The value should be set between (0.5, 1.0] to guarantee
        asymptotic convergence. When the value is 0.0 and batch_size is
        ``n_samples``, the update method is same as batch learning.

    tau : float, optional (default=10.)
        A (positive) parameter that downweights early iterations in online
        learning.  It should be greater than 1.0.
    
    scale: float, optional (default=1.0)
        scale for learning rate.

    max_iter : int, optional (default=100)
        The maximum number of iterations.

    total_samples: int, optional (default=1e6))
        number of document. Only used in the `partial_fit` method.

    batch_size : int, optional (default=256)
        MiniBatch size. Number of documents to use in each EM iteration.
        Only used in online learning.

    evaluate_every : int, optional (default=0)
        How often to evaluate perplexity. Only used in `fit` method.
        set it to 0 or negative number to not evalute perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.

    perp_tol : float, optional (default=1e-1)
        Perplexity tolerance in batch learning. Only used when
        ``evaluate_every`` is greater than 0.

    mean_change_tol : float, optional (default=1e-3)
        Stopping tolerance for updating document topic distribution in E-step.

    max_doc_update_iter : int, optional (default=100)
        Max number of iterations for updating document topic distribution in
        the E-step.

    n_jobs : int, optional (default=1)
        The number of jobs to use in the E-step. If -1, all CPUs are used. For
        ``n_jobs`` below -1, (n_cpus + 1 + n_jobs) are used. Only used in 
        batch learning now since it might not help when you do online learning
        with small batch size.

    verbose : int, optional (default=0)
        Verbosity level.

    random_state : int or RandomState instance or None, optional (default=None)
        Pseudo-random number generator seed control.

    Attributes
    ----------
    lambda_: array, [n_topic_truncate, n_features]
        Golbal Variational parameter for topic word distribution (`beta`)
        q(beta) ~ Direchlet(lambda)
    
    v_stick_: array, [2, n_topic_truncate-1]
        Golbal Variational parameter for topic stick length (`v`)
        q(v_k) ~ Beta(a_[0, k], a_1[1, k])

    n_min_batch_iter_ : int
        Number of iterations of the mini-batch EM step.

    n_iter_ : int
        Number of passes over the dataset.

    References
    ----------
    [1] "Stochastic Variational Inference", Matthew D. Hoffman, David M. Blei,
        Chong Wang, John Paisley, 2013
    
    [2] "Online Variational Inference for the Hierarchical Dirichlet Process", 
        Chong Wang, John Paisley, David M. Blei, 2011

    [3] Chong Wang's online-hdp code. Link:
        https://github.com/blei-lab/online-hdp

    """

    def __init__(self, n_topic_truncate=150, n_doc_truncate=20, omega=1.0,
                 alpha=1.0, eta=1e-2, learning_method='online', kappa=0.5, 
                 tau=10., scale=1.0, max_iter=100, total_samples=1e6,
                 batch_size=256, evaluate_every=0, perp_tol=1e-1, 
                 mean_change_tol=1e-3, max_doc_update_iter=100,
                 n_jobs=1, verbose=0, random_state=None):
        self.n_topic_truncate = int(n_topic_truncate)
        self.n_doc_truncate = int(n_doc_truncate)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.learning_method = learning_method
        self.kappa = float(kappa)
        self.tau = float(tau)
        self.scale = float(scale)
        self.max_iter = int(max_iter)
        self.total_samples = int(total_samples)
        self.batch_size = int(batch_size)
        self.evaluate_every = int(evaluate_every)
        self.perp_tol = float(perp_tol)
        self.mean_change_tol = float(mean_change_tol)
        self.max_doc_update_iter = int(max_doc_update_iter)
        self.n_jobs = int(n_jobs)
        self.verbose = int(verbose)
        self.random_state = random_state

    def _check_params(self):
        """Check model parameters."""
        pos_int_params = [
            'n_topic_truncate',
            'n_doc_truncate',
            'total_samples',
            'max_iter'
        ]

        for param in pos_int_params:
            val = getattr(self, param)
            if val <= 0:
                raise ValueError("Invalid '%s' parameter: %r", param, val)

        if self.learning_method not in ("batch", "online"):
            raise ValueError("Invalid 'learning_method' parameter: %r"
                             % self.learning_method)

    def _check_non_neg_array(self, X, whom):
        """check X format
        check X format and make sure no negative value in X.
        Parameters
        ----------
        X :  array-like or sparse matrix
        """
        X = check_array(X, accept_sparse='csr')
        check_non_negative(X, whom)
        return X

    def _init_global_latent_vars(self, n_docs, n_features):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)

        # initialize global variational variables
        # follow reference [3]
        self.lambda_ = self.random_state_.gamma(
            1.0, 1.0, (self.n_topic_truncate, n_features)) * \
                (n_docs * 100 / (self.n_topic_truncate * n_features))
        self.elog_beta_ = log_dirichlet_expectation(self.lambda_)

        # Beta distribution for stick break process
        # Note: use uniform distribution here based on [3]
        # TODO: test Beta(1., omega) later
        self.v_stick_ = np.array([np.ones(self.n_topic_truncate-1),
                                  np.arange(self.n_topic_truncate-1, 0, -1)])
        self.elog_stick_ = log_stick_expectation(self.v_stick_)

    def _init_min_batch_parameters(self):
        self.n_min_batch_iter_ = 1

    def _get_step_size(self):
        rhot = self.scale * np.power(self.tau + self.n_min_batch_iter_, -self.kappa)
        if rhot < 0.0: 
            rhot = 0.0
            # DEBUG
            #print 'rhot become 0.0 at n_min_batch_iter_ = %d' % self.n_min_batch_iter_
        self.n_min_batch_iter_ += 1
        return rhot

    def _e_step(self, X, cal_sstats, cal_doc_distr, parallel=None):

        if parallel:
            n_jobs = parallel.n_jobs
            results = parallel(delayed(
                _update_local_variational_parameters)(X[idx_slice, :],
                                                      self.elog_beta_,
                                                      self.elog_stick_,
                                                      self.n_doc_truncate,
                                                      self.alpha,
                                                      self.max_doc_update_iter,
                                                      self.mean_change_tol,
                                                      cal_sstats,
                                                      cal_doc_distr)
                for idx_slice in gen_even_slices(X.shape[0], n_jobs))
            doc_topics, sstats_list = zip(*results)
        
            doc_topic_distr = np.vstack(doc_topics) if cal_doc_distr else None
            sstats = None
            if cal_sstats:
                lambda_sstats = np.zeros(self.lambda_.shape)
                v_stick_sstats = np.zeros((self.n_topic_truncate, ))
                for sstats in sstats_list:
                    lambda_sstats += sstats['lambda']
                    v_stick_sstats += sstats['v_stick']
                sstats = {
                    'lambda': lambda_sstats,
                    'v_stick': v_stick_sstats,
                }
        else:
            doc_topic_distr, sstats = \
                _update_local_variational_parameters(X,
                                                     self.elog_beta_,
                                                     self.elog_stick_,
                                                     self.n_doc_truncate,
                                                     self.alpha,
                                                     self.max_doc_update_iter,
                                                     self.mean_change_tol,
                                                     cal_sstats,
                                                     cal_doc_distr)
        return (doc_topic_distr, sstats)


    def _m_step(self, sstats, n_samples, online_update=False):
        n_topics = self.n_topic_truncate

        if online_update:
            doc_ratio = float(self.total_samples) / n_samples
            weight = self._get_step_size()
            # update lambda
            sstats_lambda = sstats['lambda']
            sstats_lambda *= doc_ratio
            sstats_lambda += self.eta
            sstats_lambda *= weight
            self.lambda_ *= (1. - weight)
            self.lambda_ += sstats_lambda

            # update v_stick
            sstats_v_stick = sstats['v_stick']
            sstats_v_stick *= doc_ratio

            v_stick_new = np.zeros((2, n_topics - 1))
            v_stick_new[0] += 1.
            v_stick_new[0] += sstats_v_stick[:n_topics-1]
            v_stick_new[1] += self.omega
            # flip -> cumsum -> flip back
            sum_from_end = np.flipud(np.cumsum(np.flipud(sstats_v_stick[1:])))
            v_stick_new[1] += sum_from_end
            v_stick_new *= weight
            self.v_stick_ *= (1. - weight)
            self.v_stick_ += v_stick_new
        else:
            # batch update
            # lambda
            self.lambda_ = self.eta + sstats['lambda']

            # stick
            self.v_stick_[0] = 1. + sstats['v_stick'][:n_topics-1]
            # flip -> cumsum -> flip back
            self.v_stick_[1] = self.omega + \
                np.flipud(np.cumsum(np.flipud(sstats['v_stick'][1:])))

        self.elog_beta_ = log_dirichlet_expectation(self.lambda_)
        self.elog_stick_ = log_stick_expectation(self.v_stick_)

    def partial_fit(self, X, y=None):
        """Online VB with Mini-Batch update.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.
        Returns
        -------
        self
        """
        self._check_params()
        X = self._check_non_neg_array(
            X, "HierarchicalDirichletProcess.partial_fit")
        n_samples, n_features = X.shape
        batch_size = self.batch_size

        if not hasattr(self, "lambda_"):
            self._init_global_latent_vars(*X.shape)
            self._init_min_batch_parameters()

        if n_features != self.lambda_.shape[1]:
            raise ValueError(
                "The provided data has %d dimensions while "
                "the model was trained with feature size %d." %
                (n_features, self.lambda_.shape[1]))

        for idx_slice in gen_batches(n_samples, batch_size):
            X_slice = X[idx_slice, :]
            _, sstats = self._e_step(X_slice,
                                     cal_sstats=True,
                                     cal_doc_distr=False,
                                     parallel=None)
            self._m_step(sstats, n_samples=X_slice.shape[0], online_update=True)
        return self

    def fit(self, X, y=None):
        """Learn model for the data X

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.

        Returns
        -------
        self

        """
        self._check_params()
        X = self._check_non_neg_array(
            X, "HierarchicalDirichletProcess.fit")
        self._init_global_latent_vars(*X.shape)

        n_jobs = _get_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for i in xrange(self.max_iter):
                # batch update
                _, sstats = self._e_step(X,
                                         cal_sstats=True,
                                         cal_doc_distr=False,
                                         parallel=parallel)
                self._m_step(sstats, n_samples=X.shape[0], online_update=False)
                # TODO: check perplexity

        return self

    def transform(self, X):
        """Transform data X according to the fitted model.
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.


        Returns
        -------
        doc_topic_distr : shape=(n_samples, n_topics)
            Unnormalized document topic distribution for X.

        """
        if not hasattr(self, "lambda_"):
            raise NotFittedError("no 'lambda_' attribute in model."
                                 " Please fit model first.")

        X = self._check_non_neg_array(
            X, "HierarchicalDirichletProcess.transform")

        n_jobs = _get_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            doc_topic_distr, _ = self._e_step(X,
                                              cal_sstats=False,
                                              cal_doc_distr=True,
                                              parallel=parallel)
        return doc_topic_distr

    def topic_distribution(self):
        """Topic distribution from stick-break process

        Returns
        -------
        topic_distr : shape=(n_topics,)
            topic distribution from stick-breaking process. (sum to 1.)
        """
        if not hasattr(self, "lambda_"):
            raise NotFittedError("no 'lambda_' attribute in model."
                                 " Please fit model first.")

        topic_distr = stick_expectation(self.v_stick_)
        return topic_distr

    def score(self, X, y=None):
        """Calculate approximate log-likelihood as score.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.

        Returns
        -------
        score : float
            Use approximate bound as score.
        """
        pass
