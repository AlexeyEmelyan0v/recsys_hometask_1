import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from models.common import recommend_topk


class IALS:
    def __init__(self, factors=32, alpha=40.0, regularization=0.1,
                 iterations=10, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def _update(self, interactions, fixed):
        gram = fixed.T @ fixed
        gram += self.regularization * np.eye(self.factors)
        updated = np.zeros((interactions.shape[0], self.factors))
        for row in range(interactions.shape[0]):
            left, right = interactions.indptr[row:row + 2]
            neighbors = fixed[interactions.indices[left:right]]
            if len(neighbors) == 0:
                continue
            a = gram + self.alpha * (neighbors.T @ neighbors)
            b = (1 + self.alpha) * neighbors.sum(axis=0)
            updated[row] = cho_solve(
                cho_factor(a, lower=True, check_finite=False), b,
                check_finite=False,
            )
        return updated

    def fit(self, interactions):
        x = sparse.csr_matrix(interactions, dtype=np.float64, copy=True)
        x.sum_duplicates()
        x.eliminate_zeros()
        x.data[:] = 1
        self.popularity_ = np.asarray(x.sum(axis=0)).ravel()
        rng = np.random.default_rng(self.seed)
        self.item_factors_ = rng.normal(0, 0.01, (x.shape[1], self.factors))
        xt = x.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors_ = self._update(x, self.item_factors_)
            self.item_factors_ = self._update(xt, self.user_factors_)
        return self

    def score(self, user_ids):
        return self.user_factors_[user_ids] @ self.item_factors_.T

    def recommend(self, user_ids, seen, k=10):
        return recommend_topk(self, user_ids, seen, k)
