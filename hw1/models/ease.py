import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from models.common import recommend_topk


class EASE:
    def __init__(self, regularization=500.0):
        self.regularization = regularization

    def fit(self, interactions):
        x = sparse.csr_matrix(interactions, dtype=np.float64, copy=True)
        x.sum_duplicates()
        x.eliminate_zeros()
        x.data[:] = 1
        self.interactions_ = x
        self.popularity_ = np.asarray(x.sum(axis=0)).ravel()
        n_items = x.shape[1]
        gram = (x.T @ x).toarray(order="F")
        gram.flat[::n_items + 1] += self.regularization
        factor, lower = cho_factor(
            gram, lower=True, overwrite_a=True, check_finite=False,
        )
        inverse = cho_solve(
            (factor, lower), np.eye(n_items, order="F"),
            overwrite_b=True, check_finite=False,
        )
        diagonal = inverse.diagonal().copy()
        inverse /= -diagonal[None, :]
        np.fill_diagonal(inverse, 0)
        self.weights_ = inverse.astype(np.float32)
        return self

    def score(self, user_ids):
        return np.asarray(self.interactions_[user_ids] @ self.weights_)

    def recommend(self, user_ids, seen, k=10):
        return recommend_topk(self, user_ids, seen, k)
