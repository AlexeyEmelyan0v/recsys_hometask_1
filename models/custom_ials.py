import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from models.ials import IALS


class CustomIALS(IALS):
    def _update(self, targets, fixed):
        gram = fixed.T @ fixed
        gram += self.regularization * np.eye(self.factors)
        updated = np.zeros((targets.shape[0], self.factors))
        for row in range(targets.shape[0]):
            left, right = targets.indptr[row:row + 2]
            neighbors = fixed[targets.indices[left:right]]
            values = targets.data[left:right]
            if len(neighbors) == 0:
                continue
            a = gram + self.alpha * (neighbors.T @ neighbors)
            b = (1 + self.alpha) * (neighbors.T @ values)
            updated[row] = cho_solve(
                cho_factor(a, lower=True, check_finite=False), b,
                check_finite=False,
            )
        return updated

    def fit(self, targets):
        x = sparse.csr_matrix(targets, dtype=np.float64, copy=True)
        x.sum_duplicates()
        x.eliminate_zeros()
        self.popularity_ = np.asarray((x > 0).sum(axis=0)).ravel()
        rng = np.random.default_rng(self.seed)
        self.item_factors_ = rng.normal(0, 0.01, (x.shape[1], self.factors))
        xt = x.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors_ = self._update(x, self.item_factors_)
            self.item_factors_ = self._update(xt, self.user_factors_)
        return self
