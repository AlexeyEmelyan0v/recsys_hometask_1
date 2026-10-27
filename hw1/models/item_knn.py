import numpy as np
from scipy import sparse


class ItemKNN:

    def __init__(self, n_neighbors=100, shrinkage=10.0, block_size=256):
        if n_neighbors < 1 or block_size < 1 or shrinkage < 0:
            raise ValueError("Invalid n_neighbors, block_size or shrinkage")
        self.n_neighbors = int(n_neighbors)
        self.shrinkage = float(shrinkage)
        self.block_size = int(block_size)

    def fit(self, interactions):
        x = sparse.csr_matrix(interactions, dtype=np.float32, copy=True)
        x.sum_duplicates()
        x.eliminate_zeros()
        if not np.isfinite(x.data).all() or (x.data < 0).any():
            raise ValueError("Expected finite nonnegative interactions")
        if min(x.shape) == 0 or x.nnz == 0:
            raise ValueError("Training matrix must contain positive feedback")
        x.data[:] = 1.0
        self.interactions_ = x
        self.popularity_ = np.asarray(x.sum(axis=0)).ravel()
        norms = np.sqrt(self.popularity_)
        n_items = x.shape[1]
        xt = x.T.tocsr()
        indices, values = [], []
        indptr = [0]
        for start in range(0, n_items, self.block_size):
            counts = (xt[start:start + self.block_size] @ x).tocsr()
            for offset in range(counts.shape[0]):
                item = start + offset
                left, right = counts.indptr[offset:offset + 2]
                ids = counts.indices[left:right]
                common = counts.data[left:right]
                keep = ids != item
                ids, common = ids[keep], common[keep]

                sims = common / (norms[item] * norms[ids])
                if self.shrinkage:
                    sims *= common / (common + self.shrinkage)

                order = np.lexsort((ids, -sims))[:self.n_neighbors]
                indices.extend(ids[order])
                values.extend(sims[order])
                indptr.append(len(indices))
        self.similarity_ = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32),
             np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)),
            shape=(n_items, n_items),
        )
        self.similarity_.sort_indices()
        return self

    def score(self, user_ids):
        return (self.interactions_[user_ids] @ self.similarity_).toarray()

    def recommend(self, user_ids, seen, k=10):
        if k < 1:
            raise ValueError("k must be positive")
        user_ids = np.asarray(user_ids, dtype=np.int64)
        if seen.shape != self.interactions_.shape:
            raise ValueError("seen and training matrices must have equal shapes")
        if ((user_ids < 0) | (user_ids >= seen.shape[0])).any():
            raise ValueError("Unknown user index")
        seen = sparse.csr_matrix(seen)
        scores = self.score(user_ids)
        item_ids = np.arange(scores.shape[1])
        result = np.full((len(user_ids), k), -1, dtype=np.int64)
        for row, user in enumerate(user_ids):
            left, right = seen.indptr[user:user + 2]
            scores[row, seen.indices[left:right]] = -np.inf
            candidates = item_ids[np.isfinite(scores[row])]
            order = np.lexsort((candidates, -self.popularity_[candidates],
                                -scores[row, candidates]))
            selected = candidates[order[:k]]
            result[row, :len(selected)] = selected
        return result
