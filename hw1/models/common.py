import numpy as np


def recommend_topk(model, user_ids, seen, k=10):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = model.score(user_ids)
    item_ids = np.arange(scores.shape[1])
    result = np.full((len(user_ids), k), -1, dtype=np.int64)
    for row, user in enumerate(user_ids):
        left, right = seen.indptr[user:user + 2]
        scores[row, seen.indices[left:right]] = -np.inf
        candidates = item_ids[np.isfinite(scores[row])]
        order = np.lexsort((candidates, -model.popularity_[candidates],
                            -scores[row, candidates]))
        selected = candidates[order[:k]]
        result[row, :len(selected)] = selected
    return result
