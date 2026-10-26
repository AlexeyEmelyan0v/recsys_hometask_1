import numpy as np


def ranking_metrics(recommendations, targets, k=10):
    recall, ndcg = [], []
    discounts = 1 / np.log2(np.arange(2, k + 2))
    for user, relevant in targets.items():
        items = np.asarray(recommendations[user])[:k]
        hits = np.array([item >= 0 and item in relevant for item in items], dtype=float)
        recall.append(hits.sum() / len(relevant))
        ideal = discounts[:min(k, len(relevant))].sum()
        ndcg.append(float(hits @ discounts[:len(hits)]) / ideal)
    return {f"NDCG@{k}": float(np.mean(ndcg)),
            f"Recall@{k}": float(np.mean(recall)), "users": len(targets)}


def evaluate(model, targets, seen, k=10, batch_size=128):
    users = np.array(sorted(targets), dtype=np.int64)
    recommendations = {}
    for left in range(0, len(users), batch_size):
        batch = users[left:left + batch_size]
        predicted = model.recommend(batch, seen=seen, k=k)
        recommendations.update(zip(batch.tolist(), predicted))
    return ranking_metrics(recommendations, targets, k=k)
