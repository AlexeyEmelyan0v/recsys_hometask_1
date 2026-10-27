import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from models.common import recommend_topk


class _FM(nn.Module):
    def __init__(self, n_features, factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(n_features, 1, padding_idx=0)
        self.factors = nn.Embedding(n_features, factors, padding_idx=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.01)
        with torch.no_grad():
            self.factors.weight[0].zero_()

    def forward(self, feature_ids, feature_values):
        vectors = self.factors(feature_ids) * feature_values.unsqueeze(-1)
        linear = (self.linear(feature_ids).squeeze(-1) * feature_values).sum(dim=1)
        interactions = 0.5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1)).sum(dim=1)
        return self.bias + linear + interactions


class FM:
    def __init__(self, factors=32, learning_rate=0.003, regularization=1e-5,
                 epochs=10, batch_size=2048, negatives=2, seed=42,
                 use_metadata=True, device=None):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.negatives = negatives
        self.seed = seed
        self.use_metadata = use_metadata
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_features(self, items, metadata):
        categories = metadata.set_index("item").categories.to_dict()
        item_categories = []
        for item in items:
            values = set()
            if self.use_metadata:
                for path in categories.get(item, []):
                    for depth in range(2, len(path) + 1):
                        values.add(" > ".join(path[:depth]))
                if not values:
                    values.add("unknown")
            item_categories.append(sorted(values))
        vocabulary = sorted({value for values in item_categories for value in values})
        offset = 1 + self.n_users_ + self.n_items_
        category_ids = {value: offset + i for i, value in enumerate(vocabulary)}
        width = 1 + max(map(len, item_categories))
        ids = np.zeros((self.n_items_, width), dtype=np.int64)
        weights = np.zeros((self.n_items_, width), dtype=np.float32)
        for item, values in zip(items, item_categories):
            index = items[item]
            ids[index, 0] = 1 + self.n_users_ + index
            weights[index, 0] = 1
            if values:
                ids[index, 1:len(values) + 1] = [category_ids[value] for value in values]
                weights[index, 1:len(values) + 1] = 1 / np.sqrt(len(values))
        self.item_features_ = torch.tensor(ids, device=self.device)
        self.item_values_ = torch.tensor(weights, device=self.device)
        return offset + len(vocabulary)

    def _predict(self, users, items):
        users = torch.as_tensor(users, dtype=torch.long, device=self.device)
        items = torch.as_tensor(items, dtype=torch.long, device=self.device)
        ids = torch.cat([(users + 1).unsqueeze(1), self.item_features_[items]], dim=1)
        values = torch.cat([torch.ones((len(users), 1), device=self.device),
                            self.item_values_[items]], dim=1)
        return self.model_(ids, values)

    def fit(self, train, users, items, metadata, seen, min_rating=4):
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(self.seed)
        self.n_users_, self.n_items_ = len(users), len(items)
        n_features = self._prepare_features(items, metadata)
        self.model_ = _FM(n_features, self.factors).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate,
                                     weight_decay=self.regularization)
        loss_function = nn.BCEWithLogitsLoss()
        train_users = train.user.map(users).to_numpy(dtype=np.int64)
        train_items = train.item.map(items).to_numpy(dtype=np.int64)
        labels = (train.rating.to_numpy() >= min_rating).astype(np.float32)
        self.popularity_ = np.bincount(train_items[labels == 1], minlength=self.n_items_)
        can_sample = seen.getnnz(axis=1) < self.n_items_
        positive_users = train_users[(labels == 1) & can_sample[train_users]]
        negative_users = np.repeat(positive_users, self.negatives)
        self.model_.train()
        for _ in range(self.epochs):
            negative_items = rng.integers(self.n_items_, size=len(negative_users))
            known = np.asarray(seen[negative_users, negative_items]).ravel() != 0
            while known.any():
                negative_items[known] = rng.integers(self.n_items_, size=int(known.sum()))
                known = np.asarray(seen[negative_users, negative_items]).ravel() != 0
            epoch_users = np.concatenate([train_users, negative_users])
            epoch_items = np.concatenate([train_items, negative_items])
            epoch_labels = np.concatenate([labels, np.zeros(len(negative_users), dtype=np.float32)])
            order = rng.permutation(len(epoch_users))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                prediction = self._predict(epoch_users[batch], epoch_items[batch])
                target = torch.as_tensor(epoch_labels[batch], device=self.device)
                loss = loss_function(prediction, target)
                loss.backward()
                optimizer.step()
        self.model_.eval()
        return self

    def score(self, user_ids):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        users = np.repeat(user_ids, self.n_items_)
        items = np.tile(np.arange(self.n_items_), len(user_ids))
        scores = np.empty(len(users), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(users), 16384):
                end = start + 16384
                scores[start:end] = self._predict(users[start:end], items[start:end]).cpu().numpy()
        return scores.reshape(len(user_ids), self.n_items_)

    def recommend(self, user_ids, seen, k=10):
        return recommend_topk(self, user_ids, seen, k)
