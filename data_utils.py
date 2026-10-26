import ast
import gzip
import html
import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy import sparse


BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
REVIEWS_URL = BASE_URL + "reviews_Video_Games_5.json.gz"
METADATA_URL = BASE_URL + "meta_Video_Games.json.gz"


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "recsys-homework/1.0"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    temporary.replace(destination)
    return destination


def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = ast.literal_eval(line)
            yield record


def load_reviews(path):
    fields = ["reviewerID", "asin", "overall", "unixReviewTime"]
    return pd.DataFrame.from_records(
        ({key: row.get(key) for key in fields} for row in records(path)),
        columns=fields,
    ).rename(columns={"reviewerID": "user", "asin": "item",
                      "overall": "rating", "unixReviewTime": "timestamp"})


def clean_reviews(raw):
    df = raw.copy()
    rows = [{"stage": "raw", "interactions": len(df)}]
    for col in ["rating", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid_ids = df["user"].map(lambda x: isinstance(x, str) and bool(x.strip()))
    valid_ids &= df["item"].map(lambda x: isinstance(x, str) and bool(x.strip()))
    valid = (valid_ids & df["rating"].isin([1, 2, 3, 4, 5])
             & np.isfinite(df["timestamp"]) & (df["timestamp"] > 0)
             & (df["timestamp"] % 1 == 0))
    df = df.loc[valid].copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    rows.append({"stage": "valid fields", "interactions": len(df)})
    df = df.drop_duplicates()
    rows.append({"stage": "no exact duplicates", "interactions": len(df)})
    df = df.sort_values("timestamp", kind="stable").drop_duplicates(
        ["user", "item"], keep="first")
    rows.append({"stage": "first user-item observation", "interactions": len(df)})
    return df.reset_index(drop=True), pd.DataFrame(rows)


def describe_interactions(df):
    n_users = df.user.nunique()
    n_items = df.item.nunique()
    return {
        "interactions": len(df),
        "users": n_users,
        "items": n_items,
        "sparsity": 1 - len(df) / (n_users * n_items),
    }


def load_metadata(path, item_ids):
    wanted = set(item_ids)
    rows = []
    for record in records(path):
        if record.get("asin") not in wanted:
            continue
        rows.append({
            "item": record["asin"],
            "title": html.unescape(str(record.get("title") or "")).strip(),
            "brand": html.unescape(str(record.get("brand") or "")).strip(),
            "categories": record.get("categories") or [],
        })
    return pd.DataFrame(rows, columns=["item", "title", "brand", "categories"]).drop_duplicates("item")


def temporal_split(df, train_fraction=0.8, validation_fraction=0.1):
    times = np.sort(df["timestamp"].to_numpy())
    t1 = int(times[int(len(times) * train_fraction)])
    t2 = int(times[int(len(times) * (train_fraction + validation_fraction))])
    train = df[df.timestamp < t1].copy()
    validation = df[(df.timestamp >= t1) & (df.timestamp < t2)].copy()
    test = df[df.timestamp >= t2].copy()
    return train, validation, test, (t1, t2)


def make_mappings(train):
    users = {value: idx for idx, value in enumerate(sorted(train.user.unique()))}
    items = {value: idx for idx, value in enumerate(sorted(train.item.unique()))}
    return users, items


def interaction_matrix(df, users, items, min_rating=None):
    selected = df if min_rating is None else df[df.rating >= min_rating]
    u, i = selected.user.map(users), selected.item.map(items)
    valid = u.notna() & i.notna()
    matrix = sparse.csr_matrix(
        (np.ones(int(valid.sum()), dtype=np.float32),
         (u[valid].to_numpy(dtype=np.int64), i[valid].to_numpy(dtype=np.int64))),
        shape=(len(users), len(items)),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1
    return matrix


def evaluation_targets(holdout, users, items, seen, min_rating=4):
    known_user = holdout.user.isin(users)
    known_item = holdout.item.isin(items)
    relevant = holdout[known_user & known_item & (holdout.rating >= min_rating)]
    targets = {}
    excluded_seen = 0
    for row in relevant.itertuples(index=False):
        user, item = users[row.user], items[row.item]
        if seen[user, item] != 0:
            excluded_seen += 1
            continue
        targets.setdefault(user, set()).add(item)
    report = {
        "holdout_rows": len(holdout),
        "cold_user_rows": int((~known_user).sum()),
        "cold_item_rows": int((~known_item).sum()),
        "warm_rows": int((known_user & known_item).sum()),
        "excluded_seen_positive_rows": excluded_seen,
        "evaluated_users": len(targets),
        "relevant_pairs": sum(map(len, targets.values())),
    }
    return targets, report


def rating_time_targets(train, users, items, cutoff, half_life, min_rating=4):
    positive = train[train.rating >= min_rating]
    age_days = (cutoff - positive.timestamp.to_numpy()) / 86400
    strength = (positive.rating.to_numpy() - 3) / 2
    values = strength * np.exp2(-age_days / half_life)
    return sparse.csr_matrix(
        (values, (positive.user.map(users).to_numpy(dtype=np.int64),
                  positive.item.map(items).to_numpy(dtype=np.int64))),
        shape=(len(users), len(items)),
    )
