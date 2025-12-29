"""
rf_reranker.py
================

This module wraps the Random Forest reranker used in our lightGCN + RF
pipeline.  After the LightGCN model has produced a candidate list of
items for each user, the reranker computes a richer feature vector for
each ``(user, item)`` pair and learns to reorder the candidate list
based on click/engagement labels.  The resulting model typically
improves ranking metrics such as NDCG and Recall in our experiments.

Functions
---------
get_topN_candidates(model, users, cf_args)
    Uses a trained LightGCN model to produce a fixed‑size candidate list
    for each user.

build_behavior_features_ui(raw_df_train)
    Aggregate raw training data into per‑user–item behavioural features.

build_rerank_features_v2(candidates, model, users, raw_df_train, raw_df_all, ...)
    Core feature engineering routine.  Computes user/item embeddings,
    behavioural stats, demographics and a variety of interaction features
    including engagement scores, category matching and cosine similarity.
    Supports checkpointing for long jobs.

train_reranker_rf(model, train_users, train_ur, raw_df_train, raw_df_all, cf_args)
    Train a ``RandomForestClassifier`` to rerank LightGCN candidates.  It
    automatically saves the fitted model to ``./checkpoint/rf_reranker.pkl``.

rerank_candidates(model, rf, test_users, raw_df_train, raw_df_all, cf_args)
    Apply a fitted reranker to reorder the candidate list for a set of
    users.  Caches computed features for efficiency.

The implementation here is distilled from the original notebook to
facilitate reuse in scripts and services.
"""

import os
import pickle
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import joblib
import torch


def get_topN_candidates(model, users: List[int], cf_args: Dict) -> np.ndarray:
    """Return the top N candidate item indices for each user from a LightGCN model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LightGCN model supporting a ``rank`` method.
    users : list of int
        User indices for which to generate candidates.
    cf_args : dict
        Configuration dictionary with key ``rerank_topN`` specifying the
        number of candidates to return per user.

    Returns
    -------
    np.ndarray of shape (len(users), topN)
        Each row contains item indices sorted by their LightGCN score.
    """
    topN = cf_args.get("rerank_topN", 20)
    model.topk = topN
    # This loader generates batches of users for prediction.  Cf_valDataset
    # simply wraps the list of user ids.
    from torch.utils.data import DataLoader

    class Cf_valDataset(torch.utils.data.Dataset):
        def __init__(self, users):
            self.users = users
        def __len__(self):
            return len(self.users)
        def __getitem__(self, idx):
            return self.users[idx]

    loader = DataLoader(Cf_valDataset(users), batch_size=cf_args.get("val_batch_size", 64))
    return model.rank(loader)


def build_behavior_features_ui(raw_df_train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate user–item interactions into behavioural features.

    Returns a DataFrame with counts of each interaction type and average
    watch time per (user, item) pair.
    """
    return raw_df_train.groupby(["user_id", "item_id"]).agg(
        cnt_click=("click", "sum"),
        cnt_like=("like", "sum"),
        cnt_share=("share", "sum"),
        cnt_follow=("follow", "sum"),
        avg_watch=("watching_times", "mean"),
    ).reset_index()


def get_user_item_meta(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract distinct user and item metadata from the raw dataframe."""
    user_meta = raw_df[["user_id", "gender", "age"]].drop_duplicates()
    item_meta = raw_df[["item_id", "video_category"]].drop_duplicates()
    return user_meta, item_meta


def build_rerank_features_v2(
    candidates: np.ndarray,
    model,
    users: List[int],
    raw_df_train: pd.DataFrame,
    raw_df_all: pd.DataFrame,
    checkpoint_dir: str = "./checkpoint/rerank_features",
    batch_size: int = 1000,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Compute enhanced feature vectors for reranking.

    This function generates a large feature matrix where each row
    corresponds to a candidate pair (user, item).  To support very large
    datasets it checkpoints intermediate results to disk every ``batch_size``
    users.  The features combine embeddings from the LightGCN model,
    behavioural counts, demographics and engineered engagement metrics.

    Parameters
    ----------
    candidates : np.ndarray
        2‑D array with shape (num_users, topN) listing candidate item
        indices for each user.
    model : LightGCN
        Trained model used to obtain user and item embeddings.
    users : list of int
        List of user indices aligned with the rows of ``candidates``.
    raw_df_train : pd.DataFrame
        Training interactions used to compute statistics.
    raw_df_all : pd.DataFrame
        Full dataset with user/item metadata (gender, age, video_category).
    checkpoint_dir : str, optional
        Directory where partial feature batches are saved.  Defaults to
        ``./checkpoint/rerank_features``.
    batch_size : int, optional
        Number of users to process between checkpoints.  Defaults to 1000.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (num_users * topN, n_features).
    keys : list of (int, int)
        List of (user_id, item_id) pairs corresponding to each row of X.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "features_checkpoint.pkl")
    # Resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        X = checkpoint["X"]
        keys = checkpoint["keys"]
        start_idx = checkpoint["last_user_idx"] + 1
        print(f"[INFO] Loaded checkpoint with {len(X)} feature rows; resuming from user {start_idx}.")
    else:
        X = []
        keys = []
        start_idx = 0

    # Compute embeddings once
    U_emb, I_emb = model.forward()
    U_emb = U_emb.detach().cpu().numpy()
    I_emb = I_emb.detach().cpu().numpy()

    # Behaviour, metadata and stats
    beh = build_behavior_features_ui(raw_df_train)
    user_meta, item_meta = get_user_item_meta(raw_df_all)
    # User‑level statistics
    user_stats = raw_df_train.groupby("user_id").agg(
        user_activity=("item_id", "nunique"),
        user_avg_watch=("watching_times", "mean"),
        user_like_rate=("like", lambda x: x.sum() / (x.sum() + (x == 0).sum())),
        user_share_rate=("share", "mean"),
    ).reset_index()
    # Item‑level statistics
    item_stats = raw_df_train.groupby("item_id").agg(
        item_popularity=("user_id", "nunique"),
        item_avg_watch=("watching_times", "mean"),
        item_like_rate=("like", "mean"),
        item_share_rate=("share", "mean"),
    ).reset_index()
    # Encode categorical metadata via label encoding (simple)
    from sklearn.preprocessing import LabelEncoder
    enc_gender = LabelEncoder().fit(user_meta["gender"].astype(str).fillna("unknown"))
    enc_age = LabelEncoder().fit(user_meta["age"].astype(str).fillna("unknown"))
    enc_video_category = LabelEncoder().fit(item_meta["video_category"].astype(str).fillna("unknown"))
    user_meta = user_meta.copy()
    item_meta = item_meta.copy()
    user_meta["gender"] = enc_gender.transform(user_meta["gender"].astype(str).fillna("unknown"))
    user_meta["age"] = enc_age.transform(user_meta["age"].astype(str).fillna("unknown"))
    item_meta["video_category"] = enc_video_category.transform(item_meta["video_category"].astype(str).fillna("unknown"))
    user_stats["user_id"] = user_stats["user_id"].astype(int)
    item_stats["item_id"] = item_stats["item_id"].astype(int)
    # Build lookup dicts for quick feature access
    beh_dict = {(u, i): list(row) for u, i, *row in beh.itertuples(index=False)}
    user_stats_dict = {u: list(row) for u, *row in user_stats.itertuples(index=False)}
    item_stats_dict = {i: list(row) for i, *row in item_stats.itertuples(index=False)}
    # For each user starting at start_idx, build features for each candidate item
    for ui, u in enumerate(tqdm(users[start_idx:], desc="building rerank features", initial=start_idx, total=len(users))):
        actual_ui = start_idx + ui
        for rank, i in enumerate(candidates[actual_ui]):
            u_id = int(u)
            i_id = int(i)
            feat = []
            # (1) Raw embeddings
            feat.extend([float(x) for x in U_emb[u_id]])  # user embedding
            feat.extend([float(x) for x in I_emb[i_id]])  # item embedding
            # (2) LightGCN score (dot product)
            feat.append(float(np.dot(U_emb[u_id], I_emb[i_id])))
            # (3) Candidate rank position
            feat.append(float(rank))
            # (4) Behavioural counts
            beh_feats = beh_dict.get((u_id, i_id), [0, 0, 0, 0, 0])
            feat.extend([float(x) for x in beh_feats])
            # (5) User metadata (gender, age)
            um = user_meta[user_meta["user_id"] == u_id]
            if len(um) > 0:
                feat.extend([float(um.iloc[0]["gender"]), float(um.iloc[0]["age"])])
            else:
                feat.extend([0.0, 0.0])
            # (6) Item metadata (video_category)
            im = item_meta[item_meta["item_id"] == i_id]
            feat.append(float(im.iloc[0]["video_category"]) if len(im) > 0 else 0.0)
            # (7) User stats
            u_stats = user_stats_dict.get(u_id, [0, 0, 0, 0])
            feat.extend([float(x) for x in u_stats])
            # (8) Item stats
            i_stats = item_stats_dict.get(i_id, [0, 0, 0, 0])
            feat.extend([float(x) for x in i_stats])
            # (9) Engagement score (weighted sum of behavioural counts)
            engagement = (
                beh_feats[0] * 1.0  # click
                + beh_feats[1] * 2.0  # like
                + beh_feats[2] * 3.0  # share
                + beh_feats[3] * 5.0  # follow
                + beh_feats[4] * 0.1  # watch time
            )
            feat.append(float(engagement))
            # (10) Normalized watch time (relative to user average)
            u_avg_watch = user_stats_dict.get(u_id, [0, 0, 0, 0])[1]
            feat.append(float(beh_feats[4] / (u_avg_watch + 1e-6)))
            # (11) Category match (1 if user's most frequent category matches item)
            # Compute user's preferred category from raw_df_train on the fly
            user_history = raw_df_train[raw_df_train["user_id"] == u_id]
            if len(user_history) > 0 and "video_category" in user_history.columns:
                # mode() returns a series – take first element
                user_pref_cat = user_history["video_category"].mode().iloc[0]
            else:
                user_pref_cat = -1
            item_cat = im.iloc[0]["video_category"] if len(im) > 0 else -1
            feat.append(float(int(user_pref_cat == item_cat)))
            # (12) Cosine similarity between embeddings
            cosine_sim = np.dot(U_emb[u_id], I_emb[i_id]) / (
                np.linalg.norm(U_emb[u_id]) * np.linalg.norm(I_emb[i_id]) + 1e-6
            )
            feat.append(float(cosine_sim))
            # (13) Item popularity percentile
            all_pops = list(item_stats_dict.values())
            if len(all_pops) > 0:
                i_pop = item_stats_dict.get(i_id, [0, 0, 0, 0])[0]
                pop_percentile = sum(p[0] < i_pop for p in all_pops) / len(all_pops)
                feat.append(float(pop_percentile))
            else:
                feat.append(0.0)
            # Append to lists
            X.append(feat)
            keys.append((u_id, i_id))
        # Checkpoint every batch_size users processed
        if (actual_ui + 1) % batch_size == 0:
            checkpoint = {"X": X, "keys": keys, "last_user_idx": actual_ui}
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"[CHECKPOINT] Saved features at user {actual_ui + 1}.")
    # Return final arrays
    return np.array(X, dtype=np.float32), keys


def train_reranker_rf(
    model,
    train_users: List[int],
    train_ur: Dict[int, List[int]],
    raw_df_train: pd.DataFrame,
    raw_df_all: pd.DataFrame,
    cf_args: Dict,
) -> RandomForestClassifier:
    """Train a Random Forest to rerank LightGCN candidate lists.

    ``train_ur`` is a dict mapping each user to the list of items they
    interacted with in the training set.  These are used to generate
    binary labels for the reranking task (1 if the candidate is in the
    user's training interactions, 0 otherwise).
    """
    # Generate candidate item lists
    candidates = get_topN_candidates(model, train_users, cf_args)
    # Build features and keys
    X, keys = build_rerank_features_v2(
        candidates, model, train_users, raw_df_train, raw_df_all
    )
    # Generate labels (1 if candidate item appears in train_ur[u])
    y = np.array([1 if i in train_ur.get(u, []) else 0 for u, i in keys], dtype=np.int32)
    # Train classifier
    rf = RandomForestClassifier(
        n_estimators=cf_args.get("rerank_rf_estimators", 300),
        max_depth=cf_args.get("rerank_rf_max_depth", 14),
        random_state=cf_args.get("rerank_rf_random_state", 42),
        n_jobs=-1,
    )
    rf.fit(X, y)
    # Persist model
    rf_path = os.path.join("./checkpoint/", "rf_reranker.pkl")
    joblib.dump(rf, rf_path)
    print(f"[INFO] Random Forest reranker saved to {rf_path}")
    return rf


def rerank_candidates(
    model,
    rf: RandomForestClassifier,
    test_users: List[int],
    raw_df_train: pd.DataFrame,
    raw_df_all: pd.DataFrame,
    cf_args: Dict,
) -> np.ndarray:
    """Apply the fitted reranker to reorder LightGCN candidates for test users.

    Uses disk caches to avoid recomputing features when running repeated
    evaluations.  The returned array has the same shape as the input
    candidate list, but with items sorted by descending Random Forest
    probability.
    """
    # Derive cache paths from experiment directory
    save_path = cf_args.get("save_path", "./checkpoint")
    feature_cache_path = os.path.join(save_path, "test_features_cache.pkl")
    candidate_cache_path = os.path.join(save_path, "test_candidates_cache.pkl")
    # Attempt to load cached features
    if os.path.exists(feature_cache_path) and os.path.exists(candidate_cache_path):
        print(f"[INFO] Loading cached test features from {feature_cache_path}")
        with open(feature_cache_path, "rb") as f:
            cache = pickle.load(f)
        X = cache["X"]
        keys = cache["keys"]
        with open(candidate_cache_path, "rb") as f:
            candidates = pickle.load(f)
        print(f"[INFO] Loaded {len(X)} cached feature rows; skipping feature building.")
    else:
        # Build features from scratch
        candidates = get_topN_candidates(model, test_users, cf_args)
        X, keys = build_rerank_features_v2(
            candidates,
            model,
            test_users,
            raw_df_train,
            raw_df_all,
            checkpoint_dir=os.path.join(save_path, "rerank_test_features"),
            batch_size=cf_args.get("rerank_checkpoint_batch", 500),
        )
        # Persist caches
        with open(feature_cache_path, "wb") as f:
            pickle.dump({"X": X, "keys": keys}, f)
        with open(candidate_cache_path, "wb") as f:
            pickle.dump(candidates, f)
        print(f"[INFO] Cached test features to {feature_cache_path}")
    # Predict probabilities
    scores = rf.predict_proba(X)[:, 1]
    reranked = []
    idx = 0
    topN = cf_args.get("rerank_topN", 20)
    # For each user, slice their scores and reorder items
    for _ in tqdm(test_users, desc="reranking"):
        s = scores[idx : idx + topN]
        c = candidates[len(reranked)]
        order = np.argsort(-s)  # descending order
        reranked.append(c[order])
        idx += topN
    return np.array(reranked)