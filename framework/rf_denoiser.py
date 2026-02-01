import os
import json
import pickle
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per‑user–item counts for basic interactions.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing at least the columns ``user_id``, ``item_id``,
        ``click``, ``like``, ``share``, ``follow`` and ``watching_times``.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by ``user_id`` and ``item_id`` with the
        aggregated counts and average watch time.
    """
    agg = df.groupby(["user_id", "item_id"]).agg(
        count_click=("click", "sum"),
        count_like=("like", "sum"),
        count_share=("share", "sum"),
        count_follow=("follow", "sum"),
        avg_watch_time=("watching_times", "mean"),
    ).reset_index()
    return agg


def normalize_video_category(x) -> int:
    """Normalize a variety of video category encodings to simple integers.

    This helper cleans up the ``video_category`` column which can be 0/1,
    strings "0"/"1" or other strings.  Unknown values are mapped to 4.
    """
    if x == 0:
        return 0
    if x == 1:
        return 1
    if isinstance(x, str):
        x = x.strip()
        if x == "0":
            return 2
        if x == "1":
            return 3
        return 4
    return 4


def build_rf_features_v2(raw_df_train: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
    """Construct improved RF features and labels from raw training data.

    The returned ``ui_keys`` DataFrame contains the ``user_id`` and
    ``item_id`` for each interaction.  ``X`` is a 2‑D numpy array of
    engineered features and ``y`` is a binary label computed using
    multiple criteria (explicit engagement, high watch time, repeated
    clicks and long absolute watch time).  ``feature_names`` is a list
    describing each column of ``X``.
    """
    # Basic behaviour features (count_click, count_like, count_share, count_follow, avg_watch_time)
    beh = build_behavior_features(raw_df_train)

    # User metadata (gender, age)
    user_meta = raw_df_train[["user_id", "gender", "age"]].drop_duplicates()

    # Item metadata (video_category) – normalized
    item_meta = raw_df_train[["item_id", "video_category"]].drop_duplicates()
    item_meta["video_category"] = (
        item_meta["video_category"].apply(normalize_video_category).astype(np.int32)
    )

    # Per‑user watch time statistics
    user_watch_stats = raw_df_train.groupby("user_id")["watching_times"].agg(["mean", "std"]).reset_index()
    user_watch_stats["std"] = user_watch_stats["std"].fillna(0)

    # Merge all feature sources
    feat = (
        beh.merge(user_meta, on="user_id", how="left")
        .merge(item_meta, on="item_id", how="left")
        .merge(user_watch_stats, on="user_id", how="left")
    )
    # Fill any missing metadata
    feat[["gender", "age", "video_category", "mean", "std"]] = feat[
        ["gender", "age", "video_category", "mean", "std"]
    ].fillna(0)

    # === Multi‑criteria label definition ===
    # Criterion 1: the user explicitly liked/shared/followed this item
    feat["has_engagement"] = ((feat["count_like"] > 0) | (feat["count_share"] > 0) | (feat["count_follow"] > 0)).astype(int)
    # Criterion 2: watch time is significantly above this user's average
    feat["high_watch"] = (feat["avg_watch_time"] > (feat["mean"] + 0.5 * feat["std"])).astype(int)
    # Criterion 3: the user clicked this item multiple times
    feat["repeated"] = (feat["count_click"] >= 2).astype(int)
    # Criterion 4: long absolute watch time (>30 seconds)
    feat["long_watch"] = (feat["avg_watch_time"] > 30).astype(int)
    # Final label: positive if any criterion is satisfied
    feat["label"] = (
        (feat["has_engagement"] == 1)
        | (feat["high_watch"] == 1)
        | (feat["repeated"] == 1)
        | (feat["long_watch"] == 1)
    ).astype(int)

    # Collect statistics for the user when fitting
    print("\n[RF Label V2 Statistics]")
    print(f"  Total interactions: {len(feat)}")
    print(f"  Positive ratio: {feat['label'].mean():.2%}")
    print("  Breakdown:")
    print(f"    - Has engagement (like/share/follow): {feat['has_engagement'].mean():.2%}")
    print(f"    - High watch time (relative): {feat['high_watch'].mean():.2%}")
    print(f"    - Repeated clicks (≥2): {feat['repeated'].mean():.2%}")
    print(f"    - Long watch time (>30s): {feat['long_watch'].mean():.2%}")

    y = feat["label"].values
    # Drop non‑feature columns
    X = feat.drop(
        columns=[
            "user_id",
            "item_id",
            "label",
            "has_engagement",
            "high_watch",
            "repeated",
            "long_watch",
            "mean",
            "std",
        ]
    )
    return feat[["user_id", "item_id"]], X.values, y, X.columns.tolist()


def train_rf_denoiser_v2(raw_df_train: pd.DataFrame, cf_args: Dict) -> Tuple[RandomForestClassifier, pd.DataFrame, list]:
    """Train a Random Forest denoiser using the improved feature set.

    Returns the fitted classifier, a DataFrame with probabilities for each
    ``(user_id, item_id)`` pair, and the list of feature names.
    """
    ui_keys, X, y, feat_names = build_rf_features_v2(raw_df_train)

    print("\n[Training RF with improved labels]")
    print(f"  Features: {len(feat_names)}")
    print(f"  Samples: {len(X)}")
    print(f"  Positive class: {y.sum()} ({y.mean():.2%})")

    rf = RandomForestClassifier(
        n_estimators=cf_args.get("rf_n_estimators", 200),
        max_depth=cf_args.get("rf_max_depth", 14),
        random_state=cf_args.get("rf_random_state", 42),
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    probs = rf.predict_proba(X)[:, 1]

    # Merge keys and probabilities into a single result DataFrame
    res = ui_keys.copy()
    res["prob"] = probs
    res["label"] = y

    # Feature importance summary
    importance = (
        pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    print("\n[Top 10 Important Features]")
    print(importance.head(10).to_string(index=False))

    return rf, res, feat_names


def prepare_training_data_v2(
    raw_df_train: pd.DataFrame,
    train_click: pd.DataFrame,
    cf_args: Dict,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Prepare edge lists for LightGCN training using the RF denoiser.

    When ``denoise`` is disabled in the configuration, this function simply
    returns the click edges and ``None`` for the negative set and weights.

    Otherwise it trains a denoiser, extracts true negative edges and
    optionally computes soft weights for each click edge.  The weights
    transformation can be configured via ``weight_transform``.
    """
    # If denoising is off, return all click edges
    if not cf_args.get("denoise", False):
        print("[INFO] Denoise OFF → use click‑only training")
        return train_click[["user_id", "item_id"]], None, None

    print("[INFO] Denoise ON → training IMPROVED RF denoiser")
    rf, rf_res, _ = train_rf_denoiser_v2(raw_df_train, cf_args)

    # Inspect distribution of probabilities
    print("\n[RF Probability Distribution]")
    print(f"  Min: {rf_res['prob'].min():.4f}")
    print(f"  25%: {rf_res['prob'].quantile(0.25):.4f}")
    print(f"  50%: {rf_res['prob'].quantile(0.50):.4f}")
    print(f"  75%: {rf_res['prob'].quantile(0.75):.4f}")
    print(f"  Max: {rf_res['prob'].max():.4f}")
    print(f"  Mean: {rf_res['prob'].mean():.4f}")

    # Threshold for true negatives
    neg_th = cf_args.get("soft_denoise_neg_th", 0.3)
    # Identify true negative edges (low probability clicks)
    true_neg_df = rf_res[rf_res["prob"] <= neg_th][["user_id", "item_id"]]
    # Keep all clicks as positives
    pos_df = train_click[["user_id", "item_id"]].copy()

    # Soft weights (optional)
    weights = None
    if cf_args.get("use_soft_weights", False):
        # Merge RF probabilities into click edges
        train_click_with_prob = train_click.merge(
            rf_res[["user_id", "item_id", "prob"]],
            on=["user_id", "item_id"],
            how="left",
        )
        train_click_with_prob["prob"].fillna(0.5, inplace=True)
        weights = train_click_with_prob["prob"].values.copy()

        transform = cf_args.get("weight_transform", "clip_rescale")
        if transform == "clip_rescale":
            # Clip to [0.3, 1.0]
            weights = np.clip(weights, 0.3, 1.0)
        elif transform == "sqrt_shift":
            weights = np.sqrt(weights) * 0.7 + 0.3
            weights = np.clip(weights, 0.3, 1.0)
        elif transform == "sigmoid":
            weights = 1 / (1 + np.exp(-10 * (weights - 0.5)))
            weights = weights * 0.7 + 0.3

        print("\n[Edge Weights]")
        print(f"  Transform: {transform}")
        print(f"  Min: {weights.min():.4f}")
        print(f"  Mean: {weights.mean():.4f}")
        print(f"  Max: {weights.max():.4f}")
        print(f"  % weights > 0.5: {(weights > 0.5).mean():.2%}")

    print("\n[Final Data Summary]")
    print(f"  Training edges (ALL clicks): {len(pos_df)}")
    print(f"  True negatives (prob ≤ {neg_th}): {len(true_neg_df)}")
    if weights is not None:
        print("  Using soft weights: YES")
    else:
        print("  Using soft weights: NO (binary graph)")

    return pos_df, true_neg_df, weights


def denoise_interactions(rf_res: pd.DataFrame, pos_th: float = 0.6, neg_th: float = 0.4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return positive and true negative edges given RF probabilities.

    Parameters
    ----------
    rf_res : pd.DataFrame
        DataFrame returned by ``train_rf_denoiser_v2`` with columns
        ``user_id``, ``item_id`` and ``prob``.
    pos_th : float, optional
        Probability threshold above which a click is considered a positive
        training edge (default 0.6).
    neg_th : float, optional
        Probability threshold below which a click is considered a true
        negative (default 0.4).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The positive edge list and true negative edge list, each
        containing ``user_id`` and ``item_id`` columns.
    """
    pos_df = rf_res[rf_res["prob"] >= pos_th][["user_id", "item_id"]]
    true_neg_df = rf_res[rf_res["prob"] <= neg_th][["user_id", "item_id"]]
    return pos_df.drop_duplicates(), true_neg_df.drop_duplicates()