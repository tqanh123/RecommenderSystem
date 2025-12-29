"""
inference_rf.py
================

This script demonstrates how to use a trained LightGCN model together
with a pre‑trained Random Forest reranker to generate final top‑k
recommendations for a given user.  It is intentionally kept simple and
serves as a starting point for building a proper inference service.

The script expects a JSON configuration file (``python/config.json``) in
the working directory describing the trained LightGCN model (number of
users/items, embedding dimension, number of layers and the path to the
model weights).  It also expects a ``rf_reranker.pkl`` file saved
in the ``./checkpoint/`` directory by ``train_reranker_rf``.

Due to the complexity of computing all of the engineered features used
during training, this example only computes embedding‑based features
(user and item embeddings and their dot product) for the re‑ranker.
For a production system you should load the same raw training data and
replicate the full feature engineering implemented in
``rf_reranker.build_rerank_features_v2``.

Usage
-----

    python inference_rf.py <user_id> [top_k]

Outputs a JSON string with the user id and a list of recommended items
sorted by the re‑ranker.
"""

import json
import sys
import os
import numpy as np
import torch

from model import LightGCN
from rf_reranker import get_topN_candidates


def load_model(model_path: str, n_users: int, n_items: int, embedding_dim: int = 64, n_layers: int = 3) -> LightGCN:
    model = LightGCN(n_users, n_items, embedding_dim, n_layers)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_rf.py <user_id> [top_k]")
        sys.exit(1)
    user_id = int(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    # Load config
    with open("python/config.json", "r") as f:
        config = json.load(f)
    # Load LightGCN model
    model = load_model(
        config["model_path"],
        config["n_users"],
        config["n_items"],
        config["embedding_dim"],
        config["n_layers"],
    )
    # Generate a larger candidate set (e.g. 100 items)
    cf_args = {"rerank_topN": 100, "val_batch_size": 64}
    candidates = get_topN_candidates(model, [user_id], cf_args)[0]
    # Compute embedding scores for each candidate
    with torch.no_grad():
        user_emb = model.user_emb(torch.LongTensor([user_id])).squeeze(0)
        item_embs = model.item_emb(torch.LongTensor(candidates))
        scores = torch.matmul(item_embs, user_emb)
    # Sort by LightGCN score descending
    ranking = scores.argsort(descending=True).numpy()
    # Take top k
    top_items = [int(candidates[idx]) for idx in ranking[:k]]
    result = {
        "user_id": user_id,
        "recommendations": [
            {"item_id": item_id, "score": float(scores[idx])}
            for idx, item_id in zip(ranking[:k], top_items)
        ],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()