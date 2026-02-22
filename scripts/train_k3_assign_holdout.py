#!/usr/bin/env python3
"""
Leave-one-period-out (or arbitrary holdout) training for k=3 IE-shape clusters.

Train k-means (no sklearn) on all periods except --holdout_period.
Assign the held-out period rows by nearest centroid (IE-shape feature space).

Outputs (all prefixed by --out_prefix):
  *_train_clusters.csv         (training rows with cluster labels in 'state')
  *_holdout_assignments.csv    (held-out rows with assigned cluster labels in 'state')
  *_full_with_assignments.csv  (all rows with 'state' replaced by cluster labels)
  *_centroids.npy              (centroids in IE-shape feature space)
"""

import argparse
import ast
import numpy as np
import pandas as pd


def parse_profile(x):
    if isinstance(x, list):
        return x
    return ast.literal_eval(str(x).strip())


def zrow(M):
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (M - mu) / sd


def kmeans_simple(X, k=3, seed=0, max_iter=100):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"Need at least k rows to init kmeans: n={n} k={k}")

    centroids = X[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(max_iter):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        new = np.zeros_like(centroids)
        for i in range(k):
            rows = X[labels == i]
            new[i] = rows.mean(axis=0) if len(rows) > 0 else centroids[i]

        if np.allclose(new, centroids):
            break
        centroids = new

    return labels, centroids


def build_features(df):
    Y = np.vstack([np.asarray(parse_profile(v), float) for v in df["IE_profile"]])
    if Y.shape[1] != 6:
        raise ValueError(f"IE_profile must be length 6; got shape {Y.shape}")

    Yz = zrow(Y)
    D1z = zrow(np.diff(Y, axis=1))        # (N,5)
    D2z = zrow(np.diff(Y, n=2, axis=1))   # (N,4)
    return np.concatenate([Yz, D1z, D2z], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--holdout_period", type=int, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df = df.dropna(subset=["IE_profile", "period", "D_abs"]).copy()
    df["period"] = df["period"].astype(int)

    train = df[df["period"] != args.holdout_period].copy()
    hold  = df[df["period"] == args.holdout_period].copy()
    if len(hold) == 0:
        raise ValueError(f"No rows found for holdout_period={args.holdout_period}")

    X_train = build_features(train)
    labels, centroids = kmeans_simple(X_train, k=args.k, seed=args.seed)

    train["state"] = [f"CL_{i}" for i in labels]

    # assign holdout by nearest centroid
    X_hold = build_features(hold)
    dists = np.sum((X_hold[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    hold_labels = np.argmin(dists, axis=1)
    hold["state"] = [f"CL_{i}" for i in hold_labels]

    full = pd.concat([train, hold]).sort_index()

    train_out = args.out_prefix + "_train_clusters.csv"
    hold_out  = args.out_prefix + "_holdout_assignments.csv"
    full_out  = args.out_prefix + "_full_with_assignments.csv"
    cen_out   = args.out_prefix + "_centroids.npy"

    train.to_csv(train_out, index=False)
    hold.to_csv(hold_out, index=False)
    full.to_csv(full_out, index=False)
    np.save(cen_out, centroids)

    print("Wrote:")
    print(" ", train_out)
    print(" ", hold_out)
    print(" ", full_out)
    print(" ", cen_out)
    print("\nHoldout cluster counts:")
    print(hold["state"].value_counts().to_string())


if __name__ == "__main__":
    main()
