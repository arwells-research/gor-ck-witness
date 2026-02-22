#!/usr/bin/env python3
"""
Cluster IE-shape features (no sklearn) and replace 'state' with cluster labels.
Then write new CSV for collapse testing.

Option 1: cluster "prototype sign patterns" are computed from the *Yz centroid* (shape-normalized).

Run:
  PYTHONPATH=. python3 scripts/cluster_ie_shape_and_test_collapse.py \
      --in_csv data/derived/closure_phase_diagram_v2_canon_with_ie.csv \
      --qc 4 --k 2 --seed 0 \
      --out_csv data/derived/closure_phase_diagram_v2_iecluster_k2.csv
"""

import argparse
import ast
import os
import numpy as np
import pandas as pd


def parse_profile(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x.strip())
    raise ValueError(f"Cannot parse IE_profile: {x!r}")


def second_differences(y):
    y = np.asarray(y, dtype=float)
    return y[2:] - 2 * y[1:-1] + y[:-2]


def zrow(M):
    M = np.asarray(M, dtype=float)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (M - mu) / sd


def sign_pattern(d2, eps=1e-12):
    d2 = np.asarray(d2, dtype=float)
    out = []
    for v in d2:
        if v > eps:
            out.append("+")
        elif v < -eps:
            out.append("-")
        else:
            out.append("0")
    return "".join(out)


def kmeans_simple(X, k=2, seed=0, max_iter=100):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"Need n >= k, got n={n}, k={k}")

    # random initialization from rows
    centroids = X[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(max_iter):
        # assign
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            rows = X[labels == i]
            if rows.shape[0] == 0:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = rows.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--qc", type=int, required=True)  # kept for interface symmetry; not used here
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df = df.dropna(subset=["IE_profile", "period", "ion_charge", "D_abs"]).copy()

    # --- parse IE profiles ---
    Y = []
    for v in df["IE_profile"].tolist():
        y = np.asarray(parse_profile(v), dtype=float)
        if y.size != 6:
            raise ValueError(f"IE_profile length must be 6, got {y.size}")
        Y.append(y)
    Y = np.vstack(Y)  # (N,6)

    # --- build IE-shape feature block (same as curvature_vector_models.py) ---
    Yz = zrow(Y)                 # (N,6)  rowwise z
    D1 = np.diff(Y, axis=1)      # (N,5)
    D2 = np.diff(Y, n=2, axis=1) # (N,4)
    D1z = zrow(D1)               # (N,5)
    D2z = zrow(D2)               # (N,4)

    X = np.concatenate([Yz, D1z, D2z], axis=1)  # (N,15)

    # --- cluster ---
    labels, _centroids = kmeans_simple(X, k=args.k, seed=args.seed)

    # replace state with cluster labels
    df["state"] = [f"CL_{i}" for i in labels]

    # save clustered CSV
    df.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)
    print("Cluster counts:")
    print(pd.Series(df["state"]).value_counts().to_string())
    print()

    # --- Option 1 prototypes: use Yz centroid, then d2 of that centroid ---
    base, ext = os.path.splitext(args.out_csv)
    out_proto = f"{base}_prototypes_yz.csv"
    out_curv = f"{base}_curvature_summary_yz.csv"

    proto_rows = []
    curv_rows = []

    for i in range(args.k):
        mask = (labels == i)
        n_i = int(np.sum(mask))
        if n_i == 0:
            continue

        mu_yz = Yz[mask].mean(axis=0)          # length 6
        d2_mu = second_differences(mu_yz)      # length 4
        pat = sign_pattern(d2_mu)

        # also compute per-row curvature stats (from raw second differences D2)
        d2_rows = D2[mask]                     # (n_i,4) raw second diffs (not z)
        mean_d2 = d2_rows.mean(axis=1)         # (n_i,)
        max_abs_d2 = np.max(np.abs(d2_rows), axis=1)
        pos_frac = np.mean(d2_rows > 0, axis=1)
        sign_changes = np.sum(np.sign(d2_rows[:, 1:]) != np.sign(d2_rows[:, :-1]), axis=1)

        # prototypes table (centroid-based)
        proto_rows.append({
            "cluster": f"CL_{i}",
            "n": n_i,
            "mu_yz_p1": float(mu_yz[0]),
            "mu_yz_p2": float(mu_yz[1]),
            "mu_yz_p3": float(mu_yz[2]),
            "mu_yz_p4": float(mu_yz[3]),
            "mu_yz_p5": float(mu_yz[4]),
            "mu_yz_p6": float(mu_yz[5]),
            "d2_mu_yz_0": float(d2_mu[0]),
            "d2_mu_yz_1": float(d2_mu[1]),
            "d2_mu_yz_2": float(d2_mu[2]),
            "d2_mu_yz_3": float(d2_mu[3]),
            "d2_sign_pattern": pat,
            "d2_norm": float(np.linalg.norm(d2_mu)),
        })

        # curvature summary table (distribution within cluster)
        curv_rows.append({
            "cluster": f"CL_{i}",
            "n": n_i,
            "mean_d2_mean": float(np.mean(mean_d2)),
            "mean_d2_std": float(np.std(mean_d2)),
            "maxabs_d2_mean": float(np.mean(max_abs_d2)),
            "maxabs_d2_median": float(np.median(max_abs_d2)),
            "pos_frac_mean": float(np.mean(pos_frac)),
            "sign_changes_mean": float(np.mean(sign_changes)),
            "mean_d2_median": float(np.median(mean_d2)),
        })

    proto_df = pd.DataFrame(proto_rows).sort_values("cluster")
    curv_df = pd.DataFrame(curv_rows).sort_values("cluster")

    proto_df.to_csv(out_proto, index=False)
    curv_df.to_csv(out_curv, index=False)

    print("Wrote:", out_proto)
    print("Wrote:", out_curv)


if __name__ == "__main__":
    main()