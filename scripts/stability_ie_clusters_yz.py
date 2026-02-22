#!/usr/bin/env python3
"""
Stability analysis across seeds using cluster prototypes (Yz) and d2(Yz).

Input: many *_prototypes_yz.csv files with columns:
  cluster, n, d2_mu_yz_0..d2_mu_yz_3, d2_sign_pattern, d2_norm

Run:
  PYTHONPATH=. python3 scripts/stability_ie_clusters_yz.py \
    --glob "data/derived/iecluster_k2_seed*_prototypes_yz.csv" \
    --canonical_seed 0 \
    --out_csv data/derived/stability_k2_yz_assignments.csv
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd


def extract_seed(path: str) -> int:
    m = re.search(r"seed(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not extract seed from filename: {path}")
    return int(m.group(1))


def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def assignment_min_cost(d2_seed: np.ndarray, d2_canon: np.ndarray):
    """
    Solve tiny assignment by brute force for k<=4.
    Returns (perm, cost) where perm maps canon index -> seed row index.
    """
    k = d2_canon.shape[0]
    idx = list(range(k))
    best = None
    best_cost = None

    # brute force permutations
    import itertools
    for perm in itertools.permutations(idx):
        cost = 0.0
        for j, i_seed in enumerate(perm):
            diff = d2_seed[i_seed] - d2_canon[j]
            cost += float(np.dot(diff, diff))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = perm
    return best, best_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob to prototypes_yz.csv files")
    ap.add_argument("--canonical_seed", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    # load all
    by_seed = {}
    for p in paths:
        seed = extract_seed(p)
        df = pd.read_csv(p)
        need = ["cluster", "n", "d2_mu_yz_0", "d2_mu_yz_1", "d2_mu_yz_2", "d2_mu_yz_3", "d2_sign_pattern", "d2_norm"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"{p}: missing columns {miss}")
        by_seed[seed] = df.copy()

    if args.canonical_seed not in by_seed:
        raise ValueError(f"canonical_seed={args.canonical_seed} not present in inputs. have={sorted(by_seed.keys())}")

    canon_df = by_seed[args.canonical_seed].sort_values("cluster").reset_index(drop=True)
    d2_canon = canon_df[["d2_mu_yz_0","d2_mu_yz_1","d2_mu_yz_2","d2_mu_yz_3"]].to_numpy(dtype=float)
    canon_patterns = canon_df["d2_sign_pattern"].tolist()

    print("Canonical seed:", args.canonical_seed)
    print("Canonical patterns:", canon_patterns)

    rows_out = []
    pattern_counts = {}

    assign_costs = []

    for seed, sdf0 in sorted(by_seed.items()):
        sdf = sdf0.sort_values("cluster").reset_index(drop=True)
        d2_seed = sdf[["d2_mu_yz_0","d2_mu_yz_1","d2_mu_yz_2","d2_mu_yz_3"]].to_numpy(dtype=float)

        perm, cost = assignment_min_cost(d2_seed, d2_canon)
        assign_costs.append({"seed": seed, "assignment_cost_l2_d2": cost})

        for j_canon, i_seed in enumerate(perm):
            canon_pat = canon_patterns[j_canon]
            seed_pat = str(sdf.loc[i_seed, "d2_sign_pattern"])
            pattern_counts[seed_pat] = pattern_counts.get(seed_pat, 0) + 1

            v_seed = d2_seed[i_seed]
            v_canon = d2_canon[j_canon]
            rows_out.append({
                "seed": seed,
                "canon_idx": j_canon,
                "canon_pat": canon_pat,
                "seed_cluster": str(sdf.loc[i_seed, "cluster"]),
                "seed_pat": seed_pat,
                "n": int(sdf.loc[i_seed, "n"]),
                "cosine_align": cosine(v_seed, v_canon),
                "d2_0": float(v_seed[0]),
                "d2_1": float(v_seed[1]),
                "d2_2": float(v_seed[2]),
                "d2_3": float(v_seed[3]),
            })

    out = pd.DataFrame(rows_out)
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)

    # stability summary
    print("\nPattern counts across matched clusters:")
    for k, v in sorted(pattern_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k}: {v}")

    c = out["cosine_align"].to_numpy(dtype=float)
    c = c[np.isfinite(c)]
    if c.size:
        print("\nCosine alignment stats:")
        print("mean =", float(np.mean(c)))
        print("min  =", float(np.min(c)))
        print("std  =", float(np.std(c)))

    ac = pd.DataFrame(assign_costs).sort_values("seed")
    print("\nAssignment costs (lower = more stable centroid geometry):")
    print(ac.to_string(index=False))


if __name__ == "__main__":
    main()
    