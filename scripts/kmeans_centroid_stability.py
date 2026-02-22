#!/usr/bin/env python3
"""
KMeans centroid stability across seeds (no sklearn).

What it does:
- For each seed:
  - (optional) run cluster_ie_shape_and_test_collapse.py to create clustered CSV
  - read clustered CSV
  - compute centroid IE mean vector (length 6)
  - compute centroid d2(mean) (length 4)
  - compute sign pattern of d2(mean): e.g. "+-+-"
- Match clusters across seeds to a canonical ordering using minimal L2 cost on d2(mean).
- Summarize sign-pattern stability across seeds.

Assumptions about clustered CSV:
- Has IE_profile column (list or stringified list length 6)
- Has a cluster label column: prefer 'state' (e.g. CL_0, CL_1, CL_2)
- Has 'topology' and (optionally) 'lock'

Usage examples:
  PYTHONPATH=. python3 scripts/kmeans_centroid_stability.py \
    --in_csv data/derived/closure_phase_diagram_v2_canon_with_ie.csv \
    --qc 4 --k 3 --seeds 0-9 --out_dir data/derived/stability_k3_qc4 \
    --rerun

  PYTHONPATH=. python3 scripts/kmeans_centroid_stability.py \
    --in_csv data/derived/closure_phase_diagram_v2_canon_with_ie.csv \
    --qc 4 --k 3 --seeds 0-9 --out_dir data/derived/stability_k3_qc4
"""

from __future__ import annotations

import argparse
import ast
import itertools
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_profile(x) -> np.ndarray:
    if isinstance(x, list):
        y = np.asarray(x, dtype=float)
    elif isinstance(x, str):
        y = np.asarray(ast.literal_eval(x.strip()), dtype=float)
    else:
        raise ValueError(f"Bad IE_profile type: {type(x)}")
    if y.size != 6:
        raise ValueError(f"Expected IE_profile length 6, got {y.size}")
    return y


def second_differences(y: np.ndarray) -> np.ndarray:
    # length 6 -> length 4
    return y[2:] - 2.0 * y[1:-1] + y[:-2]


def sign_pattern(v: np.ndarray, eps: float = 1e-12) -> str:
    out = []
    for x in v:
        if x > eps:
            out.append("+")
        elif x < -eps:
            out.append("-")
        else:
            out.append("0")
    return "".join(out)


def best_assignment(cost: np.ndarray) -> Tuple[float, Tuple[int, ...]]:
    """
    Solve assignment by brute force permutations (k <= 6 is fine).
    cost is (k,k): cost[i, j] = cost matching canonical i to current j
    Returns (min_cost, perm) where perm[i] = j
    """
    k = cost.shape[0]
    best = None
    best_perm = None
    for perm in itertools.permutations(range(k)):
        c = float(sum(cost[i, perm[i]] for i in range(k)))
        if best is None or c < best:
            best = c
            best_perm = perm
    return best, best_perm  # type: ignore


@dataclass
class ClusterSummary:
    seed: int
    label: str
    n: int
    mean_ie: np.ndarray        # (6,)
    d2_mean: np.ndarray        # (4,)
    pat: str
    topo_frac: Dict[str, float]
    lock_frac: Dict[str, float]


def compute_cluster_summaries(df: pd.DataFrame, label_col: str) -> List[ClusterSummary]:
    if "IE_profile" not in df.columns:
        raise ValueError("Clustered CSV missing IE_profile")
    if "topology" not in df.columns:
        raise ValueError("Clustered CSV missing topology")

    # seed is not stored in csv; caller provides.
    summaries: List[ClusterSummary] = []

    for lab, g in df.groupby(label_col):
        Y = np.vstack([parse_profile(x) for x in g["IE_profile"].tolist()])  # (n,6)
        mean_ie = Y.mean(axis=0)
        d2 = second_differences(mean_ie)
        pat = sign_pattern(d2)

        topo = g["topology"].astype(str)
        topo_frac = (topo.value_counts(normalize=True)).to_dict()

        lock_frac: Dict[str, float] = {}
        if "lock" in g.columns:
            lk = g["lock"].astype(str)
            # keep only meaningful values if present
            lk = lk[lk.isin(["locked", "unlocked"])]
            if len(lk) > 0:
                lock_frac = (lk.value_counts(normalize=True)).to_dict()

        summaries.append(
            ClusterSummary(
                seed=-1,  # filled by caller
                label=str(lab),
                n=int(len(g)),
                mean_ie=mean_ie,
                d2_mean=d2,
                pat=pat,
                topo_frac=topo_frac,
                lock_frac=lock_frac,
            )
        )

    # sort by label for deterministic output pre-matching
    summaries.sort(key=lambda s: s.label)
    return summaries


def fmt_frac(d: Dict[str, float]) -> str:
    if not d:
        return "-"
    items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join([f"{k}:{v:.3f}" for k, v in items])


def run_kmeans_cluster(in_csv: str, qc: int, k: int, seed: int, out_csv: str) -> None:
    cmd = [
        "python3",
        "scripts/cluster_ie_shape_and_test_collapse.py",
        "--in_csv", in_csv,
        "--qc", str(qc),
        "--k", str(k),
        "--seed", str(seed),
        "--out_csv", out_csv,
    ]
    # Use PYTHONPATH=. externally (caller command) or set it here:
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--qc", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,1,2,5")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rerun", action="store_true", help="rerun kmeans for each seed")
    ap.add_argument("--label_col", default="state", help="cluster label column in out_csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # parse seeds
    seeds: List[int] = []
    s = args.seeds.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in s.split(",") if x.strip()]

    # Collect per-seed cluster summaries
    all_seed_summaries: Dict[int, List[ClusterSummary]] = {}

    for seed in seeds:
        out_csv = os.path.join(args.out_dir, f"tmp_kmeans_k{args.k}_qc{args.qc}_seed{seed}.csv")
        if args.rerun or (not os.path.exists(out_csv)):
            run_kmeans_cluster(args.in_csv, args.qc, args.k, seed, out_csv)

        df = pd.read_csv(out_csv)
        if args.label_col not in df.columns:
            raise ValueError(f"{out_csv}: missing label_col={args.label_col}. "
                             f"Have columns: {df.columns.tolist()}")

        summaries = compute_cluster_summaries(df, args.label_col)
        for cs in summaries:
            cs.seed = seed
        all_seed_summaries[seed] = summaries

    # Choose canonical ordering from the first seed (smallest seed)
    seed0 = sorted(all_seed_summaries.keys())[0]
    canon = all_seed_summaries[seed0]
    if len(canon) != args.k:
        raise ValueError(f"Seed {seed0} produced {len(canon)} clusters, expected k={args.k}")

    canon_labels = [f"CAN_{i}" for i in range(args.k)]
    canon_d2 = np.vstack([c.d2_mean for c in canon])  # (k,4)
    canon_pat = [c.pat for c in canon]

    # Match each seed’s clusters to canonical by minimizing ||d2_mean - canon_d2||
    matched_rows = []
    pat_counts: Dict[str, int] = {}
    match_costs = []

    for seed in sorted(all_seed_summaries.keys()):
        summ = all_seed_summaries[seed]
        if len(summ) != args.k:
            raise ValueError(f"Seed {seed} produced {len(summ)} clusters, expected k={args.k}")

        d2 = np.vstack([c.d2_mean for c in summ])  # (k,4)
        cost = np.zeros((args.k, args.k), dtype=float)
        for i in range(args.k):
            for j in range(args.k):
                cost[i, j] = float(np.linalg.norm(canon_d2[i] - d2[j]))

        min_cost, perm = best_assignment(cost)
        match_costs.append((seed, min_cost))

        # perm[i] = j : canonical i matched to seed cluster j
        for i in range(args.k):
            j = perm[i]
            c = summ[j]
            pat_counts[c.pat] = pat_counts.get(c.pat, 0) + 1
            matched_rows.append({
                "seed": seed,
                "canon": canon_labels[i],
                "canon_pat": canon_pat[i],
                "cluster_label": c.label,
                "n": c.n,
                "pat": c.pat,
                "d2_0": c.d2_mean[0],
                "d2_1": c.d2_mean[1],
                "d2_2": c.d2_mean[2],
                "d2_3": c.d2_mean[3],
                "topology_frac": fmt_frac(c.topo_frac),
                "lock_frac": fmt_frac(c.lock_frac),
            })

    out_match = pd.DataFrame(matched_rows).sort_values(["seed", "canon"])
    out_match_path = os.path.join(args.out_dir, "centroid_match_table.csv")
    out_match.to_csv(out_match_path, index=False)

    out_cost = pd.DataFrame(match_costs, columns=["seed", "assignment_cost_l2_d2"])
    out_cost_path = os.path.join(args.out_dir, "assignment_costs.csv")
    out_cost.to_csv(out_cost_path, index=False)

    out_pat = pd.DataFrame(
        sorted(pat_counts.items(), key=lambda kv: (-kv[1], kv[0])),
        columns=["d2_sign_pattern", "count_over_all_seed_clusters"]
    )
    out_pat_path = os.path.join(args.out_dir, "pattern_counts.csv")
    out_pat.to_csv(out_pat_path, index=False)

    # Console summary
    print(f"\nWrote:\n  {out_match_path}\n  {out_cost_path}\n  {out_pat_path}\n")
    print("Canonical seed:", seed0)
    print("Canonical patterns:", canon_pat)
    print("\nPattern counts over all (seed,cluster):")
    print(out_pat.to_string(index=False))
    print("\nAssignment costs (lower = more stable centroid geometry):")
    print(out_cost.to_string(index=False))


if __name__ == "__main__":
    main()
