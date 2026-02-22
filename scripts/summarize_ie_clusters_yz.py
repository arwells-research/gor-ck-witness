#!/usr/bin/env python3
"""
Summarize IE-shape clusters (state=CL_*) by producing Yz prototypes and d2(Yz) sign patterns.

Run:
  PYTHONPATH=. python3 scripts/summarize_ie_clusters_yz.py \
    --in_csv data/derived/closure_phase_diagram_v2_iecluster_k2_seed0.csv \
    --out_prefix data/derived/iecluster_k2_seed0
"""

import argparse
import ast
import numpy as np
import pandas as pd


def parse_profile(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x.strip())
    raise ValueError(f"Cannot parse IE_profile: {x!r}")


def second_differences(y):
    # length 6 -> length 4
    return y[2:] - 2 * y[1:-1] + y[:-2]


def zscore(v):
    v = np.asarray(v, dtype=float)
    mu = np.mean(v)
    sd = np.std(v)
    if sd == 0:
        return np.zeros_like(v)
    return (v - mu) / sd


def sign_pattern(d2, eps=1e-12):
    out = []
    for x in d2:
        if x > eps:
            out.append("+")
        elif x < -eps:
            out.append("-")
        else:
            out.append("0")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    need = ["state", "IE_profile"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # parse all profiles
    Y = []
    for v in df["IE_profile"].tolist():
        y = np.asarray(parse_profile(v), dtype=float)
        if y.size != 6:
            raise ValueError("IE_profile must have length 6")
        Y.append(y)
    Y = np.vstack(Y)

    # rowwise zscore
    Yz = np.vstack([zscore(row) for row in Y])

    # attach index mapping for grouping
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df), dtype=int)

    rows = []
    for cl, g in df.groupby("state"):
        idx = g["_row"].to_numpy(dtype=int)
        mu_yz = np.mean(Yz[idx], axis=0)
        d2 = second_differences(mu_yz)
        pat = sign_pattern(d2)
        d2_norm = float(np.linalg.norm(d2))
        rows.append({
            "cluster": cl,
            "n": int(len(idx)),
            "mu_yz_p1": float(mu_yz[0]),
            "mu_yz_p2": float(mu_yz[1]),
            "mu_yz_p3": float(mu_yz[2]),
            "mu_yz_p4": float(mu_yz[3]),
            "mu_yz_p5": float(mu_yz[4]),
            "mu_yz_p6": float(mu_yz[5]),
            "d2_mu_yz_0": float(d2[0]),
            "d2_mu_yz_1": float(d2[1]),
            "d2_mu_yz_2": float(d2[2]),
            "d2_mu_yz_3": float(d2[3]),
            "d2_sign_pattern": pat,
            "d2_norm": d2_norm,
        })

    out_proto = args.out_prefix + "_prototypes_yz.csv"
    pd.DataFrame(rows).sort_values(["cluster"]).to_csv(out_proto, index=False)
    print("Wrote:", out_proto)


if __name__ == "__main__":
    main()
    