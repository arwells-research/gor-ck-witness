#!/usr/bin/env python3
"""
Permutation test: can the mixture-only weighted R^2 be achieved by chance?

Model (mixture-only):
  For each state s, compute mu_s = mean(D_abs | state=s).
  For each abs_delta d, compute w_{d,s} = P(state=s | abs_delta=d) from counts.
  Predict: D_mix_pred(d) = sum_s w_{d,s} * mu_s.
  Score: weighted R^2 across abs_delta bins, weights = n_d.

Permutation null:
  Shuffle the 'state' labels across rows (preserves cluster sizes),
  recompute mixture-only weighted R^2 each time.

No sklearn required.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def weighted_r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    w = np.asarray(w, float)

    m = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
    y, yhat, w = y[m], yhat[m], w[m]
    if y.size < 2:
        return float("nan")

    ybar = np.sum(w * y) / np.sum(w)
    sse = np.sum(w * (y - yhat) ** 2)
    sst = np.sum(w * (y - ybar) ** 2)
    if sst <= 0:
        return float("nan")
    return float(1.0 - sse / sst)


def mixture_only_r2(df: pd.DataFrame, qc: int) -> tuple[float, pd.DataFrame]:
    """
    Returns:
      (r2_weighted, table_by_abs_delta)
    where table_by_abs_delta includes observed mean and mixture prediction.
    """
    # required cols
    need = ["ion_charge", "D_abs", "state"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}; need {need}")

    d = df.copy()
    d = d.dropna(subset=["ion_charge", "D_abs", "state"])
    d["ion_charge"] = d["ion_charge"].astype(int)
    d["D_abs"] = pd.to_numeric(d["D_abs"], errors="coerce")
    d["state"] = d["state"].astype(str)

    d["abs_delta"] = (d["ion_charge"] - int(qc)).abs().astype(int)

    # per-state mean collapse
    mu = d.groupby("state")["D_abs"].mean()  # mu_s

    # weights w_{d,s} = P(s | d) from counts within abs_delta
    counts = (
        d.groupby(["abs_delta", "state"])
         .size()
         .rename("count")
         .reset_index()
    )
    totals = d.groupby("abs_delta").size().rename("n").reset_index()

    tmp = counts.merge(totals, on="abs_delta", how="left")
    tmp["w"] = tmp["count"] / tmp["n"]
    tmp["mu_state"] = tmp["state"].map(mu)

    # D_mix_pred(d) = sum_s w_{d,s} * mu_s
    pred = (
        tmp.assign(wmu=tmp["w"] * tmp["mu_state"])
           .groupby("abs_delta")["wmu"]
           .sum()
           .rename("D_mix_pred")
           .reset_index()
    )

    # observed mean per abs_delta
    obs = (
        d.groupby("abs_delta")["D_abs"]
         .mean()
         .rename("D_obs_mean")
         .reset_index()
    )

    tab = obs.merge(pred, on="abs_delta", how="left").merge(totals, on="abs_delta", how="left")
    tab = tab.sort_values("abs_delta").reset_index(drop=True)

    r2w = weighted_r2(tab["D_obs_mean"].to_numpy(), tab["D_mix_pred"].to_numpy(), tab["n"].to_numpy())
    return r2w, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--qc", type=int, required=True)
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default=None, help="Optional: write the observed table_by_abs_delta")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    r2_obs, tab = mixture_only_r2(df, qc=args.qc)
    print(f"Observed mixture-only weighted R^2 = {r2_obs:.6f}")
    if args.out_csv:
        tab.to_csv(args.out_csv, index=False)
        print(f"Wrote: {args.out_csv}")

    # permutation null: shuffle state labels across rows (preserves cluster sizes)
    rng = np.random.default_rng(args.seed)
    states = df["state"].astype(str).to_numpy()
    r2_perm = np.empty(args.nperm, dtype=float)

    for i in range(args.nperm):
        perm_states = states.copy()
        rng.shuffle(perm_states)
        dfp = df.copy()
        dfp["state"] = perm_states
        r2_perm[i], _ = mixture_only_r2(dfp, qc=args.qc)

    # p-value: P(R2_perm >= R2_obs)
    # (add +1 smoothing to avoid zero p)
    ge = np.sum(np.isfinite(r2_perm) & (r2_perm >= r2_obs))
    finite = np.sum(np.isfinite(r2_perm))
    p = (ge + 1.0) / (finite + 1.0)

    print("\nPermutation null:")
    print(f"  nperm={args.nperm}  finite={finite}")
    print(f"  mean={np.nanmean(r2_perm):.6f}  std={np.nanstd(r2_perm):.6f}")
    print(f"  max={np.nanmax(r2_perm):.6f}  min={np.nanmin(r2_perm):.6f}")
    print(f"  p(R2_perm >= R2_obs) = {p:.6g}")


if __name__ == "__main__":
    main()
    