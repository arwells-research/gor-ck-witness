#!/usr/bin/env python3
"""
closure_scaling_collapse.py

Quantify (no plotting) whether D_abs exhibits a scaling-collapse when expressed as a
function of |Δq| where Δq = q - qc, across periods.

Input expectation:
  data/derived/closure_phase_diagram_v2.csv
with columns at least:
  ion_charge, period, dataset, topology, lock, rho_A, A_plus, A_minus, p_lock_D, state, D_abs

What it does:
  1) Loads the phase diagram v2 CSV.
  2) De-duplicates repeated rows caused by multiple overlapping datasets (e.g., _2to6 vs _3to6)
     by keeping unique (period, ion_charge, A_plus, A_minus, topology, lock, state, D_abs).
  3) Computes:
        delta_q = ion_charge - qc
        abs_delta = |delta_q|
  4) Produces:
      - long table: period, ion_charge, delta_q, abs_delta, D_abs, state, dataset, etc.
      - collapse summary by abs_delta: mean/std/count of D_abs across all periods
      - collapse score: overall RMSE/MAE vs the abs_delta mean curve, plus per-period RMSE/MAE

Outputs (default prefix: data/derived/closure_scaling_collapse):
  - data/derived/closure_scaling_collapse_long.csv
  - data/derived/closure_scaling_collapse_by_absdelta.csv
  - data/derived/closure_scaling_collapse_score.csv

Usage:
  PYTHONPATH=. python3 scripts/closure_scaling_collapse.py
  PYTHONPATH=. python3 scripts/closure_scaling_collapse.py --qc 5
  PYTHONPATH=. python3 scripts/closure_scaling_collapse.py --in_csv data/derived/closure_phase_diagram_v2.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scores:
    rmse: float
    mae: float
    n: int


def _rmse_mae(y: np.ndarray, yhat: np.ndarray) -> Scores:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    if y.size == 0:
        return Scores(rmse=float("nan"), mae=float("nan"), n=0)
    err = y - yhat
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    return Scores(rmse=rmse, mae=mae, n=int(y.size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="data/derived/closure_phase_diagram_v2.csv",
        help="Input phase diagram CSV (v2 recommended).",
    )
    ap.add_argument(
        "--qc",
        type=int,
        default=5,
        help="Critical charge qc used to define Δq = q - qc.",
    )
    ap.add_argument(
        "--out_prefix",
        default="data/derived/closure_scaling_collapse",
        help="Prefix for output CSVs (no extension).",
    )
    ap.add_argument(
        "--keep_period2",
        action="store_true",
        help="Include period 2 (note: truncated q coverage can bias collapse). Default: exclude period 2.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    required = {"ion_charge", "period", "D_abs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {args.in_csv}")

    # Clean types
    df["ion_charge"] = pd.to_numeric(df["ion_charge"], errors="coerce").astype("Int64")
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["D_abs"] = pd.to_numeric(df["D_abs"], errors="coerce")

    # Optionally drop period 2 by default (truncated q coverage can create fake minima)
    if not args.keep_period2:
        df = df[df["period"] != 2].copy()

    # Drop rows without finite D_abs or missing ion_charge/period
    df = df[np.isfinite(df["D_abs"].to_numpy(dtype=float))].copy()
    df = df[df["ion_charge"].notna() & df["period"].notna()].copy()

    # De-duplicate overlapping datasets.
    # If multiple datasets produce identical measurements, keep one.
    dedupe_cols = [
        "period",
        "ion_charge",
        "D_abs",
    ]
    # Add additional columns to strengthen uniqueness when present
    for c in ["A_plus", "A_minus", "topology", "lock", "state"]:
        if c in df.columns:
            dedupe_cols.append(c)

    before = len(df)
    df = df.drop_duplicates(subset=dedupe_cols, keep="first").copy()
    after = len(df)

    qc = int(args.qc)
    df["delta_q"] = (df["ion_charge"].astype(int) - qc).astype(int)
    df["abs_delta"] = df["delta_q"].abs().astype(int)

    # Sort for readability
    df = df.sort_values(["period", "ion_charge"]).reset_index(drop=True)

    # Long output
    out_long = f"{args.out_prefix}_long.csv"
    df.to_csv(out_long, index=False)

    # Collapse mean-curve by abs_delta
    grp = df.groupby("abs_delta")["D_abs"]
    df_abs = grp.agg(["count", "mean", "std", "min", "max"]).reset_index()
    df_abs = df_abs.rename(
        columns={
            "count": "n",
            "mean": "D_abs_mean",
            "std": "D_abs_std",
            "min": "D_abs_min",
            "max": "D_abs_max",
        }
    ).sort_values("abs_delta")

    out_abs = f"{args.out_prefix}_by_absdelta.csv"
    df_abs.to_csv(out_abs, index=False)

    # Build a lookup mean curve: D_hat(|Δ|) = mean across periods
    mean_curve: Dict[int, float] = {int(r.abs_delta): float(r.D_abs_mean) for r in df_abs.itertuples(index=False)}

    # Overall collapse score (vs mean curve)
    y = df["D_abs"].to_numpy(dtype=float)
    yhat = np.array([mean_curve[int(a)] for a in df["abs_delta"].to_numpy(dtype=int)], dtype=float)
    overall = _rmse_mae(y, yhat)

    # Per-period collapse score (vs the same mean curve)
    per_period_rows = []
    for p, dpf in df.groupby("period"):
        yp = dpf["D_abs"].to_numpy(dtype=float)
        yhp = np.array([mean_curve[int(a)] for a in dpf["abs_delta"].to_numpy(dtype=int)], dtype=float)
        sc = _rmse_mae(yp, yhp)
        per_period_rows.append(
            {
                "period": int(p),
                "n": sc.n,
                "rmse_vs_absdelta_mean_curve": sc.rmse,
                "mae_vs_absdelta_mean_curve": sc.mae,
            }
        )

    df_score = pd.DataFrame(
        [
            {
                "scope": "overall",
                "qc": qc,
                "n": overall.n,
                "rmse_vs_absdelta_mean_curve": overall.rmse,
                "mae_vs_absdelta_mean_curve": overall.mae,
                "rows_before_dedupe": int(before),
                "rows_after_dedupe": int(after),
                "keep_period2": bool(args.keep_period2),
            }
        ]
    )

    df_score_period = pd.DataFrame(per_period_rows).sort_values("period")
    out_score = f"{args.out_prefix}_score.csv"
    # Write a single CSV with the overall row + per-period rows appended (with a scope column)
    df_score_period.insert(0, "scope", "period")
    df_score_period.insert(1, "qc", qc)
    df_out = pd.concat([df_score, df_score_period], ignore_index=True)
    df_out.to_csv(out_score, index=False)

    # Console summary
    print(f"Wrote: {out_long}")
    print(f"Wrote: {out_abs}")
    print(f"Wrote: {out_score}")
    print()
    print("== collapse summary (by abs_delta) ==")
    print(df_abs.to_string(index=False))
    print()
    print("== collapse score (overall, vs abs_delta mean-curve) ==")
    print(df_score.to_string(index=False))
    print()
    print("== collapse score (per period) ==")
    print(df_score_period.to_string(index=False))


if __name__ == "__main__":
    main()
    