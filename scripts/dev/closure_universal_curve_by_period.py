#!/usr/bin/env python3
"""
Compute "critical charge" qc by period from the closure phase diagram.

Goal
----
Given the derived closure phase diagram (typically produced by:
  - scripts/build_closure_phase_diagram.py
  - scripts/closure_phase_add_state_and_strength.py

and saved as:
  data/derived/closure_phase_diagram_v2.csv

we estimate (per period) the "critical" ion_charge qc where the *mean* closure
order-parameter magnitude D_abs is minimized across available ion charges.

Why this is the right next step
-------------------------------
- If qc(period) is constant across periods -> strong universality signal.
- If qc(period) drifts with period -> indicates principal-quantum-number dependence
  (or missing-data / stage-coverage artifacts).

Inputs
------
Expected columns (from closure_phase_diagram_v2.csv):
  ion_charge, period, dataset, topology, lock, rho_A, A_plus, A_minus, p_lock_D, state, D_abs

Note: You may have duplicate rows for the same (ion_charge, period) due to multiple
raw datasets (e.g., *_2to6.csv and *_2to7.csv). This script deduplicates by
grouping and averaging numeric columns, and taking the first non-null for strings.

Outputs
-------
1) Per-period qc summary:
   data/derived/closure_universal_by_period.csv

2) Long-form per-period/per-q means for audit:
   data/derived/closure_universal_by_period_long.csv

Example
-------
PYTHONPATH=. python3 scripts/closure_universal_curve_by_period.py \
  --in_csv data/derived/closure_phase_diagram_v2.csv \
  --out_csv data/derived/closure_universal_by_period.csv

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QcResult:
    period: int
    qc: int
    Dmin: float
    Dsecond: float
    delta_second_minus_min: float
    n_q: int
    q_min: int
    q_max: int
    frac_locked_two_lobe: float
    frac_two_lobe: float
    q_locked_any: Optional[int]
    q_locked_majority: Optional[int]
    q_balanced_any: Optional[int]
    q_balanced_majority: Optional[int]


def _first_non_null(series: pd.Series) -> str | float | None:
    for x in series:
        if pd.notna(x):
            return x
    return None


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate rows with the same (period, ion_charge) coming from multiple datasets.
    Numeric columns are averaged; string columns take first non-null.
    """
    # Ensure expected columns exist (fail loudly if the pipeline changed).
    required = {"period", "ion_charge", "D_abs", "topology", "lock", "p_lock_D", "state"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    str_cols = [c for c in df.columns if c not in num_cols]

    g = df.groupby(["period", "ion_charge"], as_index=False)

    # Build aggregation dict
    agg = {}
    for c in num_cols:
        agg[c] = "mean"
    for c in str_cols:
        agg[c] = _first_non_null

    out = g.agg(agg)

    # Ensure integer typing where appropriate
    out["period"] = out["period"].astype(int)
    out["ion_charge"] = out["ion_charge"].astype(int)
    return out


def _is_two_lobe(row: pd.Series) -> bool:
    return str(row.get("topology", "")).strip().lower() == "two_lobe"


def _is_locked(row: pd.Series, p_thresh: float) -> bool:
    lk = str(row.get("lock", "")).strip().lower()
    if lk == "locked":
        return True
    # Some rows may have NaN lock but have p_lock_D; use it as fallback.
    p = row.get("p_lock_D", np.nan)
    if pd.notna(p) and float(p) <= p_thresh and _is_two_lobe(row):
        return True
    return False


def _balanced(row: pd.Series, d_thresh: float) -> bool:
    d = row.get("D_abs", np.nan)
    if pd.isna(d):
        return False
    return float(d) <= d_thresh


def _qc_min_Dabs(period_df: pd.DataFrame) -> Tuple[int, float, float]:
    """
    Return qc, Dmin, Dsecond based on D_abs (mean per q already).
    Ties: choose smallest q among minima.
    """
    sdf = period_df.sort_values("ion_charge")
    d = sdf["D_abs"].to_numpy(dtype=float)
    q = sdf["ion_charge"].to_numpy(dtype=int)

    # ignore NaNs
    mask = np.isfinite(d)
    d = d[mask]
    q = q[mask]

    if len(q) == 0:
        raise ValueError("No finite D_abs values.")

    idx_min = np.where(d == np.min(d))[0]
    qc = int(np.min(q[idx_min]))
    Dmin = float(np.min(d))

    # second smallest (strict)
    d_sorted = np.sort(d)
    if len(d_sorted) >= 2:
        Dsecond = float(d_sorted[1])
    else:
        Dsecond = float("nan")

    return qc, Dmin, Dsecond


def _q_threshold_any(period_df: pd.DataFrame, pred) -> Optional[int]:
    sdf = period_df.sort_values("ion_charge")
    qs = []
    for _, r in sdf.iterrows():
        if pred(r):
            qs.append(int(r["ion_charge"]))
    return min(qs) if qs else None


def _q_threshold_majority(period_df: pd.DataFrame, pred) -> Optional[int]:
    """
    Smallest q such that among all rows with ion_charge >= q, a strict majority satisfy pred.
    """
    sdf = period_df.sort_values("ion_charge")
    charges = list(map(int, sdf["ion_charge"].tolist()))
    for q0 in charges:
        tail = sdf[sdf["ion_charge"] >= q0]
        if len(tail) == 0:
            continue
        frac = float(np.mean([bool(pred(r)) for _, r in tail.iterrows()]))
        if frac > 0.5:
            return int(q0)
    return None


def analyze_by_period(
    df: pd.DataFrame,
    *,
    p_lock_thresh: float,
    d_balanced_thresh: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (summary_df, long_df)
    """
    # Long form: mean D_abs per (period, ion_charge) already after dedupe.
    long_df = df[["period", "ion_charge", "D_abs", "state", "topology", "lock", "p_lock_D"]].copy()
    long_df = long_df.sort_values(["period", "ion_charge"]).reset_index(drop=True)

    results: list[QcResult] = []

    for period, g in long_df.groupby("period"):
        g = g.sort_values("ion_charge").reset_index(drop=True)

        qc, Dmin, Dsecond = _qc_min_Dabs(g)
        delta = float(Dsecond - Dmin) if np.isfinite(Dsecond) else float("nan")

        # Fractions (among available q points for this period)
        is_two = np.array([_is_two_lobe(r) for _, r in g.iterrows()], dtype=bool)
        is_lock = np.array([_is_locked(r, p_lock_thresh) for _, r in g.iterrows()], dtype=bool)
        frac_two = float(np.mean(is_two)) if len(is_two) else float("nan")
        frac_locked_two = float(np.mean(is_lock & is_two)) if len(is_two) else float("nan")

        q_min = int(g["ion_charge"].min())
        q_max = int(g["ion_charge"].max())
        n_q = int(g["ion_charge"].nunique())

        q_locked_any = _q_threshold_any(g, lambda r: _is_locked(r, p_lock_thresh) and _is_two_lobe(r))
        q_locked_maj = _q_threshold_majority(g, lambda r: _is_locked(r, p_lock_thresh) and _is_two_lobe(r))

        q_bal_any = _q_threshold_any(g, lambda r: _balanced(r, d_balanced_thresh))
        q_bal_maj = _q_threshold_majority(g, lambda r: _balanced(r, d_balanced_thresh))

        results.append(
            QcResult(
                period=int(period),
                qc=int(qc),
                Dmin=float(Dmin),
                Dsecond=float(Dsecond),
                delta_second_minus_min=float(delta),
                n_q=n_q,
                q_min=q_min,
                q_max=q_max,
                frac_locked_two_lobe=float(frac_locked_two),
                frac_two_lobe=float(frac_two),
                q_locked_any=q_locked_any,
                q_locked_majority=q_locked_maj,
                q_balanced_any=q_bal_any,
                q_balanced_majority=q_bal_maj,
            )
        )

    summary_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("period").reset_index(drop=True)
    return summary_df, long_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/derived/closure_phase_diagram_v2.csv", help="Input phase diagram (v2 recommended)")
    ap.add_argument("--out_csv", default="data/derived/closure_universal_by_period.csv", help="Output per-period qc summary CSV")
    ap.add_argument(
        "--out_long_csv",
        default="data/derived/closure_universal_by_period_long.csv",
        help="Output long-form per (period,q) means CSV",
    )
    ap.add_argument(
        "--p_lock_thresh",
        type=float,
        default=0.05,
        help="Treat (two_lobe) as locked if p_lock_D <= this (fallback if lock label missing)",
    )
    ap.add_argument(
        "--d_balanced_thresh",
        type=float,
        default=0.25,
        help="Balanced threshold: D_abs <= this is 'balanced'",
    )

    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df = _dedupe(df)

    summary_df, long_df = analyze_by_period(
        df,
        p_lock_thresh=float(args.p_lock_thresh),
        d_balanced_thresh=float(args.d_balanced_thresh),
    )

    summary_df.to_csv(args.out_csv, index=False)
    long_df.to_csv(args.out_long_csv, index=False)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_long_csv}")
    print("\n== qc by period (preview) ==")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
    