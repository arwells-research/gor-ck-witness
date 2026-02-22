#!/usr/bin/env python3
"""
State-conditioned collapse + mixture-vs-within-state tests for closure phase data.

Goal
----
Quantify whether the apparent qc-centered "universal curve" in D_abs is mostly:
  (a) a change in mixture weights over discrete states (P+, P-, C*, C), versus
  (b) a smooth dependence of D_abs on |q - qc| within each state.

This script produces the original collapse and mixture tables, and adds explicit tests:

1) Mixture-only prediction (global):
      D_mix(abs_delta) = sum_s P(s|abs_delta) * mean(D_abs | state=s)
   Compare to observed mean D_abs by abs_delta (weighted RMSE/MAE/R^2 over bins).

2) Mixture-only prediction (by period):
      D_mix(abs_delta, period) = sum_s P(s|abs_delta, period) * mean(D_abs | state=s, period)
   Compare to observed mean D_abs by (period, abs_delta) (weighted RMSE/MAE/R^2 per period).

3) Within-state dependence test:
   For each state, test whether D_abs depends on abs_delta within the state.
   Default uses a permutation test on OLS slope (two-sided), with finite-sample correction
   to avoid p=0 under finite permutations.

Inputs
------
Default expects:
  data/derived/closure_phase_diagram_v2.csv
with columns (at minimum):
  ion_charge, period, state, D_abs
Optional:
  data/derived/closure_universal_by_period.csv
with columns:
  period, qc

If qc_by_period_csv is provided, qc is taken per period (fallback to --qc where missing),
unless --require_qc_by_period is set (then missing qc is a hard error).

Outputs
-------
Original:
- data/derived/closure_collapse_by_absdelta.csv
- data/derived/closure_collapse_by_absdelta_state.csv
- data/derived/closure_state_mixture_by_absdelta.csv
- data/derived/closure_state_mixture_by_absdelta_period.csv

New:
- data/derived/closure_mixture_only_prediction.csv
- data/derived/closure_mixture_only_prediction_by_period.csv
- data/derived/closure_mixture_only_prediction_by_period_scores.csv
- data/derived/closure_within_state_dependence.csv
- data/derived/closure_qc_assignment_report.csv

Dedupe policy (important)
-------------------------
Your build step can introduce duplicate (period, ion_charge) rows because multiple
raw datasets cover overlapping period ranges (e.g. *_2to6 and *_3to6).

Modes
-----
strict:
  Do NOT silently dedupe. Instead FAIL if any (period, ion_charge) duplicates exist.
  This is the safest mode when you want upstream canonicalization.

prefer_dataset:
  Keep exactly ONE row per (period, ion_charge) via deterministic tie-break rules:
    1) optionally prefer dataset strings containing --prefer_dataset_contains (if provided)
    2) then stable lexical ordering of dataset (and other columns as secondary)
  This prevents silent double-counting even when float columns differ across overlaps.

Usage
-----
  PYTHONPATH=. python3 scripts/closure_state_conditioned_collapse.py

  # Use per-period qc values (recommended if available):
  PYTHONPATH=. python3 scripts/closure_state_conditioned_collapse.py \
    --qc_by_period_csv data/derived/closure_universal_by_period.csv

  # Force a global qc:
  PYTHONPATH=. python3 scripts/closure_state_conditioned_collapse.py --qc 5

  # Enforce canonicalization (fail on duplicates):
  PYTHONPATH=. python3 scripts/closure_state_conditioned_collapse.py --dedupe_mode strict

  # Deterministic dedupe by dataset preference:
  PYTHONPATH=. python3 scripts/closure_state_conditioned_collapse.py \
    --dedupe_mode prefer_dataset \
    --prefer_dataset_contains "_3to6"

  # Increase permutation trials for within-state slope test:
  PYTHONPATH=. python3 scripts/closure_state_conditioned_collapse.py --nperm_within_state 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# -------------------------
# Utilities / stats helpers
# -------------------------

def _require_cols(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Have={list(df.columns)}")


def _compute_d_abs(df: pd.DataFrame) -> pd.Series:
    # Prefer an explicit D_abs if present; otherwise fall back to abs(D_norm_obs) if available.
    if "D_abs" in df.columns:
        return pd.to_numeric(df["D_abs"], errors="coerce")
    if "D_norm_obs" in df.columns:
        return pd.to_numeric(df["D_norm_obs"], errors="coerce").abs()
    raise ValueError("Need either D_abs or D_norm_obs in input CSV to compute D_abs.")


def _load_qc_by_period(path: str) -> dict[int, int]:
    df = pd.read_csv(path)
    _require_cols(df, ["period", "qc"], name="qc_by_period_csv")
    df = df.dropna(subset=["period", "qc"]).copy()
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype(int)
    df["qc"] = pd.to_numeric(df["qc"], errors="coerce").astype(int)
    return dict(zip(df["period"].tolist(), df["qc"].tolist()))


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _summary_stats(g: pd.DataFrame, value_col: str) -> pd.Series:
    x = pd.to_numeric(g[value_col], errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return pd.Series(
            {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
                "mad": np.nan,
            }
        )
    med = float(np.median(x))
    return pd.Series(
        {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if x.size >= 2 else 0.0,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "median": med,
            "mad": float(np.median(np.abs(x - med))),
        }
    )


def _rmse(a: np.ndarray, b: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if w is None:
        if not np.any(m):
            return float("nan")
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
    wm = m & np.isfinite(w) & (w > 0)
    if not np.any(wm):
        return float("nan")
    ww = w[wm].astype(float)
    r2 = (a[wm] - b[wm]) ** 2
    return float(np.sqrt(np.sum(ww * r2) / np.sum(ww)))


def _mae(a: np.ndarray, b: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if w is None:
        if not np.any(m):
            return float("nan")
        return float(np.mean(np.abs(a[m] - b[m])))
    wm = m & np.isfinite(w) & (w > 0)
    if not np.any(wm):
        return float("nan")
    ww = w[wm].astype(float)
    r1 = np.abs(a[wm] - b[wm])
    return float(np.sum(ww * r1) / np.sum(ww))


def _r2_weighted(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    ww = w[m].astype(float)
    y0 = y[m].astype(float)
    yhat0 = yhat[m].astype(float)
    mu = float(np.sum(ww * y0) / np.sum(ww))
    sse = float(np.sum(ww * (y0 - yhat0) ** 2))
    sst = float(np.sum(ww * (y0 - mu) ** 2))
    if sst <= 0:
        return float("nan")
    return float(1.0 - sse / sst)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return slope of y ~ a + b x via least squares, ignoring non-finite.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x2 = x[m].astype(float)
    y2 = y[m].astype(float)
    if x2.size < 2:
        return float("nan")
    x0 = x2 - np.mean(x2)
    denom = float(np.sum(x0 * x0))
    if denom <= 0:
        return float("nan")
    b = float(np.sum(x0 * (y2 - np.mean(y2))) / denom)
    return b


# -------------------------
# Dedupe policy
# -------------------------

def _dedupe_rows(
    df: pd.DataFrame,
    *,
    mode: str,
    prefer_contains: Optional[str] = None,
) -> tuple[pd.DataFrame, int, int]:
    """
    Handle duplicate (period, ion_charge) rows.

    strict:
      Fail if duplicates exist.

    prefer_dataset:
      Keep one row per (period, ion_charge) with deterministic tie-break:
        - prefer dataset strings containing prefer_contains (if provided)
        - then stable lexical ordering of dataset, then optional columns as secondary
    """
    before = int(len(df))

    if mode not in {"strict", "prefer_dataset"}:
        raise ValueError(f"Unknown dedupe_mode={mode}. Choose strict or prefer_dataset")

    key = ["period", "ion_charge"]

    if mode == "strict":
        dup = df.duplicated(subset=key, keep=False)
        if dup.any():
            cols = key + (["dataset"] if "dataset" in df.columns else [])
            ex = df.loc[dup, cols].copy()
            ex = ex.sort_values(key + (["dataset"] if "dataset" in ex.columns else []), kind="mergesort")
            msg = (
                "Duplicates detected under --dedupe_mode strict for (period, ion_charge).\n"
                "Either fix upstream canonicalization OR rerun with --dedupe_mode prefer_dataset.\n\n"
                f"{ex.to_string(index=False)}"
            )
            raise ValueError(msg)
        after = before
        return df.copy(), before, after

    # prefer_dataset mode
    if "dataset" not in df.columns:
        # No dataset to tie-break on -> fall back to a stable ordering and drop duplicates
        df2 = df.sort_values(key, kind="mergesort").drop_duplicates(subset=key, keep="first").copy()
        return df2, before, int(len(df2))

    df2 = df.copy()

    # preference flag: preferred datasets come first
    if prefer_contains is not None and str(prefer_contains) != "":
        df2["_prefer"] = df2["dataset"].astype(str).str.contains(str(prefer_contains), regex=False).astype(int)
    else:
        df2["_prefer"] = 0

    # Stable tie-break: prefer flag (desc), then dataset (asc), then a few optional columns
    sort_cols = ["period", "ion_charge", "_prefer", "dataset"]
    # Put preferred first => sort by _prefer descending. We'll accomplish by sorting ascending on period/ion_charge,
    # then descending on _prefer using a helper by negation.
    df2["_prefer_neg"] = -df2["_prefer"]

    sort_cols2 = ["period", "ion_charge", "_prefer_neg", "dataset"]
    for c in ["topology", "lock", "state"]:
        if c in df2.columns:
            sort_cols2.append(c)

    df2 = df2.sort_values(sort_cols2, kind="mergesort")
    df2 = df2.drop_duplicates(subset=key, keep="first").copy()
    df2 = df2.drop(columns=["_prefer", "_prefer_neg"], errors="ignore")
    after = int(len(df2))
    return df2, before, after


# -------------------------
# Within-state dependence test
# -------------------------

def _within_state_slope_perm_test(
    df: pd.DataFrame,
    *,
    state: str,
    nperm: int,
    seed: int,
) -> Dict[str, float]:
    """
    Permutation test for whether slope of D_abs vs abs_delta differs from 0 within a state.
    We permute abs_delta among rows (within state), recompute slope, and compute a two-sided p-value.

    Finite-sample correction:
      p = (k + 1) / (nperm + 1) where k is #perm slopes with |b*| >= |b_obs|
    """
    g = df[df["state"] == state].copy()
    x = g["abs_delta"].to_numpy(dtype=float)
    y = g["D_abs"].to_numpy(dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    out: Dict[str, float] = {
        "state": state,
        "n": float(x.size),
        "n_unique_abs_delta": float(np.unique(x).size),
        "slope_obs": float("nan"),
        "slope_perm_median": float("nan"),
        "slope_perm_mad": float("nan"),
        "p_slope_two_sided": float("nan"),
    }

    # Need enough data and at least 2 distinct abs_delta values
    if x.size < 3 or np.unique(x).size < 2:
        return out

    slope_obs = _ols_slope(x, y)
    out["slope_obs"] = float(slope_obs)

    rng = np.random.default_rng(int(seed))
    slopes = np.empty(int(nperm), dtype=float)
    for i in range(int(nperm)):
        x_perm = rng.permutation(x)
        slopes[i] = _ols_slope(x_perm, y)

    out["slope_perm_median"] = float(np.median(slopes))
    out["slope_perm_mad"] = _mad(slopes)

    if np.isfinite(slope_obs):
        k = int(np.sum(np.abs(slopes) >= abs(slope_obs)))
        out["p_slope_two_sided"] = float((k + 1) / (slopes.size + 1))
    return out


# -------------------------
# qc assignment reporting
# -------------------------

def _qc_assignment_report(
    df: pd.DataFrame,
    qc_source: str,
    qc_map: Optional[Dict[int, int]],
    qc_fallback: int,
) -> pd.DataFrame:
    """
    Report qc used per period and whether it came from map or fallback.
    """
    periods = sorted(df["period"].unique().tolist())
    rows = []
    for p in periods:
        qc_used = int(df.loc[df["period"] == p, "qc"].iloc[0])
        source_mode = "global"
        if qc_map is not None:
            source_mode = "map" if p in qc_map else "fallback"
        rows.append(
            {
                "period": int(p),
                "qc_used": qc_used,
                "qc_source": qc_source,
                "qc_source_mode": source_mode,
                "qc_fallback": int(qc_fallback),
            }
        )
    return pd.DataFrame(rows).sort_values("period")


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/derived/closure_phase_diagram_v2.csv")
    ap.add_argument("--qc", type=int, default=5)
    ap.add_argument("--qc_by_period_csv", default=None)
    ap.add_argument(
        "--require_qc_by_period",
        action="store_true",
        help="Fail if any period in data lacks qc in qc_by_period_csv.",
    )
    ap.add_argument("--out_dir", default="data/derived")

    ap.add_argument(
        "--dedupe_mode",
        default="prefer_dataset",
        choices=["strict", "prefer_dataset"],
        help="Duplicate handling. strict fails on duplicates; prefer_dataset keeps one row per (period,ion_charge).",
    )
    ap.add_argument(
        "--prefer_dataset_contains",
        default=None,
        help="If set and dedupe_mode=prefer_dataset, prefer rows whose dataset contains this substring.",
    )

    ap.add_argument(
        "--period_filter",
        type=int,
        default=None,
        help="If set, restrict analysis to a single period (e.g., --period_filter 6).",
    )

    ap.add_argument(
        "--nperm_within_state",
        type=int,
        default=5000,
        help="Permutation trials for within-state slope test.",
    )
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # optional period restriction (apply early)
    if args.period_filter is not None:
        if "period" not in df.columns:
            raise ValueError("Input CSV missing required column 'period' for --period_filter.")
        df = df[df["period"].astype(int) == int(args.period_filter)].copy()
        if len(df) == 0:
            raise ValueError(f"--period_filter {args.period_filter} left zero rows; check input data/period values.")

    # Normalize types
    _require_cols(df, ["ion_charge", "period"], name="phase_diagram_v2")
    df["ion_charge"] = pd.to_numeric(df["ion_charge"], errors="coerce").astype("Int64")
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")

    # State handling
    if "state" not in df.columns:
        df["state"] = "UNKNOWN"
    df["state"] = df["state"].astype(str)

    # D_abs
    df["D_abs"] = _compute_d_abs(df)

    # Drop rows missing essentials
    df = df.dropna(subset=["ion_charge", "period", "D_abs"]).copy()
    df["ion_charge"] = df["ion_charge"].astype(int)
    df["period"] = df["period"].astype(int)

    # Dedupe overlapping datasets
    df, before, after = _dedupe_rows(
        df,
        mode=str(args.dedupe_mode),
        prefer_contains=args.prefer_dataset_contains,
    )

    # qc assignment
    qc_map: Optional[Dict[int, int]] = None
    if args.qc_by_period_csv is not None:
        qc_map = _load_qc_by_period(args.qc_by_period_csv)
        missing_periods = sorted(set(df["period"].unique().tolist()) - set(qc_map.keys()))
        if args.require_qc_by_period and len(missing_periods) > 0:
            raise ValueError(
                f"--require_qc_by_period set, but qc_by_period_csv lacks qc for periods: {missing_periods}"
            )
        df["qc"] = df["period"].map(qc_map)
        df["qc"] = df["qc"].fillna(args.qc).astype(int)
        qc_source = f"per-period qc from {args.qc_by_period_csv} (fallback qc={args.qc})"
    else:
        df["qc"] = int(args.qc)
        qc_source = f"global qc={args.qc}"

    df["abs_delta"] = (df["ion_charge"] - df["qc"]).abs().astype(int)

    # --- qc assignment report ---
    qc_rep = _qc_assignment_report(df, qc_source, qc_map, args.qc)
    out_qc_rep = out_dir / "closure_qc_assignment_report.csv"
    qc_rep.to_csv(out_qc_rep, index=False)

    # --- Aggregate collapse by abs_delta ---
    agg = (
        df.groupby("abs_delta", as_index=False)
        .apply(lambda g: _summary_stats(g, "D_abs"))
        .reset_index(drop=True)
        .sort_values("abs_delta")
    )
    agg.insert(1, "qc_source", qc_source)

    # --- State-conditioned collapse by abs_delta,state ---
    agg_state = (
        df.groupby(["abs_delta", "state"], as_index=False)
        .apply(lambda g: _summary_stats(g, "D_abs"))
        .reset_index(drop=True)
        .sort_values(["abs_delta", "state"])
    )
    agg_state.insert(1, "qc_source", qc_source)

    # --- Mixture weights P(state | abs_delta) ---
    mix = (
        df.groupby(["abs_delta", "state"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["abs_delta", "state"])
    )
    totals = mix.groupby("abs_delta", as_index=False)["count"].sum().rename(columns={"count": "count_total"})
    mix = mix.merge(totals, on="abs_delta", how="left")
    mix["p_state_given_absdelta"] = mix["count"] / mix["count_total"]
    mix.insert(1, "qc_source", qc_source)

    # --- Mixture weights P(state | abs_delta, period) ---
    mix_period = (
        df.groupby(["period", "abs_delta", "state"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["period", "abs_delta", "state"])
    )
    totals_p = (
        mix_period.groupby(["period", "abs_delta"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "count_total"})
    )
    mix_period = mix_period.merge(totals_p, on=["period", "abs_delta"], how="left")
    mix_period["p_state_given_absdelta_period"] = mix_period["count"] / mix_period["count_total"]
    mix_period.insert(1, "qc_source", qc_source)

    # ---------------------------
    # NEW: Mixture-only prediction (global)
    # ---------------------------
    state_means = (
        df.groupby("state", as_index=False)["D_abs"]
        .mean()
        .rename(columns={"D_abs": "D_state_mean"})
    )

    mix2 = mix.merge(state_means, on="state", how="left")
    mix2["contrib"] = mix2["p_state_given_absdelta"] * mix2["D_state_mean"]

    pred = (
        mix2.groupby("abs_delta", as_index=False)
        .agg(D_mix_pred=("contrib", "sum"), n=("count_total", "first"))
        .sort_values("abs_delta")
    )

    obs_mean = (
        df.groupby("abs_delta", as_index=False)
        .agg(D_obs_mean=("D_abs", "mean"), n_obs=("D_abs", "size"))
        .sort_values("abs_delta")
    )

    pred = pred.merge(obs_mean[["abs_delta", "D_obs_mean"]], on="abs_delta", how="left")
    pred["residual_obs_minus_mix"] = pred["D_obs_mean"] - pred["D_mix_pred"]

    w_bins = pred["n"].to_numpy(dtype=float)
    y_obs = pred["D_obs_mean"].to_numpy(dtype=float)
    y_hat = pred["D_mix_pred"].to_numpy(dtype=float)

    rmse_mix = _rmse(y_obs, y_hat, w_bins)
    mae_mix = _mae(y_obs, y_hat, w_bins)
    r2_mix = _r2_weighted(y_obs, y_hat, w_bins)

    pred.insert(1, "qc_source", qc_source)
    pred.insert(2, "rmse_weighted_over_absdelta", rmse_mix)
    pred.insert(3, "mae_weighted_over_absdelta", mae_mix)
    pred.insert(4, "r2_weighted_over_absdelta", r2_mix)

    # ---------------------------
    # NEW: Mixture-only prediction (by period)
    # ---------------------------
    state_means_p = (
        df.groupby(["period", "state"], as_index=False)["D_abs"]
        .mean()
        .rename(columns={"D_abs": "D_state_mean_period"})
    )

    mixp2 = mix_period.merge(state_means_p, on=["period", "state"], how="left")

    # If some (period,state) lacks mean, fall back to global state mean
    mixp2 = mixp2.merge(state_means, on="state", how="left")
    mixp2["D_state_mean_period"] = mixp2["D_state_mean_period"].fillna(mixp2["D_state_mean"])
    mixp2["contrib"] = mixp2["p_state_given_absdelta_period"] * mixp2["D_state_mean_period"]

    pred_p = (
        mixp2.groupby(["period", "abs_delta"], as_index=False)
        .agg(D_mix_pred=("contrib", "sum"), n=("count_total", "first"))
        .sort_values(["period", "abs_delta"])
    )

    obs_p = (
        df.groupby(["period", "abs_delta"], as_index=False)
        .agg(D_obs_mean=("D_abs", "mean"), n_obs=("D_abs", "size"))
        .sort_values(["period", "abs_delta"])
    )

    pred_p = pred_p.merge(obs_p[["period", "abs_delta", "D_obs_mean"]], on=["period", "abs_delta"], how="left")
    pred_p["residual_obs_minus_mix"] = pred_p["D_obs_mean"] - pred_p["D_mix_pred"]
    pred_p.insert(1, "qc_source", qc_source)

    # Per-period scores (weighted over abs_delta bins within each period)
    rows_scores = []
    for p, gp in pred_p.groupby("period"):
        y_obs_p = gp["D_obs_mean"].to_numpy(dtype=float)
        y_hat_p = gp["D_mix_pred"].to_numpy(dtype=float)
        w_p = gp["n"].to_numpy(dtype=float)
        rmse_p = _rmse(y_obs_p, y_hat_p, w_p)
        mae_p = _mae(y_obs_p, y_hat_p, w_p)
        r2_p = _r2_weighted(y_obs_p, y_hat_p, w_p)
        rows_scores.append(
            {
                "period": int(p),
                "rmse_weighted": rmse_p,
                "mae_weighted": mae_p,
                "r2_weighted": r2_p,
                "n_bins": int(len(gp)),
                "n_total_rows": int(np.sum(gp["n"].to_numpy(dtype=float))),
            }
        )
    scores_p = pd.DataFrame(rows_scores).sort_values("period")

    # ---------------------------
    # NEW: Within-state dependence test (slope permutation)
    # ---------------------------
    # To avoid any subtle coupling across states, give each state its own derived seed.
    base_rng = np.random.default_rng(int(args.seed))
    states = sorted(df["state"].unique().tolist())
    within_rows = []
    for s in states:
        state_seed = int(base_rng.integers(0, 2**31 - 1))
        within_rows.append(
            _within_state_slope_perm_test(
                df,
                state=s,
                nperm=int(args.nperm_within_state),
                seed=state_seed,
            )
        )
    within = pd.DataFrame(within_rows)
    within.insert(0, "qc_source", qc_source)
    within.insert(1, "nperm_within_state", int(args.nperm_within_state))
    within.insert(2, "seed", int(args.seed))

    # Write outputs (original + new)
    out_agg = out_dir / "closure_collapse_by_absdelta.csv"
    out_agg_state = out_dir / "closure_collapse_by_absdelta_state.csv"
    out_mix = out_dir / "closure_state_mixture_by_absdelta.csv"
    out_mix_period = out_dir / "closure_state_mixture_by_absdelta_period.csv"

    out_pred = out_dir / "closure_mixture_only_prediction.csv"
    out_pred_p = out_dir / "closure_mixture_only_prediction_by_period.csv"
    out_within = out_dir / "closure_within_state_dependence.csv"
    out_scores_p = out_dir / "closure_mixture_only_prediction_by_period_scores.csv"

    agg.to_csv(out_agg, index=False)
    agg_state.to_csv(out_agg_state, index=False)
    mix.to_csv(out_mix, index=False)
    mix_period.to_csv(out_mix_period, index=False)

    pred.to_csv(out_pred, index=False)
    pred_p.to_csv(out_pred_p, index=False)
    within.to_csv(out_within, index=False)
    scores_p.to_csv(out_scores_p, index=False)
    qc_rep.to_csv(out_qc_rep, index=False)

    # Print concise previews
    print(f"Wrote: {out_agg}")
    print(f"Wrote: {out_agg_state}")
    print(f"Wrote: {out_mix}")
    print(f"Wrote: {out_mix_period}")
    print(f"Wrote: {out_pred}")
    print(f"Wrote: {out_pred_p}")
    print(f"Wrote: {out_scores_p}")
    print(f"Wrote: {out_within}")
    print(f"Wrote: {out_qc_rep}")
    print()

    print("== dedupe ==")
    print(f"mode={args.dedupe_mode} rows_before={before} rows_after={after}")
    if args.dedupe_mode == "prefer_dataset" and args.prefer_dataset_contains:
        print(f"prefer_dataset_contains={args.prefer_dataset_contains}")
    print()

    print("== qc assignment report ==")
    print(qc_rep.to_string(index=False))
    print()

    print("== collapse by abs_delta (D_abs) ==")
    print(agg.to_string(index=False))
    print()

    print("== mixture-only prediction (global) ==")
    print(f"(weighted RMSE over abs_delta bins) {rmse_mix:.6g}")
    print(f"(weighted MAE  over abs_delta bins) {mae_mix:.6g}")
    print(f"(weighted R^2  over abs_delta bins) {r2_mix:.6g}")
    print(pred[["abs_delta", "D_obs_mean", "D_mix_pred", "residual_obs_minus_mix", "n"]].to_string(index=False))
    print()

    print("== mixture-only prediction scores (by period) ==")
    print(scores_p.to_string(index=False))
    print()

    print("== within-state dependence (slope permutation test) ==")
    show_cols = ["state", "n", "n_unique_abs_delta", "slope_obs", "slope_perm_median", "slope_perm_mad", "p_slope_two_sided"]
    print(within[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
