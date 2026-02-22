#!/usr/bin/env python3
"""
Fit "universal" curves for closure strength D_abs as a function of ion charge.

Primary purpose
---------------
This script summarizes D_abs by ion_charge (optionally within a subset of states),
then fits simple descriptive parametric families to the mean curve:
  - power-law-to-qc:     y ≈ A * |qc - q|^beta
  - exponential decay:   y ≈ A * exp(-k*(q - q0))
  - logistic (grid):     y ≈ D_inf + (D0 - D_inf)/(1 + exp(k*(q - qc)))

Notes / guardrails
------------------
- This is descriptive fitting, not a claim of true critical behavior.
- If you include P+/P- states, the mean curve can be dominated by D_abs=1 plateaus.
  Use --include_states "C,C*" to focus on nontrivial two-lobe regimes.
- The power-law model can produce a negative beta; evaluating at q==qc would yield
  0**negative -> inf. We explicitly floor |qc-q| by eps to avoid that.

Inputs / Outputs
----------------
Input:
  data/derived/closure_phase_diagram_v2.csv  (default)

Outputs:
  data/derived/closure_universal_Dabs_summary.csv
  data/derived/closure_universal_Dabs_fits.csv

Usage examples
--------------
  # Default (all rows)
  PYTHONPATH=. python3 scripts/fit_closure_universal_curve.py

  # Focus on nontrivial closure states only
  PYTHONPATH=. python3 scripts/fit_closure_universal_curve.py \
    --include_states "C,C*" \
    --out_summary_csv data/derived/closure_universal_Dabs_summary_C_only.csv \
    --out_fits_csv data/derived/closure_universal_Dabs_fits_C_only.csv

  # Exclude P states explicitly
  PYTHONPATH=. python3 scripts/fit_closure_universal_curve.py --exclude_states "P+,P-"

  # Dedupe by (ion_charge,period) to avoid double-counting multiple dataset variants
  PYTHONPATH=. python3 scripts/fit_closure_universal_curve.py --dedupe_on "ion_charge,period"
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    model: str
    params: Dict[str, float]
    sse: float
    n: int
    k_params: int

    @property
    def aic(self) -> float:
        # Gaussian SSE AIC up to constant: n*log(SSE/n) + 2k
        if self.n <= 0:
            return float("nan")
        if not np.isfinite(self.sse):
            return float("inf")
        if self.sse <= 0:
            return -float("inf")
        return float(self.n * math.log(self.sse / self.n) + 2 * self.k_params)

    @property
    def bic(self) -> float:
        if self.n <= 0:
            return float("nan")
        if not np.isfinite(self.sse):
            return float("inf")
        if self.sse <= 0:
            return -float("inf")
        return float(self.n * math.log(self.sse / self.n) + self.k_params * math.log(self.n))


def _ensure_cols(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Present: {list(df.columns)}")


def _parse_csv_list(s: str | None) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def summarize_universal_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build mean/std/count of D_abs vs ion_charge across all periods present.
    """
    _ensure_cols(df, ["ion_charge", "period", "D_abs"])
    g = (
        df.dropna(subset=["D_abs"])
        .groupby("ion_charge")["D_abs"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "mean": "D_abs_mean",
                "std": "D_abs_std",
                "min": "D_abs_min",
                "max": "D_abs_max",
            }
        )
        .sort_values("ion_charge")
    )
    return g


def _sse(y: np.ndarray, yhat: np.ndarray) -> float:
    r = y - yhat
    return float(np.sum(r * r))


def fit_power_to_qc(q: np.ndarray, y: np.ndarray, qc: float, eps: float = 1e-12) -> FitResult:
    """
    Fit y ≈ A * |qc - q|^beta using log-linear regression.
    Requires q != qc and y > 0 for fitting.

    For prediction (yhat) we floor |qc-q| by eps to avoid 0**negative when beta < 0.
    """
    dq = np.abs(qc - q)
    mask = (dq > 0) & np.isfinite(y) & (y > eps) & np.isfinite(dq) & np.isfinite(q)
    y2 = y[mask]
    dq2 = dq[mask]
    n_fit = int(len(y2))
    if n_fit < 2:
        return FitResult("power_to_qc", {"qc": float(qc), "A": float("nan"), "beta": float("nan")}, float("nan"), int(len(y)), 2)

    X = np.log(dq2)
    Y = np.log(y2)

    beta, logA = np.polyfit(X, Y, 1)
    A = float(math.exp(logA))

    dq_full = np.maximum(np.abs(qc - q), eps)
    yhat = A * (dq_full ** beta)

    sse = _sse(y, yhat)
    return FitResult("power_to_qc", {"qc": float(qc), "A": A, "beta": float(beta)}, sse, int(len(y)), 2)


def fit_exponential(q: np.ndarray, y: np.ndarray, q0: float, eps: float = 1e-12) -> FitResult:
    """
    Fit y ≈ A * exp(-k*(q-q0)), with fixed q0.
    Log-linear on y>0.
    """
    mask = np.isfinite(y) & (y > eps) & np.isfinite(q)
    q2 = q[mask]
    y2 = y[mask]
    n_fit = int(len(y2))
    if n_fit < 2:
        return FitResult("exponential", {"q0": float(q0), "A": float("nan"), "k": float("nan")}, float("nan"), int(len(y)), 2)

    X = (q2 - q0)
    Y = np.log(y2)
    # Y = logA - k*X  -> slope = -k
    slope, logA = np.polyfit(X, Y, 1)
    k = float(-slope)
    A = float(math.exp(logA))

    yhat = A * np.exp(-k * (q - q0))
    sse = _sse(y, yhat)
    return FitResult("exponential", {"q0": float(q0), "A": A, "k": k}, sse, int(len(y)), 2)


def fit_logistic_grid(
    q: np.ndarray,
    y: np.ndarray,
    *,
    qc_grid: np.ndarray,
    k_grid: np.ndarray,
    dinf_grid: np.ndarray,
) -> FitResult:
    """
    Fit y ≈ D_inf + (D0 - D_inf)/(1 + exp(k*(q - qc))).
    Solve for D0 in closed form for each (qc,k,D_inf) by least squares.

    Lightweight grid-search fit (no SciPy).
    """
    mask = np.isfinite(q) & np.isfinite(y)
    q2 = q[mask]
    y2 = y[mask]
    n = int(len(y2))
    if n < 3:
        return FitResult(
            "logistic_grid",
            {"qc": float("nan"), "k": float("nan"), "D_inf": float("nan"), "D0": float("nan")},
            float("nan"),
            n,
            4,
        )

    best: FitResult | None = None

    for qc in qc_grid:
        for k in k_grid:
            # f = 1/(1+exp(k*(q-qc)))
            f = 1.0 / (1.0 + np.exp(k * (q2 - qc)))

            ff = float(np.sum(f * f))
            if ff <= 0 or not np.isfinite(ff):
                continue

            for dinf in dinf_grid:
                # Model: y = dinf + (D0 - dinf)*f = dinf + f*D0 - f*dinf
                # Rearr: y - dinf + f*dinf = f*D0
                y_adj = (y2 - dinf + f * dinf)

                D0 = float(np.sum(f * y_adj) / ff)
                yhat2 = dinf + (D0 - dinf) * f
                sse2 = _sse(y2, yhat2)

                if (best is None) or (sse2 < best.sse):
                    best = FitResult(
                        "logistic_grid",
                        {"qc": float(qc), "k": float(k), "D_inf": float(dinf), "D0": float(D0)},
                        sse2,
                        n,
                        4,
                    )

    assert best is not None
    return best


def _apply_filters_and_dedupe(df: pd.DataFrame, *, include_states: List[str], exclude_states: List[str], dedupe_cols: List[str]) -> pd.DataFrame:
    # state filters
    if include_states:
        _ensure_cols(df, ["state"])
        df = df[df["state"].isin(include_states)].copy()

    if exclude_states:
        _ensure_cols(df, ["state"])
        df = df[~df["state"].isin(exclude_states)].copy()

    # dedupe (optional)
    if dedupe_cols:
        _ensure_cols(df, dedupe_cols)
        df = df.sort_values(dedupe_cols).drop_duplicates(subset=dedupe_cols, keep="first").copy()

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/derived/closure_phase_diagram_v2.csv")
    ap.add_argument("--out_summary_csv", default="data/derived/closure_universal_Dabs_summary.csv")
    ap.add_argument("--out_fits_csv", default="data/derived/closure_universal_Dabs_fits.csv")

    ap.add_argument("--include_states", default=None, help="Comma list to keep, e.g. 'C,C*'. If set, keep only these.")
    ap.add_argument("--exclude_states", default=None, help="Comma list to drop, e.g. 'P+,P-'. Applied after include.")
    ap.add_argument("--dedupe_on", default="", help="Comma list of columns to dedupe on (e.g. 'ion_charge,period'). Empty disables.")

    ap.add_argument("--d_floor", type=float, default=1e-9, help="Floor to avoid log(0) in log-fits.")
    ap.add_argument("--qc_power", type=float, default=5.0, help="Critical charge used in power-law-to-qc fit.")
    ap.add_argument("--q0_exp", type=float, default=1.0, help="Reference charge used in exponential fit.")

    # logistic grid controls (kept simple; no SciPy)
    ap.add_argument("--logistic_qc_steps", type=int, default=41, help="Number of qc grid points for logistic fit.")
    ap.add_argument("--logistic_k_min", type=float, default=0.1)
    ap.add_argument("--logistic_k_max", type=float, default=3.0)
    ap.add_argument("--logistic_k_steps", type=int, default=60)
    ap.add_argument("--logistic_dinf_min", type=float, default=0.0)
    ap.add_argument("--logistic_dinf_max", type=float, default=0.4)
    ap.add_argument("--logistic_dinf_steps", type=int, default=41)

    args = ap.parse_args()

    include_states = _parse_csv_list(args.include_states)
    exclude_states = _parse_csv_list(args.exclude_states)
    dedupe_cols = _parse_csv_list(args.dedupe_on)

    df = pd.read_csv(args.in_csv)
    _ensure_cols(df, ["ion_charge", "period", "D_abs"])

    df = _apply_filters_and_dedupe(df, include_states=include_states, exclude_states=exclude_states, dedupe_cols=dedupe_cols)

    summary = summarize_universal_curve(df)
    summary.to_csv(args.out_summary_csv, index=False)

    # Prepare fit arrays from mean curve
    q = summary["ion_charge"].astype(float).to_numpy()
    y = summary["D_abs_mean"].astype(float).to_numpy()

    # Apply floor for log-based fits (only affects fit, not summary)
    y_fit = np.maximum(y, float(args.d_floor))

    fits: List[FitResult] = []

    # 1) Power law to fixed qc
    fits.append(fit_power_to_qc(q, y_fit, qc=float(args.qc_power), eps=float(args.d_floor)))

    # 2) Exponential decay
    fits.append(fit_exponential(q, y_fit, q0=float(args.q0_exp), eps=float(args.d_floor)))

    # 3) Logistic (grid)
    qc_grid = np.linspace(float(np.min(q)), float(np.max(q)), int(args.logistic_qc_steps))
    k_grid = np.linspace(float(args.logistic_k_min), float(args.logistic_k_max), int(args.logistic_k_steps))
    dinf_grid = np.linspace(float(args.logistic_dinf_min), float(args.logistic_dinf_max), int(args.logistic_dinf_steps))
    fits.append(fit_logistic_grid(q, y, qc_grid=qc_grid, k_grid=k_grid, dinf_grid=dinf_grid))

    # Write fits table
    rows = []
    for fr in fits:
        row = {
            "model": fr.model,
            "n": fr.n,
            "k_params": fr.k_params,
            "sse": fr.sse,
            "aic": fr.aic,
            "bic": fr.bic,
        }
        for k, v in fr.params.items():
            row[f"param_{k}"] = v
        rows.append(row)

    fits_df = pd.DataFrame(rows).sort_values(["aic", "bic"], ascending=True)
    fits_df.to_csv(args.out_fits_csv, index=False)

    # Print report
    print(f"Wrote: {args.out_summary_csv}")
    if include_states or exclude_states or dedupe_cols:
        print("== filters ==")
        print(f"include_states={include_states if include_states else None}")
        print(f"exclude_states={exclude_states if exclude_states else None}")
        print(f"dedupe_on={dedupe_cols if dedupe_cols else None}")
        print()
    print(summary.to_string(index=False))
    print()
    print(f"Wrote: {args.out_fits_csv}")
    print(fits_df.to_string(index=False))

    # Extra: derived “beta” report for power law fit (if finite)
    pw = next((f for f in fits if f.model == "power_to_qc"), None)
    if pw is not None and np.isfinite(pw.params.get("beta", float("nan"))):
        qc = pw.params["qc"]
        beta = pw.params["beta"]
        A = pw.params["A"]
        print()
        print("== power-law-to-qc interpretation ==")
        print(f"Assuming qc = {qc:.6g}, fitted D_abs_mean ≈ A*|qc-q|^beta")
        print(f"A = {A:.6g}, beta = {beta:.6g}")
        print("Note: With integer q and few points, treat beta as descriptive, not a precise critical exponent.")


if __name__ == "__main__":
    main()
