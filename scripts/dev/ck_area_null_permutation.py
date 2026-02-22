#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gor_ck_witness.lobes import lobe_areas_from_residuals, summarize_lobes


Point = Tuple[float, float]


def caf_residuals_pblock(ies_p1_to_p6: Sequence[float]) -> List[float]:
    """
    CAF-style residuals used throughout this repo: subtract the line through (p^2, IE2) and (p^5, IE5),
    evaluated on p^1..p^6.

    Inputs are the 6 values for p^1..p^6 (ordered).
    """
    if len(ies_p1_to_p6) != 6:
        raise ValueError(f"Expected 6 IE values (p1..p6). Got {len(ies_p1_to_p6)}")

    # anchor points: p^2 and p^5 (1-indexed)
    x2, y2 = 2.0, float(ies_p1_to_p6[1])
    x5, y5 = 5.0, float(ies_p1_to_p6[4])
    m = (y5 - y2) / (x5 - x2)
    b = y2 - m * x2

    # residuals at x=1..6
    return [float(y) - (m * float(x) + b) for x, y in enumerate(ies_p1_to_p6, start=1)]


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (MAD), unscaled."""
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _infer_required_columns(df: pd.DataFrame) -> None:
    required = ["period", "p1", "p2", "p3", "p4", "p5", "p6"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Columns present: {list(df.columns)}")


@dataclass(frozen=True)
class CKFeatures:
    A_plus: float
    A_minus: float  # stored as negative or zero (per summarize_lobes convention)
    rho_A: float  # NaN if undefined
    D_norm: float  # signed imbalance in [-1,1] when denom>0 else NaN
    L_logratio: float  # log(A_plus/|A_minus|) when both >0 else NaN
    topology: str  # plus_only, minus_only, two_lobe, none
    valid_D: bool
    valid_L: bool


def ck_features_from_ies(ies: Sequence[float], *, eps_zero: float = 0.0, eps_log: float = 0.0) -> CKFeatures:
    """
    Compute lobe areas and derived invariants from IE values.

    - A_plus >= 0
    - A_minus <= 0 (or 0)
    - rho_A = A_plus / |A_minus| (NaN if A_minus==0)
    - D_norm = (A_plus - |A_minus|) / (A_plus + |A_minus|) (NaN if denom==0)
    - L_logratio = log((A_plus+eps_log)/( |A_minus|+eps_log )) (NaN unless both lobes exist)
    """
    xs = [1, 2, 3, 4, 5, 6]
    rs = caf_residuals_pblock(ies)

    lobes = lobe_areas_from_residuals(xs, rs, eps_zero=eps_zero)
    s = summarize_lobes(lobes)

    A_plus = float(s.get("A_plus", float("nan")))
    A_minus = float(s.get("A_minus", float("nan")))
    rho_A = float(s.get("rho_A", float("nan")))

    # Ensure numeric sanity
    if math.isnan(A_plus) or math.isnan(A_minus):
        # if summarize_lobes ever returns NaNs here, treat as "none"
        return CKFeatures(
            A_plus=float("nan"),
            A_minus=float("nan"),
            rho_A=float("nan"),
            D_norm=float("nan"),
            L_logratio=float("nan"),
            topology="none",
            valid_D=False,
            valid_L=False,
        )

    # We treat A_minus as negative or 0
    A_minus_abs = abs(A_minus)

    denom = A_plus + A_minus_abs
    if denom > 0.0:
        D_norm = (A_plus - A_minus_abs) / denom
        valid_D = True
    else:
        D_norm = float("nan")
        valid_D = False

    # L only defined when both lobes exist strictly
    if (A_plus > 0.0) and (A_minus_abs > 0.0):
        if eps_log > 0.0:
            L_logratio = math.log((A_plus + eps_log) / (A_minus_abs + eps_log))
        else:
            L_logratio = math.log(A_plus / A_minus_abs)
        valid_L = True
        topology = "two_lobe"
    elif A_plus > 0.0 and A_minus_abs == 0.0:
        L_logratio = float("nan")
        valid_L = False
        topology = "plus_only"
    elif A_plus == 0.0 and A_minus_abs > 0.0:
        L_logratio = float("nan")
        valid_L = False
        topology = "minus_only"
    else:
        L_logratio = float("nan")
        valid_L = False
        topology = "none"

    # rho_A may already be NaN for plus_only; keep it as returned
    return CKFeatures(
        A_plus=float(A_plus),
        A_minus=float(A_minus),
        rho_A=float(rho_A),
        D_norm=float(D_norm),
        L_logratio=float(L_logratio),
        topology=topology,
        valid_D=bool(valid_D),
        valid_L=bool(valid_L),
    )


def _parse_period_list(s: str) -> List[int]:
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with columns: period,p1..p6")
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_zero", type=float, default=0.0, help="Residual threshold for lobe segmentation")
    ap.add_argument(
        "--eps_log",
        type=float,
        default=0.0,
        help="Optional epsilon inside logratio to avoid inf. Keep 0.0 unless you want smoothing.",
    )
    ap.add_argument(
        "--anchor_periods",
        default="2,4",
        help="Comma-separated periods used to form an anchor (mean D and mean L over those periods, if present).",
    )
    ap.add_argument(
        "--out_csv",
        default="data/derived/ck_area_null_perm.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    _infer_required_columns(df)

    rng = np.random.default_rng(args.seed)

    df_sorted = df.sort_values("period").reset_index(drop=True)

    # Precompute observed features for all periods
    obs_by_period: Dict[int, CKFeatures] = {}
    for _, row in df_sorted.iterrows():
        period = int(row["period"])
        ies_obs = [float(row[f"p{i}"]) for i in range(1, 7)]
        obs_by_period[period] = ck_features_from_ies(ies_obs, eps_zero=args.eps_zero, eps_log=args.eps_log)

    # Anchor (D and L) from available anchor periods
    anchor_periods = _parse_period_list(args.anchor_periods)

    D_anchor_vals: List[float] = []
    L_anchor_vals: List[float] = []
    for p in anchor_periods:
        f = obs_by_period.get(p, None)
        if f is None:
            continue
        if f.valid_D and (not math.isnan(f.D_norm)):
            D_anchor_vals.append(float(f.D_norm))
        if f.valid_L and (not math.isnan(f.L_logratio)):
            L_anchor_vals.append(float(f.L_logratio))

    D_anchor = float(np.mean(D_anchor_vals)) if D_anchor_vals else float("nan")
    L_anchor = float(np.mean(L_anchor_vals)) if L_anchor_vals else float("nan")

    rows: List[Dict[str, object]] = []

    for _, row in df_sorted.iterrows():
        period = int(row["period"])
        ies_obs = [float(row[f"p{i}"]) for i in range(1, 7)]
        f_obs = obs_by_period[period]

        # Permutation ensembles
        D_perm: List[float] = []
        L_perm: List[float] = []
        Aplus_perm: List[float] = []
        Aminusabs_perm: List[float] = []
        topo_perm: List[str] = []

        for _k in range(int(args.nperm)):
            ies_perm = list(ies_obs)
            rng.shuffle(ies_perm)
            f = ck_features_from_ies(ies_perm, eps_zero=args.eps_zero, eps_log=args.eps_log)

            topo_perm.append(f.topology)

            # D exists whenever denom>0 (including one-lobe cases)
            if f.valid_D and (not math.isnan(f.D_norm)):
                D_perm.append(float(f.D_norm))

            # L exists only for true two-lobe
            if f.valid_L and (not math.isnan(f.L_logratio)):
                L_perm.append(float(f.L_logratio))

            # For topology-aware strength tests
            if not math.isnan(f.A_plus):
                Aplus_perm.append(float(f.A_plus))
            if not math.isnan(f.A_minus):
                Aminusabs_perm.append(float(abs(f.A_minus)))

        Dp = np.array(D_perm, dtype=float)
        Lp = np.array(L_perm, dtype=float)
        Ap = np.array(Aplus_perm, dtype=float)
        Am = np.array(Aminusabs_perm, dtype=float)

        # Existence probabilities (topology frequencies)
        topo_perm_arr = np.array(topo_perm, dtype=object)
        p_two_lobe = float(np.mean(topo_perm_arr == "two_lobe")) if topo_perm_arr.size else float("nan")
        p_plus_only = float(np.mean(topo_perm_arr == "plus_only")) if topo_perm_arr.size else float("nan")
        p_minus_only = float(np.mean(topo_perm_arr == "minus_only")) if topo_perm_arr.size else float("nan")

        # Old-style existence stats
        valid_fraction_D = float(Dp.size) / float(args.nperm)
        valid_fraction_L = float(Lp.size) / float(args.nperm)
        valid_count_D = int(Dp.size)
        valid_count_L = int(Lp.size)

        # Lock p-values (anchor-based), if anchor is defined and we have a usable observed value
        # Define distance in D-space and L-space
        if (not math.isnan(D_anchor)) and f_obs.valid_D and (not math.isnan(f_obs.D_norm)) and (Dp.size > 0):
            d_obs = abs(float(f_obs.D_norm) - D_anchor)
            d_perm = np.abs(Dp - D_anchor)
            p_lock_D = float(np.mean(d_perm <= d_obs))
        else:
            p_lock_D = float("nan")

        if (not math.isnan(L_anchor)) and f_obs.valid_L and (not math.isnan(f_obs.L_logratio)) and (Lp.size > 0):
            ell_obs = abs(float(f_obs.L_logratio) - L_anchor)
            ell_perm = np.abs(Lp - L_anchor)
            p_lock_L = float(np.mean(ell_perm <= ell_obs))
        else:
            p_lock_L = float("nan")

        # Topology-aware strength p-values:
        # - if plus_only: is A_plus unusually large under permutation?
        # - if minus_only: is |A_minus| unusually large under permutation?
        # - if two_lobe: keep p_lock_{D,L} as the primary "structure" test (rho-based tests can be added elsewhere)
        if f_obs.topology == "plus_only" and (Ap.size > 0) and (not math.isnan(f_obs.A_plus)):
            p_plus = float(np.mean(Ap >= float(f_obs.A_plus)))
        else:
            p_plus = float("nan")

        if f_obs.topology == "minus_only" and (Am.size > 0) and (not math.isnan(f_obs.A_minus)):
            p_minus = float(np.mean(Am >= float(abs(f_obs.A_minus))))
        else:
            p_minus = float("nan")

        # Derived imbalance magnitude (useful across topologies)
        D_abs_obs = float(abs(f_obs.D_norm)) if (f_obs.valid_D and (not math.isnan(f_obs.D_norm))) else float("nan")

        rows.append(
            {
                "period": int(period),
                "A_plus_obs": float(f_obs.A_plus),
                "A_minus_obs": float(f_obs.A_minus),
                "rho_A_obs": float(f_obs.rho_A),
                "D_norm_obs": float(f_obs.D_norm),
                "L_logratio_obs": float(f_obs.L_logratio),
                "D_abs_obs": float(D_abs_obs),
                "topology_obs": str(f_obs.topology),
                "D_anchor": float(D_anchor),
                "L_anchor": float(L_anchor),
                "valid_fraction_D": float(valid_fraction_D),
                "valid_fraction_L": float(valid_fraction_L),
                "valid_count_D": int(valid_count_D),
                "valid_count_L": int(valid_count_L),
                "D_perm_median": float(np.median(Dp)) if Dp.size else float("nan"),
                "D_perm_MAD": float(_mad(Dp)) if Dp.size else float("nan"),
                "L_perm_median": float(np.median(Lp)) if Lp.size else float("nan"),
                "L_perm_MAD": float(_mad(Lp)) if Lp.size else float("nan"),
                "p_two_lobe": float(p_two_lobe),
                "p_plus_only": float(p_plus_only),
                "p_minus_only": float(p_minus_only),
                "p_lock_D": float(p_lock_D),
                "p_lock_L": float(p_lock_L),
                "p_plus": float(p_plus),
                "p_minus": float(p_minus),
            }
        )

    out = pd.DataFrame(rows).sort_values("period")
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
