#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _is_locked(lock_val: object) -> bool:
    if lock_val is None:
        return False
    s = str(lock_val).strip().lower()
    return s == "locked"


def _is_two_lobe_state(state: object) -> bool:
    if state is None:
        return False
    s = str(state).strip()
    return s in ("C*", "C")


def _to_int(x: object) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _to_float(x: object) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _first_q_where(df: pd.DataFrame, pred) -> Optional[int]:
    for _, r in df.iterrows():
        if pred(r):
            q = _to_int(r["ion_charge"])
            if q is not None:
                return q
    return None


def summarize_period(df_period: pd.DataFrame, *, d_thresh: float) -> Dict[str, object]:
    """
    df_period: rows for a fixed period, sorted by ion_charge
    """
    dfp = df_period.sort_values("ion_charge").reset_index(drop=True)

    q_two_lobe = _first_q_where(dfp, lambda r: _is_two_lobe_state(r.get("state")))
    q_locked = _first_q_where(
        dfp, lambda r: _is_two_lobe_state(r.get("state")) and _is_locked(r.get("lock"))
    )
    q_unlocked_two_lobe = _first_q_where(
        dfp, lambda r: _is_two_lobe_state(r.get("state")) and (not _is_locked(r.get("lock")))
    )
    q_balanced = _first_q_where(
        dfp,
        lambda r: (_to_float(r.get("D_abs")) is not None)
        and (_to_float(r.get("D_abs")) <= d_thresh),
    )

    # also record the minimal p_lock_D among locked rows, as a "lock strength" proxy
    locked_rows = dfp[dfp["lock"].map(_is_locked)]
    p_lock_min = None
    if not locked_rows.empty and "p_lock_D" in locked_rows.columns:
        vals = [v for v in locked_rows["p_lock_D"].tolist() if not pd.isna(v)]
        if vals:
            p_lock_min = float(min(vals))

    return {
        "period": _to_int(dfp["period"].iloc[0]),
        "q_two_lobe": q_two_lobe,
        "q_locked": q_locked,
        "q_unlocked_two_lobe": q_unlocked_two_lobe,
        "q_balanced_Dabs_le": q_balanced,
        "d_thresh": float(d_thresh),
        "p_lock_D_min_among_locked": p_lock_min,
        "n_rows": int(len(dfp)),
        "q_min": _to_int(dfp["ion_charge"].min()),
        "q_max": _to_int(dfp["ion_charge"].max()),
    }


def summarize_all(df: pd.DataFrame, *, d_thresh: float) -> Dict[str, object]:
    dfa = df.sort_values(["period", "ion_charge"]).reset_index(drop=True)

    # We define "global" critical charges as the *first q where any period satisfies condition*,
    # plus also the first q where a *majority* of periods satisfy it (when periods are comparable).
    periods = sorted({int(p) for p in dfa["period"].dropna().unique().tolist()})

    def frac_at_q(q: int, pred) -> float:
        sub = dfa[dfa["ion_charge"] == q]
        if sub.empty:
            return float("nan")
        # ensure exactly one row per period if present
        ok = 0
        tot = 0
        for p in periods:
            rowp = sub[sub["period"] == p]
            if rowp.empty:
                continue
            tot += 1
            if pred(rowp.iloc[0].to_dict()):
                ok += 1
        if tot == 0:
            return float("nan")
        return ok / tot

    q_values = sorted({int(q) for q in dfa["ion_charge"].dropna().unique().tolist()})

    def first_any(pred) -> Optional[int]:
        for q in q_values:
            f = frac_at_q(q, pred)
            if not math.isnan(f) and f > 0:
                return q
        return None

    def first_majority(pred) -> Optional[int]:
        for q in q_values:
            f = frac_at_q(q, pred)
            if not math.isnan(f) and f >= 0.5:
                return q
        return None

    pred_two_lobe = lambda r: _is_two_lobe_state(r.get("state"))
    pred_locked = lambda r: _is_two_lobe_state(r.get("state")) and _is_locked(r.get("lock"))
    pred_balanced = lambda r: (_to_float(r.get("D_abs")) is not None) and (_to_float(r.get("D_abs")) <= d_thresh)

    return {
        "scope": "ALL",
        "q_two_lobe_any": first_any(pred_two_lobe),
        "q_two_lobe_majority": first_majority(pred_two_lobe),
        "q_locked_any": first_any(pred_locked),
        "q_locked_majority": first_majority(pred_locked),
        "q_balanced_any": first_any(pred_balanced),
        "q_balanced_majority": first_majority(pred_balanced),
        "d_thresh": float(d_thresh),
        "periods_present": ",".join(str(p) for p in periods),
        "q_values_present": ",".join(str(q) for q in q_values),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="data/derived/closure_phase_diagram_v2.csv",
        help="Input table with header: ion_charge,period,dataset,topology,lock,rho_A,A_plus,A_minus,p_lock_D,state,D_abs",
    )
    ap.add_argument("--out_csv", default="data/derived/closure_critical_charges_by_period.csv")
    ap.add_argument("--out_all_csv", default="data/derived/closure_critical_charges_all.csv")
    ap.add_argument("--d_thresh", type=float, default=0.25, help="Balanced threshold for D_abs (<=).")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    required = ["ion_charge", "period", "lock", "state", "D_abs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Columns present: {list(df.columns)}")

    # per period
    rows: List[Dict[str, object]] = []
    for period in sorted({int(p) for p in df["period"].dropna().unique().tolist()}):
        dfp = df[df["period"] == period]
        rows.append(summarize_period(dfp, d_thresh=args.d_thresh))

    out = pd.DataFrame(rows).sort_values("period")
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")
    print(out.to_string(index=False))

    # all periods
    all_row = summarize_all(df, d_thresh=args.d_thresh)
    out_all = pd.DataFrame([all_row])
    out_all.to_csv(args.out_all_csv, index=False)
    print(f"\nWrote: {args.out_all_csv}")
    print(out_all.to_string(index=False))


if __name__ == "__main__":
    main()
