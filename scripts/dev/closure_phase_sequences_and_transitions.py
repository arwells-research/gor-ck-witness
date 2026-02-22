#!/usr/bin/env python3

"""
closure_phase_sequences_and_transitions.py

From:
  data/derived/closure_phase_diagram_v2.csv

Produce:
  1) data/derived/closure_state_sequence_by_period.csv
     - per (period, ion_charge) with state, D_abs, D_sgn, A_plus, A_minus, dataset

  2) data/derived/closure_transition_counts.csv
     - counts of transitions state(q)->state(q+1) aggregated overall and by period

Notes:
  - D_abs = | |A+| - |A-| | / (|A+| + |A-|)
  - D_sgn = (|A+| - |A-|) / (|A+| + |A-|)
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def d_abs(ap: float, am: float) -> float:
    Ap = abs(ap)
    Am = abs(am)
    denom = Ap + Am
    if denom == 0.0:
        return float("nan")
    return float(abs(Ap - Am) / denom)


def d_sgn(ap: float, am: float) -> float:
    Ap = abs(ap)
    Am = abs(am)
    denom = Ap + Am
    if denom == 0.0:
        return float("nan")
    return float((Ap - Am) / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="data/derived/closure_phase_diagram_v2.csv",
        help="Input closure phase diagram v2",
    )
    ap.add_argument(
        "--out_seq",
        default="data/derived/closure_state_sequence_by_period.csv",
        help="Output per-(period,q) sequence CSV",
    )
    ap.add_argument(
        "--out_transitions",
        default="data/derived/closure_transition_counts.csv",
        help="Output transition counts CSV",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    required = ["ion_charge", "period", "state", "A_plus", "A_minus"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Columns present: {list(df.columns)}")

    # Types
    df["ion_charge"] = df["ion_charge"].astype(int)
    df["period"] = df["period"].astype(int)
    df["state"] = df["state"].fillna("UNKNOWN").astype(str)
    df["A_plus"] = pd.to_numeric(df["A_plus"], errors="coerce")
    df["A_minus"] = pd.to_numeric(df["A_minus"], errors="coerce")

    # Ensure D_abs exists; recompute anyway (authoritative from A+/-)
    df["D_abs"] = [d_abs(ap_, am_) for ap_, am_ in zip(df["A_plus"], df["A_minus"])]
    df["D_sgn"] = [d_sgn(ap_, am_) for ap_, am_ in zip(df["A_plus"], df["A_minus"])]

    # Sequence table
    seq_cols = [
        "period",
        "ion_charge",
        "state",
        "D_abs",
        "D_sgn",
        "A_plus",
        "A_minus",
    ]
    # Optional columns if present
    for opt in ["dataset", "topology", "lock", "rho_A", "p_lock_D"]:
        if opt in df.columns:
            seq_cols.append(opt)

    seq = df[seq_cols].sort_values(["period", "ion_charge"])
    seq.to_csv(args.out_seq, index=False)

    # Transition counts
    # For each period, sort by q and count transitions state(q)->state(q+1)
    rows = []
    for period, g in seq.groupby("period", sort=True):
        g2 = g.sort_values("ion_charge")
        states = list(g2["state"].astype(str).values)
        qs = list(g2["ion_charge"].astype(int).values)
        if len(states) < 2:
            continue
        for i in range(len(states) - 1):
            q0, q1 = qs[i], qs[i + 1]
            s0, s1 = states[i], states[i + 1]
            # Only count adjacent q if truly consecutive
            if q1 != q0 + 1:
                continue
            rows.append(
                {
                    "period": int(period),
                    "q_from": int(q0),
                    "q_to": int(q1),
                    "transition": f"{s0}->{s1}",
                }
            )

    trans = pd.DataFrame(rows)
    if len(trans) == 0:
        # still write empty artifacts with headers
        trans_summary = pd.DataFrame(
            columns=["scope", "period", "transition", "count"]
        )
    else:
        # Overall counts
        overall = (
            trans.groupby("transition")
            .size()
            .reset_index(name="count")
            .assign(scope="overall", period=pd.NA)
        )

        # By-period counts
        byp = (
            trans.groupby(["period", "transition"])
            .size()
            .reset_index(name="count")
            .assign(scope="by_period")
        )

        trans_summary = pd.concat([overall, byp], ignore_index=True)[
            ["scope", "period", "transition", "count"]
        ].sort_values(["scope", "period", "count", "transition"], ascending=[True, True, False, True])

    trans_summary.to_csv(args.out_transitions, index=False)

    print(f"Wrote: {args.out_seq}")
    print(f"Wrote: {args.out_transitions}")
    print()
    # small previews
    print("== sequence preview ==")
    print(seq.head(20).to_string(index=False))
    print()
    print("== transition summary preview ==")
    print(trans_summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
