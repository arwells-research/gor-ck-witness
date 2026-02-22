#!/usr/bin/env python3

"""
closure_phase_order_parameter.py

Compute the closure order parameter curve D(q) from:
    data/derived/closure_phase_diagram_v2.csv

Outputs:
  1) data/derived/closure_order_parameter_by_q.csv
     - mean/std/min/max/count of D_abs by ion_charge (q)
     - plus fractions of each discrete state {P+, P-, C*, C}

  2) data/derived/closure_order_parameter_by_q_period.csv
     - per-(q, period) table with D_abs and state (for quick audits)

No plotting.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


STATE_ORDER = ["P+", "P-", "C*", "C", "UNKNOWN"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="data/derived/closure_phase_diagram_v2.csv",
        help="Input CSV with state and D_abs",
    )
    ap.add_argument(
        "--out_by_q",
        default="data/derived/closure_order_parameter_by_q.csv",
        help="Output order-parameter-by-q CSV",
    )
    ap.add_argument(
        "--out_by_q_period",
        default="data/derived/closure_order_parameter_by_q_period.csv",
        help="Output per-(q,period) audit CSV",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    required = ["ion_charge", "period", "state", "D_abs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Columns present: {list(df.columns)}")

    # Clean types
    df["ion_charge"] = df["ion_charge"].astype(int)
    df["period"] = df["period"].astype(int)
    df["state"] = df["state"].fillna("UNKNOWN").astype(str)
    df["D_abs"] = pd.to_numeric(df["D_abs"], errors="coerce")

    # Per-(q,period) audit table
    audit_cols = ["ion_charge", "period", "dataset", "topology", "lock", "state", "D_abs", "A_plus", "A_minus", "rho_A", "p_lock_D"]
    audit_cols = [c for c in audit_cols if c in df.columns]
    df_audit = df[audit_cols].sort_values(["ion_charge", "period"])
    df_audit.to_csv(args.out_by_q_period, index=False)

    # Aggregate by q
    rows = []
    for q, g in df.groupby("ion_charge", sort=True):
        d = g["D_abs"].dropna()
        row = {
            "ion_charge": int(q),
            "count_total": int(len(g)),
            "count_D": int(len(d)),
            "D_mean": float(d.mean()) if len(d) else np.nan,
            "D_std": float(d.std(ddof=1)) if len(d) >= 2 else np.nan,
            "D_min": float(d.min()) if len(d) else np.nan,
            "D_max": float(d.max()) if len(d) else np.nan,
        }

        # State fractions (over total rows for that q)
        counts = g["state"].value_counts(dropna=False).to_dict()
        for s in STATE_ORDER:
            row[f"frac_{s.replace('*','star')}"] = float(counts.get(s, 0) / max(1, len(g)))

        # Also include an "other" fraction for sanity
        known = sum(counts.get(s, 0) for s in STATE_ORDER)
        row["frac_other"] = float(max(0, len(g) - known) / max(1, len(g)))

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("ion_charge")
    out.to_csv(args.out_by_q, index=False)

    print(f"Wrote: {args.out_by_q}")
    print(f"Wrote: {args.out_by_q_period}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
    