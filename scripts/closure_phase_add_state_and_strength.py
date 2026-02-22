#!/usr/bin/env python3

"""
closure_phase_add_state_and_strength.py

Promotes closure_phase_diagram.csv to canonical form by adding:

    state  ∈ {P+, P-, C*, C}
    D_abs  ∈ [0,1] closure strength scalar

Output:
    data/derived/closure_phase_diagram_v2.csv

No external dependencies beyond pandas/numpy.
"""

import argparse
import pandas as pd
import numpy as np


# -----------------------------
# State classification
# -----------------------------

def classify_state(topology, lock):
    if topology == "plus_only":
        return "P+"
    if topology == "minus_only":
        return "P-"
    if topology == "two_lobe":
        if isinstance(lock, str) and lock.strip().lower() == "locked":
            return "C"
        else:
            return "C*"
    return "UNKNOWN"


# -----------------------------
# Closure strength scalar
# -----------------------------

def compute_D_abs(A_plus, A_minus):
    Ap = abs(A_plus)
    Am = abs(A_minus)
    denom = Ap + Am
    if denom == 0:
        return np.nan
    return abs(Ap - Am) / denom


# -----------------------------
# Main
# -----------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_csv",
        default="data/derived/closure_phase_diagram.csv",
        help="Input closure phase diagram CSV",
    )

    parser.add_argument(
        "--out_csv",
        default="data/derived/closure_phase_diagram_v2.csv",
        help="Output CSV with state and strength",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    # Add state column
    df["state"] = [
        classify_state(topo, lock)
        for topo, lock in zip(df["topology"], df["lock"])
    ]

    # Add D_abs column
    df["D_abs"] = [
        compute_D_abs(Ap, Am)
        for Ap, Am in zip(df["A_plus"], df["A_minus"])
    ]

    # Sort for readability
    df = df.sort_values(["ion_charge", "period"])

    df.to_csv(args.out_csv, index=False)

    print(f"Wrote: {args.out_csv}")
    print()
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
    