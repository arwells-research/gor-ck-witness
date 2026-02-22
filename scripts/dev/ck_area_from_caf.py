from __future__ import annotations

import argparse
import pandas as pd

from gor_ck_witness.lobes import lobe_areas_from_residuals, summarize_lobes


def caf_residuals_pblock(ies_p1_to_p6: list[float]) -> list[float]:
    """
    CAF residuals for p^1..p^6 using x = 1..6.
    Baseline is the unique line through (x=2, IE_p2) and (x=5, IE_p5).
    """
    if len(ies_p1_to_p6) != 6:
        raise ValueError("Need exactly 6 values (p^1..p^6).")

    x2, y2 = 2.0, float(ies_p1_to_p6[1])  # p^2
    x5, y5 = 5.0, float(ies_p1_to_p6[4])  # p^5

    m = (y5 - y2) / (x5 - x2)
    b = y2 - m * x2

    rs: list[float] = []
    for x, y in enumerate(ies_p1_to_p6, start=1):  # x=1..6
        y_base = m * float(x) + b
        rs.append(float(y) - y_base)
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_csv", default="data/derived/ck_area_caf.csv")
    ap.add_argument("--eps_zero", type=float, default=0.0, help="Snap |r|<=eps to zero before lobe segmentation.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = ["period", "p1", "p2", "p3", "p4", "p5", "p6"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Columns present: {list(df.columns)}")

    rows = []
    for _, row in df.sort_values("period").iterrows():
        period = int(row["period"])
        ies = [float(row[f"p{i}"]) for i in range(1, 7)]

        rs = caf_residuals_pblock(ies)
        xs = [1, 2, 3, 4, 5, 6]

        lobes = lobe_areas_from_residuals(xs, rs, eps_zero=args.eps_zero)
        s = summarize_lobes(lobes)

        rows.append({"period": period, **s})

    out = pd.DataFrame(rows).sort_values("period")
    out.to_csv(args.out_csv, index=False)

    print(f"Wrote: {args.out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()


