#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _infer_ion_charge_from_path(csv_path: Path) -> int:
    """
    Expect file naming like:
      data/raw/nist_pblock_ie6_periods_3to6.csv  -> ie6 means IE6 => ion_charge=5
      data/raw/nist_pblock_ie1_periods_2to6.csv  -> ie1 means IE1 => ion_charge=0
    """
    m = re.search(r"nist_pblock_ie(\d+)_", csv_path.name)
    if not m:
        raise ValueError(f"Could not infer IE# from filename: {csv_path}")
    ie_num = int(m.group(1))
    if ie_num < 1:
        raise ValueError(f"Invalid IE number inferred from {csv_path}: {ie_num}")
    return ie_num - 1


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> None:
    p = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  rc: {p.returncode}\n"
            f"  stdout:\n{p.stdout}\n"
            f"  stderr:\n{p.stderr}\n"
        )


def _classify_topology(a_plus: float, a_minus: float, *, eps: float = 0.0) -> str:
    """
    Discrete topology classification based on lobe areas.
    a_minus is expected <= 0 in our convention.
    eps allows treating tiny numerical leakage as zero.
    """
    ap = float(a_plus)
    am = float(a_minus)

    ap0 = abs(ap) <= eps
    am0 = abs(am) <= eps

    if (not ap0) and (not am0):
        # two-lobe requires opposite signs (ap>0, am<0) in our construction
        if ap > 0.0 and am < 0.0:
            return "two_lobe"
        # fallback: unexpected sign convention
        return "two_lobe_unexpected_signs"
    if (not ap0) and am0:
        return "plus_only"
    if ap0 and (not am0):
        return "minus_only"
    return "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="data/raw/nist_pblock_ie*_periods_*.csv",
        help="Glob of input IE csv files.",
    )
    ap.add_argument(
        "--nperm",
        type=int,
        default=20000,
        help="Number of permutations for null script.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--eps_zero",
        type=float,
        default=0.0,
        help="Zero-threshold passed through to area computation (if supported).",
    )
    ap.add_argument(
        "--lock_alpha",
        type=float,
        default=0.02,
        help="Threshold on p_lock_D for calling two-lobe 'locked'.",
    )
    ap.add_argument(
        "--out_csv",
        default="data/derived/closure_phase_diagram.csv",
        help="Output phase diagram CSV.",
    )
    args = ap.parse_args()

    root = Path(".")
    script_ck_area = root / "scripts" / "ck_area_from_caf.py"
    script_null = root / "scripts" / "ck_area_null_permutation.py"

    if not script_ck_area.exists():
        raise FileNotFoundError(f"Missing script: {script_ck_area}")
    if not script_null.exists():
        raise FileNotFoundError(f"Missing script: {script_null}")

    csv_paths = sorted(Path().glob(args.glob))
    if not csv_paths:
        raise FileNotFoundError(f"No input files found for glob: {args.glob}")

    # We will use temp output files so each dataset doesn't overwrite the previous one.
    tmp_dir = Path("data/derived/_phase_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for csv_path in csv_paths:
        ion_charge = _infer_ion_charge_from_path(csv_path)

        out_caf = tmp_dir / f"ck_area_caf__q{ion_charge}__{csv_path.stem}.csv"
        out_null = tmp_dir / f"ck_area_null__q{ion_charge}__{csv_path.stem}.csv"

        # 1) run CAF area extractor
        cmd1 = [
            "python3",
            str(script_ck_area),
            "--csv",
            str(csv_path),
            "--out_csv",
            str(out_caf),
        ]
        # include eps_zero only if your ck_area_from_caf.py supports it; harmless to omit otherwise
        # cmd1 += ["--eps_zero", str(args.eps_zero)]
        _run(cmd1)

        # 2) run permutation null
        cmd2 = [
            "python3",
            str(script_null),
            "--csv",
            str(csv_path),
            "--nperm",
            str(args.nperm),
            "--seed",
            str(args.seed),
            "--out_csv",
            str(out_null),
        ]
        _run(cmd2)

        caf = pd.read_csv(out_caf)
        nul = pd.read_csv(out_null)

        # Expect caf columns: period, A_plus, A_minus, rho_A
        # Expect nul columns include: period, p_lock_D, p_exist_D, valid_fraction_D, ...
        merged = caf.merge(nul, on="period", how="left", suffixes=("", "_null"))

        for _, r in merged.iterrows():
            period = int(r["period"])
            A_plus = float(r["A_plus"])
            A_minus = float(r["A_minus"])
            rho_A = float(r["rho_A"]) if pd.notna(r["rho_A"]) else float("nan")

            topology = _classify_topology(A_plus, A_minus, eps=args.eps_zero)

            p_lock_D = float(r["p_lock_D"]) if "p_lock_D" in r and pd.notna(r["p_lock_D"]) else float("nan")

            if topology == "two_lobe":
                lock = "locked" if (pd.notna(p_lock_D) and p_lock_D <= args.lock_alpha) else "unlocked"
            else:
                lock = ""

            rows.append(
                {
                    "ion_charge": ion_charge,
                    "period": period,
                    "dataset": csv_path.name,
                    "topology": topology,
                    "lock": lock,
                    "rho_A": rho_A,
                    "A_plus": A_plus,
                    "A_minus": A_minus,
                    "p_lock_D": p_lock_D,
                }
            )

    out = pd.DataFrame(rows).sort_values(["ion_charge", "period", "dataset"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Wrote: {args.out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
    