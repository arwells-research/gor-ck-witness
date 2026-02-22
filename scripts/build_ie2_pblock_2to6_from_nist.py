#!/usr/bin/env python3
"""
Build NIST p-block ionization-energy datasets (wide CSV) for arbitrary ion stage.

This script fetches NIST ASD ionization energies via ie.pl in format=2 (CSV-ish),
then selects the row for a requested ion charge Q and extracts the ionization
energy (eV). For example:

- Q=0 corresponds to IE1 (neutral atom -> +1 ion)
- Q=1 corresponds to IE2 (+1 ion -> +2 ion)
- Q=2 corresponds to IE3 (+2 ion -> +3 ion)
etc.

Outputs:
- Wide CSV: data/raw/nist_pblock_ie{Q+1}_periods_2to6.csv   (by default)
  Columns: n, period, p1..p6, labels
- Audit CSV: data/derived/ie{Q+1}_extracted_audit.csv
  Columns: period, symbol, query, ion_charge, IE_eV

Usage examples:
  PYTHONPATH=. python3 scripts/build_pblock_ionization_csv_from_nist.py --ion_charge 1
  PYTHONPATH=. python3 scripts/build_pblock_ionization_csv_from_nist.py --ion_charge 2 --out_csv data/raw/nist_pblock_ie3_periods_2to6.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests


# p-block periods 2–6, in p1..p6 order (same ordering contract as your IE1 dataset)
P_BLOCK: Dict[int, List[Tuple[str, str]]] = {
    2: [("B", "boron"), ("C", "carbon"), ("N", "nitrogen"), ("O", "oxygen"), ("F", "fluorine"), ("Ne", "neon")],
    3: [("Al", "aluminum"), ("Si", "silicon"), ("P", "phosphorus"), ("S", "sulfur"), ("Cl", "chlorine"), ("Ar", "argon")],
    4: [("Ga", "gallium"), ("Ge", "germanium"), ("As", "arsenic"), ("Se", "selenium"), ("Br", "bromine"), ("Kr", "krypton")],
    5: [("In", "indium"), ("Sn", "tin"), ("Sb", "antimony"), ("Te", "tellurium"), ("I", "iodine"), ("Xe", "xenon")],
    6: [("Tl", "thallium"), ("Pb", "lead"), ("Bi", "bismuth"), ("Po", "polonium"), ("At", "astatine"), ("Rn", "radon")],
}

NIST_IE_URL = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"


def _de_excelify(s: str) -> str:
    """
    NIST format=2 sometimes returns cells like:
      '=""+1""' or '=""24.383143""'
    This strips the Excel-safe wrapper and returns plain content.
    """
    s = str(s).strip()
    # Remove leading = and wrapping quotes patterns like ="...".
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    # Unescape doubled quotes
    s = s.replace('""', '"')
    # Strip outer quotes if present
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.strip()


def _to_int_charge(cell: str) -> int | None:
    cell = _de_excelify(cell)
    cell = cell.strip()
    if cell == "" or cell.lower() == "nan":
        return None
    # Accept "+1", "1", "0", etc.
    m = re.match(r"^[\+\-]?\d+$", cell)
    if not m:
        return None
    return int(cell)


def _to_float_energy(cell: str) -> float | None:
    cell = _de_excelify(cell)
    cell = cell.strip()
    if cell == "" or cell.lower() == "nan":
        return None
    # Keep digits, decimal, exponent, sign
    cell = re.sub(r"[^0-9eE\.\+\-]", "", cell)
    if cell == "" or cell.lower() == "nan":
        return None
    return float(cell)


def fetch_nist_ie_table_csv(element_query: str, *, timeout_s: float = 30.0) -> List[dict]:
    """
    Fetch NIST ie.pl in format=2 CSV-like output with ion_charge_out + el_name_out.
    Returns list of dict rows with keys taken from header.
    """
    params = {
        "spectra": element_query,
        "units": "1",     # eV
        "format": "2",    # CSV-ish
        "e_out": "0",
        "ion_charge_out": "on",
        "el_name_out": "on",
    }
    r = requests.get(NIST_IE_URL, params=params, timeout=timeout_s)
    r.raise_for_status()
    text = r.text

    # NIST appends a "Notes:" section. We only want the CSV header + data rows.
    lines = text.splitlines()
    csv_lines: List[str] = []
    for line in lines:
        if line.strip().startswith("Notes:"):
            break
        if line.strip() == "":
            continue
        csv_lines.append(line)

    if not csv_lines:
        raise ValueError(f"No CSV lines returned for '{element_query}'.")

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    rows = [row for row in reader]

    if not rows:
        raise ValueError(f"No data rows parsed for '{element_query}'. First lines:\n{csv_lines[:5]}")
    return rows


def _find_columns(keys: List[str]) -> tuple[str, str]:
    """
    Identify the ion charge column and the ionization energy (eV) column.
    """
    k_charge = None
    k_energy = None

    for k in keys:
        lk = k.strip().lower()
        if lk.startswith("ion charge"):
            k_charge = k
        if ("ionization energy" in lk) and ("ev" in lk):
            k_energy = k

    if k_charge is None or k_energy is None:
        raise ValueError(f"Could not infer charge/energy columns. Keys={keys}")

    return k_charge, k_energy


def ie_for_element_at_charge(symbol: str, element_query: str, ion_charge: int, *, sleep_s: float = 0.2) -> float:
    """
    Ionization energy corresponding to the given ion charge row.
    For ion_charge=0 -> IE1
    For ion_charge=1 -> IE2
    etc.
    """
    rows = fetch_nist_ie_table_csv(element_query)

    keys = list(rows[0].keys())
    k_charge, k_energy = _find_columns(keys)

    for row in rows:
        q = _to_int_charge(row.get(k_charge, ""))
        if q == ion_charge:
            e = _to_float_energy(row.get(k_energy, ""))
            if e is None:
                raise ValueError(f"Energy missing for {symbol} ({element_query}) ion_charge={ion_charge}. Row={row}")
            time.sleep(sleep_s)
            return float(e)

    parsed_charges = [_to_int_charge(r.get(k_charge, "")) for r in rows]
    raise ValueError(
        f"Did not find ion_charge={ion_charge} row for {symbol} ({element_query}). "
        f"Parsed charges: {parsed_charges}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ion_charge", type=int, required=True, help="Ion charge Q to select (0->IE1, 1->IE2, 2->IE3, ...)")
    ap.add_argument("--out_csv", default=None, help="Output wide CSV path (default based on ion_charge)")
    ap.add_argument("--audit_csv", default=None, help="Output audit CSV path (default based on ion_charge)")
    ap.add_argument("--sleep_s", type=float, default=0.2, help="Polite delay between NIST requests")

    args = ap.parse_args()
    q = int(args.ion_charge)
    ie_k = q + 1

    out_csv = args.out_csv or f"data/raw/nist_pblock_ie{ie_k}_periods_2to6.csv"
    audit_csv = args.audit_csv or f"data/derived/ie{ie_k}_extracted_audit.csv"

    audit_rows = []
    wide_rows = []

    for period, elems in P_BLOCK.items():
        vals = []
        labels = []
        for sym, query in elems:
            ie_val = ie_for_element_at_charge(sym, query, q, sleep_s=args.sleep_s)
            vals.append(ie_val)
            labels.append(sym)
            audit_rows.append(
                {
                    "period": period,
                    "symbol": sym,
                    "query": query,
                    "ion_charge": q,
                    "IE_eV": ie_val,
                }
            )

        wide = {"n": period, "period": period}
        for i, v in enumerate(vals, start=1):
            wide[f"p{i}"] = float(v)
        wide["labels"] = ",".join(labels)
        wide_rows.append(wide)

    df_wide = pd.DataFrame(wide_rows).sort_values("period")
    df_wide.to_csv(out_csv, index=False)

    df_audit = pd.DataFrame(audit_rows).sort_values(["period", "symbol"])
    df_audit.to_csv(audit_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {audit_csv}")
    print(df_wide.to_string(index=False))


if __name__ == "__main__":
    main()
