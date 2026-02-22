#!/usr/bin/env python3
"""
Build NIST p-block ionization-energy datasets (wide CSV) for arbitrary ion stage.

Fetches NIST ASD ionization energies via ie.pl (format=2 CSV-ish) and extracts
the ionization energy (eV) at a requested ion charge Q:

- Q=0 corresponds to IE1 (neutral atom -> +1 ion)
- Q=1 corresponds to IE2 (+1 ion -> +2 ion)
- Q=2 corresponds to IE3 (+2 ion -> +3 ion)
etc.

Outputs:
- Wide CSV (default): data/raw/nist_pblock_ie{Q+1}_periods_2to6.csv
  Columns: n, period, p1..p6, labels
- Audit CSV (default): data/derived/ie{Q+1}_extracted_audit.csv
  Columns: period, symbol, query_tried, ion_charge, IE_eV, status, error

New:
- --include_periods 2,3,4,5,6,7     restrict periods to include
- Robust detection of NIST HTML error pages; cooperates with --drop_missing.
- Per-element query fallback: tries a list of query strings (symbol-first).
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


# p-block periods, p1..p6 order
# For each element: (symbol, [query strings to try, in order])
P_BLOCK: Dict[int, List[Tuple[str, List[str]]]] = {
    2: [
        ("B", ["B", "boron"]),
        ("C", ["C", "carbon"]),
        ("N", ["N", "nitrogen"]),
        ("O", ["O", "oxygen"]),
        ("F", ["F", "fluorine"]),
        ("Ne", ["Ne", "neon"]),
    ],
    3: [
        ("Al", ["Al", "aluminum"]),
        ("Si", ["Si", "silicon"]),
        ("P", ["P", "phosphorus"]),
        ("S", ["S", "sulfur"]),
        ("Cl", ["Cl", "chlorine"]),
        ("Ar", ["Ar", "argon"]),
    ],
    4: [
        ("Ga", ["Ga", "gallium"]),
        ("Ge", ["Ge", "germanium"]),
        ("As", ["As", "arsenic"]),
        ("Se", ["Se", "selenium"]),
        ("Br", ["Br", "bromine"]),
        ("Kr", ["Kr", "krypton"]),
    ],
    5: [
        ("In", ["In", "indium"]),
        ("Sn", ["Sn", "tin"]),
        ("Sb", ["Sb", "antimony"]),
        ("Te", ["Te", "tellurium"]),
        ("I", ["I", "iodine"]),
        ("Xe", ["Xe", "xenon"]),
    ],
    6: [
        ("Tl", ["Tl", "thallium"]),
        ("Pb", ["Pb", "lead"]),
        ("Bi", ["Bi", "bismuth"]),
        ("Po", ["Po", "polonium"]),
        ("At", ["At", "astatine"]),
        ("Rn", ["Rn", "radon"]),
    ],
    # Period 7 p-block: Nh Fl Mc Lv Ts Og
    # NIST coverage can be sparse/incomplete; we try symbol first, then common names,
    # then older systematic names (some datasets use those).
    7: [
        ("Nh", ["Nh", "Nihonium", "nihonium", "Ununtrium", "ununtrium", "Uut"]),
        ("Fl", ["Fl", "Flerovium", "flerovium", "Ununquadium", "ununquadium", "Uuq"]),
        ("Mc", ["Mc", "Moscovium", "moscovium", "Ununpentium", "ununpentium", "Uup"]),
        ("Lv", ["Lv", "Livermorium", "livermorium", "Ununhexium", "ununhexium", "Uuh"]),
        ("Ts", ["Ts", "Tennessine", "tennessine", "Ununseptium", "ununseptium", "Uus"]),
        ("Og", ["Og", "Oganesson", "oganesson", "Ununoctium", "ununoctium", "Uuo"]),
    ],
}

NIST_IE_URL = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"


def _de_excelify(s: str) -> str:
    """
    NIST format=2 sometimes returns cells like:
      '=""+1""' or '=""24.383143""'
    Strip Excel-safe wrapper -> plain content.
    """
    s = str(s).strip()
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    s = s.replace('""', '"')
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.strip()


def _to_int_charge(cell: str) -> int | None:
    cell = _de_excelify(cell).strip()
    if cell == "" or cell.lower() == "nan":
        return None
    if not re.match(r"^[\+\-]?\d+$", cell):
        return None
    return int(cell)


def _to_float_energy(cell: str) -> float | None:
    cell = _de_excelify(cell).strip()
    if cell == "" or cell.lower() == "nan":
        return None
    cell = re.sub(r"[^0-9eE\.\+\-]", "", cell)
    if cell == "" or cell.lower() == "nan":
        return None
    return float(cell)


def _looks_like_html_error(text: str) -> bool:
    t = text.lstrip()
    # NIST sometimes returns an HTML page with an "Error Message" section.
    if t.startswith("<!DOCTYPE") or t.startswith("<html") or t.startswith("<HTML"):
        return True
    if "<h2>Error Message" in text or "<h2> Error Message" in text:
        return True
    return False


def fetch_nist_ie_table_csv(element_query: str, *, timeout_s: float = 30.0) -> List[dict]:
    """
    Fetch NIST ie.pl in format=2 CSV-like output with ion_charge_out + el_name_out.
    Returns list of dict rows with keys from the CSV header.
    Raises ValueError on HTML error pages or parse failures.
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

    if _looks_like_html_error(text):
        # Give a short hint of the page for debugging, but keep it compact.
        snippet = text.strip().splitlines()[0:5]
        raise ValueError(f"NIST returned HTML (likely error) for spectra='{element_query}'. Head={snippet}")

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


def ie_for_element_at_charge_with_fallback(
    symbol: str,
    query_candidates: List[str],
    ion_charge: int,
    *,
    sleep_s: float = 0.2,
) -> tuple[float, str]:
    """
    Try a list of spectra queries until one returns a parseable table that includes ion_charge.
    Returns (IE_eV, query_used).
    Raises ValueError if all queries fail.
    """
    last_err: str | None = None
    for qstr in query_candidates:
        try:
            rows = fetch_nist_ie_table_csv(qstr)
            keys = list(rows[0].keys())
            k_charge, k_energy = _find_columns(keys)

            for row in rows:
                q = _to_int_charge(row.get(k_charge, ""))
                if q == ion_charge:
                    e = _to_float_energy(row.get(k_energy, ""))
                    if e is None:
                        raise ValueError(f"Energy missing in row for spectra='{qstr}'. Row={row}")
                    time.sleep(sleep_s)
                    return float(e), qstr

            parsed_charges = [_to_int_charge(r.get(k_charge, "")) for r in rows]
            last_err = (
                f"no ion_charge={ion_charge} row; parsed_charges={parsed_charges}"
            )
        except Exception as e:
            last_err = str(e)

    raise ValueError(
        f"All query candidates failed for {symbol} ion_charge={ion_charge}. "
        f"Tried={query_candidates}. LastError={last_err}"
    )


def _parse_include_periods(s: str) -> set[int]:
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.add(int(tok))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ion_charge", type=int, required=True, help="Ion charge Q to select (0->IE1, 1->IE2, 2->IE3, ...)")
    ap.add_argument("--out_csv", default=None, help="Output wide CSV path (default based on ion_charge)")
    ap.add_argument("--audit_csv", default=None, help="Output audit CSV path (default based on ion_charge)")
    ap.add_argument("--sleep_s", type=float, default=0.2, help="Polite delay between NIST requests")
    ap.add_argument("--drop_missing", action="store_true", help="Skip elements that lack the requested ion_charge row or cannot be fetched/parsed")
    ap.add_argument("--require_full_period", action="store_true", help="If set, drop any period that does not have all 6 p-block entries")
    ap.add_argument(
        "--include_periods",
        default=None,
        help="Comma-separated period list to include (e.g. '2,3,4,5,6,7'). If omitted, uses all periods in P_BLOCK.",
    )
    args = ap.parse_args()

    q = int(args.ion_charge)
    ie_k = q + 1
    out_csv = args.out_csv or f"data/raw/nist_pblock_ie{ie_k}_periods_2to6.csv"
    audit_csv = args.audit_csv or f"data/derived/ie{ie_k}_extracted_audit.csv"

    include_periods: set[int] | None = None
    if args.include_periods is not None:
        include_periods = _parse_include_periods(args.include_periods)

    audit_rows = []
    wide_rows = []

    for period in sorted(P_BLOCK.keys()):
        if include_periods is not None and period not in include_periods:
            continue

        elems = P_BLOCK[period]
        vals: List[float] = []
        labels: List[str] = []
        ok_syms: List[str] = []

        for sym, qlist in elems:
            try:
                ie_val, used = ie_for_element_at_charge_with_fallback(sym, qlist, q, sleep_s=args.sleep_s)
                vals.append(ie_val)
                labels.append(sym)
                ok_syms.append(sym)
                audit_rows.append(
                    {
                        "period": period,
                        "symbol": sym,
                        "query_tried": used,
                        "ion_charge": q,
                        "IE_eV": float(ie_val),
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as e:
                audit_rows.append(
                    {
                        "period": period,
                        "symbol": sym,
                        "query_tried": "|".join(qlist),
                        "ion_charge": q,
                        "IE_eV": float("nan"),
                        "status": "missing",
                        "error": str(e),
                    }
                )
                if args.drop_missing:
                    continue
                raise

        if args.require_full_period and len(vals) != 6:
            # period incomplete
            continue

        # If absolutely nothing was extracted for this period, skip it.
        if len(vals) == 0:
            continue
        wide = {"n": period, "period": period}
        for i, v in enumerate(vals, start=1):
            wide[f"p{i}"] = float(v)
        wide["labels"] = ",".join(ok_syms)
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
