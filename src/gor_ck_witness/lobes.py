from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence

import numpy as np
import math

Point = Tuple[float, float]


@dataclass(frozen=True)
class Lobe:
    sign: int              # +1 or -1
    x0: float
    x1: float
    area: float            # signed area under r(x) over [x0, x1]

def stable_lobe_stats(lobes):
    s = summarize_lobes(lobes)
    A_plus = abs(s["A_plus"])
    A_minus = abs(s["A_minus"])

    denom = A_plus + A_minus
    D = (A_plus - A_minus) / denom if denom > 0 else float("nan")

    # log ratio (finite if both > 0)
    L = math.log(A_plus / A_minus) if (A_plus > 0 and A_minus > 0) else float("nan")

    return {
        **s,
        "D_norm": float(D),
        "L_logratio": float(L),
        "A_sum": float(denom),
        "A_min": float(min(A_plus, A_minus)),
    }


def _insert_zero_crossings(xs: Sequence[float], rs: Sequence[float], eps: float = 0.0) -> List[Point]:
    """
    Build a piecewise-linear polyline (x, r) and insert points where it crosses r=0.
    Assumes xs are strictly increasing.
    """
    pts: List[Point] = [(float(xs[0]), float(rs[0]))]
    for i in range(len(xs) - 1):
        x0, r0 = float(xs[i]), float(rs[i])
        x1, r1 = float(xs[i + 1]), float(rs[i + 1])

        # If exactly zero, keep it (helps segmentation).
        # If a strict sign change or one endpoint is zero -> add crossing.
        pts.append((x1, r1))

        # detect crossing between endpoints (excluding trivial both-zero)
        if (r0 == 0.0 and r1 == 0.0):
            continue
        if (r0 == 0.0) or (r1 == 0.0) or (r0 < 0.0 < r1) or (r1 < 0.0 < r0):
            # if one endpoint is exactly zero, the crossing point is that endpoint
            if r0 == 0.0 or r1 == 0.0:
                continue
            # linear interpolation for r=0
            t = -r0 / (r1 - r0)
            xc = x0 + t * (x1 - x0)
            # insert crossing between (x0,r0) and (x1,r1)
            pts.insert(-1, (float(xc), 0.0))

    # Optional: snap tiny residuals to zero to prevent micro-lobes
    if eps > 0:
        pts = [(x, 0.0 if abs(r) <= eps else r) for (x, r) in pts]

    # Remove consecutive duplicate points (same x and r)
    out: List[Point] = []
    for p in pts:
        if not out or (p[0] != out[-1][0] or p[1] != out[-1][1]):
            out.append(p)
    return out


def _segment_lobes(pts: Sequence[Point]) -> List[List[Point]]:
    """
    Split the polyline into segments separated by r=0 points.
    Each segment corresponds to a lobe (all nonzero r with constant sign),
    with endpoints on r=0.
    """
    lobes: List[List[Point]] = []
    cur: List[Point] = [pts[0]]
    for p in pts[1:]:
        cur.append(p)
        if p[1] == 0.0 and len(cur) >= 2:
            # close a segment if it contains nonzero interior
            has_nonzero = any(q[1] != 0.0 for q in cur)
            if has_nonzero:
                lobes.append(cur)
            cur = [p]
    return lobes


def _trap_area(seg: Sequence[Point]) -> float:
    """Trapezoid rule area under r(x) for a polyline segment."""
    a = 0.0
    for (x0, r0), (x1, r1) in zip(seg, seg[1:]):
        a += 0.5 * (r0 + r1) * (x1 - x0)
    return float(a)


def lobe_areas_from_residuals(xs: Sequence[float], rs: Sequence[float], *, eps_zero: float = 0.0) -> List[Lobe]:
    """
    Given ordered xs and residuals rs, compute signed lobe areas between r=0 crossings.
    Returns Lobe objects with sign (+1/-1), span, and signed area.
    """
    xs = list(xs)
    rs = list(rs)
    if len(xs) != len(rs):
        raise ValueError("xs and rs must have same length.")
    if len(xs) < 2:
        return []

    pts = _insert_zero_crossings(xs, rs, eps=eps_zero)
    segs = _segment_lobes(pts)

    out: List[Lobe] = []
    for seg in segs:
        area = _trap_area(seg)
        # Determine sign by the first nonzero interior point
        sgn = 0
        for (_, r) in seg:
            if r != 0.0:
                sgn = 1 if r > 0 else -1
                break
        if sgn == 0:
            continue
        out.append(Lobe(sign=sgn, x0=seg[0][0], x1=seg[-1][0], area=area))
    return out


def summarize_lobes(lobes: Sequence[Lobe]) -> dict:
    """
    Return a stable summary:
      A_plus: total positive lobe area
      A_minus: total negative lobe area (signed, so negative)
      rho_A: |A_plus|/|A_minus| (nan if missing)
    """
    A_plus = sum(l.area for l in lobes if l.sign > 0)
    A_minus = sum(l.area for l in lobes if l.sign < 0)  # negative
    if A_minus == 0.0:
        rho = float("nan")
    else:
        rho = abs(A_plus) / abs(A_minus)
    return {"A_plus": float(A_plus), "A_minus": float(A_minus), "rho_A": float(rho)}
