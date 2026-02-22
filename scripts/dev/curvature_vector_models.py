#!/usr/bin/env python3
"""
Curvature-vector -> lobe-geometry regression + state/lock classification (two_lobe only).

Run:
  python3 scripts/curvature_vector_models.py
"""

from __future__ import annotations

import ast
import numpy as np
import pandas as pd

IN = "data/derived/closure_phase_diagram_v2_canon_with_ie.csv"

def parse_profile(x):
    # IE_profile may be stored as a Python list string; handle both list and string.
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        # Some CSV writers produce strings like "[1,2,3]" -> safe parse
        try:
            return ast.literal_eval(x)
        except Exception:
            pass
    raise ValueError(f"Could not parse IE_profile: {type(x)} {x!r}")

def second_differences(y: np.ndarray) -> np.ndarray:
    # y length 6 -> d2 length 4
    return y[2:] - 2*y[1:-1] + y[:-2]

def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[m] = a[m] / b[m]
    return out

def loocv_linear_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    LOOCV predictions via least squares on training folds.
    Returns (r2, rmse).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[m]
    y = y[m]
    n = X.shape[0]
    if n < 4:
        return (float("nan"), float("nan"))

    yhat = np.full(n, np.nan, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xt = X[mask]
        yt = y[mask]
        # add intercept
        A = np.column_stack([np.ones(Xt.shape[0]), Xt])
        coef, *_ = np.linalg.lstsq(A, yt, rcond=None)
        yhat[i] = coef[0] + X[i] @ coef[1:]

    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - (sse / sst if sst > 0 else np.nan)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return (r2, rmse)

def loocv_nearest_centroid_accuracy(X: np.ndarray, labels: np.ndarray):
    """
    Multi-class nearest-centroid classifier with LOOCV.
    Returns (accuracy, confusion_df).
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=object)

    m = np.isfinite(X).all(axis=1) & pd.notna(labels)
    X = X[m]
    labels = labels[m]
    n = X.shape[0]
    if n < 4:
        return (float("nan"), pd.DataFrame())

    classes = sorted(pd.unique(labels).tolist())
    ypred = np.empty(n, dtype=object)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xt = X[mask]
        lt = labels[mask]

        centroids = {}
        for c in classes:
            rows = Xt[lt == c]
            if rows.shape[0] == 0:
                continue
            centroids[c] = np.mean(rows, axis=0)

        xi = X[i]
        best_c = None
        best_d = None
        for c, mu in centroids.items():
            d = float(np.sum((xi - mu) ** 2))
            if best_d is None or d < best_d:
                best_d = d
                best_c = c
        ypred[i] = best_c

    acc = float(np.mean(ypred == labels))
    cm = pd.crosstab(
        pd.Series(labels, name="true"),
        pd.Series(ypred, name="pred"),
        dropna=False
    )
    return acc, cm

def permutation_test_loocv_acc(
    X: np.ndarray,
    labels: np.ndarray,
    nperm: int = 5000,
    seed: int = 0,
    return_perm: bool = False,
):
    """
    Permutation test for LOOCV nearest-centroid accuracy.
    p = P(acc_perm >= acc_obs) with +1 smoothing.
    """
    rng = np.random.default_rng(seed)

    acc_obs, cm_obs = loocv_nearest_centroid_accuracy(X, labels)
    if not np.isfinite(acc_obs):
        raise ValueError("Observed accuracy is NaN; check X/labels filtering.")

    perm_acc = np.empty(nperm, dtype=float)
    labels = np.asarray(labels, dtype=object)

    for k in range(nperm):
        perm_labels = labels.copy()
        rng.shuffle(perm_labels)
        perm_acc[k], _ = loocv_nearest_centroid_accuracy(X, perm_labels)

    # one-sided p-value with +1 smoothing
    p = float((1 + np.sum(perm_acc >= acc_obs)) / (1 + nperm))
    out = {
        "acc_obs": float(acc_obs),
        "p_ge": p,
        "perm_acc_mean": float(np.mean(perm_acc)),
        "perm_acc_std": float(np.std(perm_acc)),
        "cm_obs": cm_obs,
    }
    if return_perm:
        out["perm_acc"] = perm_acc
    return out

def print_perm_result(name: str, res: dict):
    print(f"\n== {name} ==")
    print(f"acc_obs       = {res['acc_obs']:.6f}")
    print(f"perm_mean     = {res['perm_acc_mean']:.6f}")
    print(f"perm_std      = {res['perm_acc_std']:.6f}")
    print(f"p(acc>=obs)   = {res['p_ge']:.6g}")
    print("confusion (obs):")
    print(res["cm_obs"].to_string())

def main():
    df = pd.read_csv(IN)

    # --- hard requirements ---
    required = ["period", "ion_charge", "state", "D_abs", "topology", "IE_profile"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected at least: {required}. "
            f"Rebuild canon with IE merge first."
        )

    # restrict to usable rows
    df = df.dropna(subset=required).copy()
    df["period"] = df["period"].astype(int)
    df["ion_charge"] = df["ion_charge"].astype(int)
    df["state"] = df["state"].astype(str)
    df["topology"] = df["topology"].astype(str)

    # --- parse IE profiles once ---
    Y = []
    for v in df["IE_profile"].tolist():
        y = np.asarray(parse_profile(v), dtype=float)
        if y.size != 6:
            raise ValueError(f"Expected IE_profile length 6, got {y.size}")
        Y.append(y)
    Y = np.vstack(Y)  # (N,6)

    # --- shape-derived feature blocks ---
    # z-scored profile (remove level + scale)
    y_mu = Y.mean(axis=1, keepdims=True)
    y_sd = Y.std(axis=1, keepdims=True)
    y_sd = np.where(y_sd == 0, 1.0, y_sd)
    Yz = (Y - y_mu) / y_sd

    # first and second differences
    D1 = np.diff(Y, axis=1)            # (N,5)
    D2 = np.diff(Y, n=2, axis=1)       # (N,4) == second differences

    # z-score difference blocks rowwise as well (so features comparable)
    def zrow(M):
        mu = M.mean(axis=1, keepdims=True)
        sd = M.std(axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        return (M - mu) / sd

    D1z = zrow(D1)
    D2z = zrow(D2)

    # store d2 columns for convenience/back-compat
    df["d2_0"] = D2[:, 0]
    df["d2_1"] = D2[:, 1]
    df["d2_2"] = D2[:, 2]
    df["d2_3"] = D2[:, 3]

    # --- lobe-geometry targets (optional; only if present) ---
    have_lobes = ("A_plus" in df.columns) and ("A_minus" in df.columns)
    if have_lobes:
        Aplus = pd.to_numeric(df["A_plus"], errors="coerce").to_numpy(dtype=float)
        Aminus = pd.to_numeric(df["A_minus"], errors="coerce").to_numpy(dtype=float)
        absAminus = np.abs(Aminus)

        imbalance = Aplus - absAminus
        denom = Aplus + absAminus
        asym_norm = safe_div(imbalance, denom)
        ratio_pm = safe_div(Aplus, absAminus)

        df["imbalance"] = imbalance
        df["asym_norm"] = asym_norm
        df["ratio_pm"] = ratio_pm

    # --- Two-lobe subset (for lobe-geometry + lock tasks) ---
    two = df[df["topology"] == "two_lobe"].copy()
    print(f"Two-lobe rows: {len(two)} (of {len(df)} total)")

    # feature matrices aligned with 'df' / 'two'
    idx_all = df.index.to_numpy()
    idx_two = two.index.to_numpy()
    pos_all = {ix: i for i, ix in enumerate(idx_all)}
    sel_two = np.array([pos_all[ix] for ix in idx_two], dtype=int)

    # "CAF-adjacent" curvature vector (kept for sanity checks)
    X_d2 = D2[sel_two]  # shape (N_two,4)

    # "IE-shape" features (preferred for real tasks)
    X_shape_two = np.concatenate([Yz[sel_two], D1z[sel_two], D2z[sel_two]], axis=1)  # (N_two, 6+5+4=15)
    X_shape_all = np.concatenate([Yz, D1z, D2z], axis=1)                              # (N_all,15)

    # ------------------------------------------------------------
    # 0) Leakage/identity check: is imbalance an affine functional of d2?
    # ------------------------------------------------------------
    if have_lobes and "imbalance" in two.columns:
        y_imb = pd.to_numeric(two["imbalance"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(X_d2).all(axis=1) & np.isfinite(y_imb)
        if np.sum(m) >= 4:
            A = np.column_stack([np.ones(np.sum(m)), X_d2[m]])
            coef, *_ = np.linalg.lstsq(A, y_imb[m], rcond=None)
            resid = y_imb[m] - (A @ coef)
            print("\n== imbalance linear identity check (two_lobe) ==")
            print("coef (intercept, d2_0..d2_3):", coef)
            print("max_abs_resid:", float(np.max(np.abs(resid))))
            print("rmse_resid:", float(np.sqrt(np.mean(resid**2))))
        else:
            print("\n[skip] imbalance identity check: insufficient finite rows")

    # ------------------------------------------------------------
    # 1) LOOCV regressions (two_lobe): d2 -> lobe-geometry (sanity only)
    # ------------------------------------------------------------
    if have_lobes:
        print("\n== LOOCV linear regression (SANITY): d2 -> lobe-geometry (two_lobe) ==")
        for target in ["asym_norm", "ratio_pm", "rho_A", "imbalance", "D_abs"]:
            if target not in two.columns:
                print(f"[skip] target missing: {target}")
                continue
            y = pd.to_numeric(two[target], errors="coerce").to_numpy(dtype=float)
            r2, rmse = loocv_linear_r2(X_d2, y)
            print(f"LOOCV linear: {target:10s}  r2={r2:.6f}  rmse={rmse:.6g}")
    else:
        print("\n[skip] lobe-geometry regressions: A_plus/A_minus not present in CSV")

    # ------------------------------------------------------------
    # 2) LOOCV classification: state from IE-shape (all rows)
    # ------------------------------------------------------------
    print("\n== LOOCV nearest-centroid: state from IE-shape (all rows) ==")
    labels_state = df["state"].to_numpy(dtype=object)
    acc_state_all, cm_state_all = loocv_nearest_centroid_accuracy(X_shape_all, labels_state)
    print(f"acc={acc_state_all:.6f}")
    print("Confusion (state):")
    print(cm_state_all.to_string())

    # 2b) C vs C* restricted (often cleaner)
    keep_cc = np.isin(labels_state, ["C", "C*"])
    if np.sum(keep_cc) >= 4:
        acc_cc, cm_cc = loocv_nearest_centroid_accuracy(X_shape_all[keep_cc], labels_state[keep_cc])
        print("\n== LOOCV nearest-centroid: state(C vs C*) from IE-shape ==")
        print(f"acc={acc_cc:.6f}")
        print("Confusion (C vs C*):")
        print(cm_cc.to_string())

    # ------------------------------------------------------------
    # 3) LOOCV classification: lock from IE-shape (two_lobe only)
    # ------------------------------------------------------------
    if len(two) > 0 and "lock" in two.columns:
        print("\n== LOOCV nearest-centroid: lock from IE-shape (two_lobe) ==")
        lock = two["lock"].astype(str).to_numpy(dtype=object)
        keep_lock = np.isin(lock, ["locked", "unlocked"])
        if np.sum(keep_lock) >= 4:
            acc_lock, cm_lock = loocv_nearest_centroid_accuracy(X_shape_two[keep_lock], lock[keep_lock])
            print(f"acc={acc_lock:.6f}")
            print("Confusion (lock):")
            print(cm_lock.to_string())
        else:
            print("[skip] lock classification: insufficient locked/unlocked rows")
    else:
        print("\n[skip] lock classification: no two_lobe rows or missing 'lock' column")

if __name__ == "__main__":
    main()
    