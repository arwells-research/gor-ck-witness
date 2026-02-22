# REPRODUCIBILITY — gor-ck-witness

Authority: Sigma-Order Phenomenological Diagnostic Repository  
Classification: [L4-DIAG] Reproducibility Protocol  
Status: Curated Public Release (v2)  
Author: A. R. Wells  
Affiliation: Dual-Frame Research Group  
License: CC BY 4.0  

---

# 1. Environment

Tested with:

- Python ≥ 3.10
- NumPy
- Pandas
- scikit-learn

Recommended setup:

    python -m venv .venv
    source .venv/bin/activate
    pip install -e .

All commands assume execution from repository root with:

    PYTHONPATH=.

---

# 2. Domain and Determinism

All curated datasets in this repository are restricted to:

- p-block columns (p1..p6)
- Periods 2–6
- Specified ion_charge values

Reproducibility requires fixing:

- ion_charge
- k (cluster count)
- random seed
- nperm (permutation count)

Published results use:

- k ∈ {3, 4, 5}
- seed = 0 (unless otherwise noted)
- nperm = 20000

All outputs are deterministic given these inputs.

---

# 3. Raw Data Construction

To regenerate raw IE datasets:

Example (IE2):

    PYTHONPATH=. python3 scripts/build_pblock_ionization_csv_from_nist.py \
        --ion_charge 1 \
        --out_csv data/raw/nist_pblock_ie2_periods_2to6.csv

Repeat for required ion stages.

All stages should use the same period restriction (2–6).

If curated raw CSVs are already present in `data/raw/`, this step may be skipped.

---

# 4. Build Closure Phase Diagram

Aggregate raw datasets:

    PYTHONPATH=. python3 scripts/build_closure_phase_diagram.py \
        --glob "data/raw/nist_pblock_ie*_periods_2to6.csv" \
        --out_csv data/derived/closure_phase_diagram_v2.csv

This produces the base phase diagram table used for clustering and collapse diagnostics.

---

# 5. Clustering and Collapse Diagnostics

Run IE-profile clustering and state-conditioned collapse:

    PYTHONPATH=. python3 scripts/cluster_ie_shape_and_test_collapse.py
    PYTHONPATH=. python3 scripts/closure_state_conditioned_collapse.py

Ensure that k and seed match the reported configuration.

Outputs include:

- closure_mixture_only_prediction*.csv
- mixture_only_by_absdelta_obs_*.csv

---

# 6. Holdout Generalization (Primary Test)

Train on Periods 3–5 and evaluate Period 6:

    PYTHONPATH=. python3 scripts/train_k3_assign_holdout.py \
        --k 3 --seed 0

Repeat for k = 4 and k = 5.

Key artifact:

- period_r2_summary.csv

Verify that R² values match those reported in `docs/RESULTS.md`.

---

# 7. Permutation Test

Evaluate non-random collapse–regime association:

    PYTHONPATH=. python3 scripts/permutation_test_mixture_r2.py \
        --k 3 --nperm 20000 --seed 0

Repeat for k = 4 if required.

Verify reported p-values.

---

# 8. Centroid Stability / Distance Check

Optional but part of the diagnostic chain:

    PYTHONPATH=. python3 scripts/kmeans_centroid_stability.py \
        --k 3 --seed 0

Confirm that Period 6 centroid distances are not elevated relative to training periods.

---

# 9. Expected Deterministic Outputs

Given fixed:

- dataset (Periods 2–6 only)
- k
- seed
- nperm

The following files should reproduce exactly:

- closure_phase_diagram_v2*.csv
- closure_mixture_only_prediction*.csv
- mixture_only_by_absdelta_obs_*.csv
- k3/k4/k5_train_* artifacts
- period_r2_summary.csv
- qc_by_period_best_strict_canon.csv
- ck_area_null_perm.csv (if used)

Numerical differences indicate:

- mismatched seed
- inconsistent ion_charge selection
- altered period restriction
- or environment/library differences

---

# 10. Verification Checklist

Reproduction is successful if:

- Element-level R² for k=3 (Period 6 holdout) matches reported value.
- k=4 redistributes explanatory power rather than uniformly improving it.
- k=5 produces stabilization (no further improvement).
- Permutation p-value is small (non-random association).
- Centroid distances for Period 6 are not elevated.

---

# 11. Interpretation Boundary

This protocol ensures reproducibility of:

- clustering,
- collapse diagnostics,
- holdout evaluation,
- permutation tests,
- model-selection conclusions.

It does not validate any mechanistic or physical interpretation beyond the diagnostic layer.
