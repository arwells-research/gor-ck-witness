# gor-ck-witness

Authority: Sigma-Order Phenomenological Diagnostic Repository  
Classification: [L4-DIAG] Diagnostic and Adjudication Instrument (Open Data)  
Status: Active / v1 (curated public release)  
Author: A. R. Wells  
Affiliation: Dual-Frame Research Group  
License: MIT (code), CC BY 4.0 (docs)  
Contact: No solicitation for correspondence or media contact  
Paper email: arwells.research@proton.me  

---

## Purpose

This repository builds a **closure/phase diagnostic (“CK witness”)** from tabulated atomic ionization energies (NIST ASD), restricted to **p-block columns (p1..p6)** across periods (2–6).

It does **not** attempt to model chemistry or atomic structure directly.

Instead, it provides a **reproducible structural witness pipeline**:

- Extract ionization-energy shape profiles
- Cluster IE-profile geometry into coarse regimes (k-means)
- Evaluate state-conditioned collapse diagnostics
- Perform holdout generalization tests (train on Periods 3–5, evaluate Period 6)
- Test non-randomness via permutation
- Evaluate centroid stability and resolution limits

The output is a **model-selection result about regime granularity**, not a chemical fit.

See `docs/RESULTS.md` for the full diagnostic chain and conclusions.

---

## Core Diagnostic Questions

1. Does IE-profile clustering produce regime-conditioned collapse structure?
2. Is the collapse association non-random?
3. Are holdout periods geometrically in-distribution?
4. Does increasing k improve generalization uniformly?
5. Is there a single global regime resolution that simultaneously optimizes all periods?

The results show:

- Collapse–regime association is highly non-random (permutation test).
- Period 6 is not geometrically out-of-distribution.
- Increasing k redistributes explanatory power rather than uniformly improving it.
- Performance stabilizes at k=4.
- No single low-k partition simultaneously optimizes lighter and heavier periods.

---

## Repository Layout

### scripts/

Core reproducible pipeline:

- build_pblock_ionization_csv_from_nist.py  
  Fetch + extract ion-stage IE data into wide per-period CSV.

- build_closure_phase_diagram.py  
  Construct phase diagram dataset from raw wide CSVs.

- cluster_ie_shape_and_test_collapse.py  
  Perform IE-shape clustering and evaluate collapse diagnostics.

- closure_state_conditioned_collapse.py  
  Compute state-conditioned collapse summaries.

- train_k3_assign_holdout.py  
  Train on selected periods and evaluate holdout generalization.

- permutation_test_mixture_r2.py  
  Permutation test for collapse–regime association.

- kmeans_centroid_stability.py  
  Evaluate centroid distance and stability diagnostics.

- summarize_ie_clusters_yz.py  
- stability_ie_clusters_yz.py  

### scripts/dev/

Exploratory and legacy tools retained for transparency (not part of the mainline pipeline).

---

### data/raw/

Wide per-ion-stage CSV datasets derived from NIST ASD.

Each file contains:
- period
- p1..p6 IE values
- labels and metadata

---

### data/derived/

Curated deterministic outputs used in the published results:

- closure_phase_diagram_v2*.csv
- closure_mixture_only_prediction*.csv
- period_r2_summary.csv
- k3/k4/k5_train_* files
- mixture_only_by_absdelta_obs_*.csv
- qc_by_period_best_strict_canon.csv
- ck_area_null_perm.csv

These files reproduce all tables in `docs/RESULTS.md`.

---

## Quickstart

1. Build ion-stage dataset (example: IE2, ion_charge=1)

    PYTHONPATH=. python3 scripts/build_pblock_ionization_csv_from_nist.py \
        --ion_charge 1 \
        --out_csv data/raw/nist_pblock_ie2_periods_2to6.csv

2. Build closure phase diagram

    PYTHONPATH=. python3 scripts/build_closure_phase_diagram.py \
        --glob "data/raw/nist_pblock_ie*_periods_*.csv" \
        --out_csv data/derived/closure_phase_diagram_v2.csv

3. Perform clustering and collapse diagnostics

    PYTHONPATH=. python3 scripts/cluster_ie_shape_and_test_collapse.py
    PYTHONPATH=. python3 scripts/closure_state_conditioned_collapse.py

4. Holdout generalization test

    PYTHONPATH=. python3 scripts/train_k3_assign_holdout.py

5. Permutation test

    PYTHONPATH=. python3 scripts/permutation_test_mixture_r2.py \
        --nperm 20000 --seed 0

---

## Interpretation Discipline

- Binned R² measures structure of mean collapse profiles.
- Element-level R² measures predictive fidelity under regime assignment.
- Permutation tests establish non-random association.
- k-selection is treated as a model-selection problem, not as proof of chemical mechanism.
- Fitted curves are descriptive only.

---

## What “Success” Means

- Deterministic outputs given (dataset, k, seed, nperm)
- Reproducible holdout evaluation
- Explicit ruling-out of:
  - joint clustering artifacts
  - geometric out-of-distribution explanations
  - single-period anomaly narratives
- Clear resolution-stabilization at k=4

---

## License

- Code: MIT  
- Documentation: CC BY 4.0  

---

## Citation

If publishing results derived from this repository:

- Cite NIST ASD.
- Optionally cite this repository as:

  A. R. Wells, Dual-Frame Research Group, gor-ck-witness (software), MIT license.
  