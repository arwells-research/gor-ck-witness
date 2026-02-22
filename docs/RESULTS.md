# Results: Period-Dependent Mixture Resolution in Ionization Energy Collapse

## 1. Dataset and Scope

We analyze p-block elements in Periods 2–6 using:

- The 6-point successive ionization energy (IE) profile for each element.
- A collapse statistic \( D(|\Delta|) \), defined as the normalized drop in ionization energy as a function of charge-step magnitude.
- Unsupervised clustering in IE-profile feature space (profile z-scores, first differences, second differences).
- A regime-conditioned mixture model predicting collapse from cluster membership.

Clustering is performed independently of collapse behavior. Predictive evaluation is conducted using unbinned, element-level out-of-sample \( R^2 \).

---

## 2. Baseline k = 3 Performance

Under k = 3 clustering:

- Aggregate (binned) collapse curve fit achieves \( R^2 \approx 0.98 \).
- Element-level (unbinned) overall \( R^2 \approx 0.53 \).

A permutation test (20,000 regime-label permutations) yields \( p \le 5 \times 10^{-5} \), confirming that the regime–collapse association is highly unlikely under random assignment.

The binned and unbinned statistics measure different properties. The binned \( R^2 \) quantifies how well the model captures the aggregate collapse-vs-\(|\Delta|\) curve shape after averaging within each \(|\Delta|\) bin. The unbinned \( R^2 \) measures element-level predictive accuracy. The gap between 0.98 and 0.53 reflects within-bin variance rather than internal inconsistency in the model.

Out-of-sample performance by period (k=3):

| Period | R² |
|--------|-----|
| 3 | ~0.53 |
| 4 | ~0.82 |
| 5 | ~0.54 |
| 6 | ~0.41 |
| 2 | ~0.00 |

Performance is:

- Strong for Period 4,
- Moderate for Periods 3 and 5,
- Degraded for Period 6,
- Absent for Period 2.

Period 4 is both the most regime-predictable and the least sensitive to k changes (see below), suggesting particularly well-separated regime structure in its IE profiles relative to other periods.

---

## 3. Out-of-Sample Validation (Period 6 Holdout)

To test whether Period 6 degradation reflects joint clustering artifacts:

- Clusters were trained on Periods ≠ 6.
- Period 6 was assigned to learned centroids.
- Collapse prediction was evaluated out-of-sample.

Result:

- Period 6 \( R^2 \approx 0.41 \), matching in-sample performance.

Thus, Period 6 degradation under k=3 is not a joint-clustering artifact and reflects stable transfer limitations.

---

## 4. Geometric Evaluation in Feature Space

To test whether Period 6 profiles are geometrically out-of-distribution:

- Squared distances to assigned centroids were computed.
- Mean centroid distances were compared across periods.

Mean squared distances:

| Period | Mean Distance |
|--------|----------------|
| 3 | ~9.00 |
| 4 | ~9.19 |
| 5 | ~9.18 |
| 6 | ~9.47 |

Centroid distances are nearly identical across Periods 3–6. Period 6 does not exhibit elevated geometric displacement in IE-profile feature space.

Therefore, the degradation under k=3 arises from collapse-prediction structure rather than from geometric out-of-distribution effects.

---

## 5. Resolution Sweep (k = 3–5)

To assess whether Period 6 degradation reflects insufficient regime resolution, mixture resolution was increased.

### Out-of-sample R² by period:

| Period | k=3 | k=4 | k=5 |
|--------|------|------|------|
| 3 | ~0.53 | 0.45 | 0.45 |
| 4 | ~0.82 | 0.79 | 0.79 |
| 5 | ~0.54 | 0.59 | 0.60 |
| 6 | ~0.41 | 0.53 | 0.53 |

Observations:

- Increasing from k=3 to k=4:
  - Improves Periods 5 and 6,
  - Slightly reduces Period 3,
  - Leaves Period 4 approximately stable.
- Increasing from k=4 to k=5:
  - Produces negligible additional change across all periods.
  - The performance plateau is nearly exact numerically.

This indicates that mixture resolution stabilizes at k ≈ 4. Increasing k beyond 4 does not meaningfully improve global out-of-sample performance and introduces no new predictive structure.

The redistribution pattern is monotonic with period number: lighter periods (especially Period 3) are best captured at lower k, whereas heavier periods benefit modestly from higher resolution. This tradeoff reveals genuine internal structure in the IE-profile manifold rather than a simple underfitting artifact.

---

## 6. Integrated Finding

The full diagnostic chain yields the following conclusion:

> The IE-profile mixture manifold does not admit a single uniformly optimal low-k partition across Periods 3–6. Predictive performance redistributes across periods as regime resolution increases, and resolution stabilizes at k ≈ 4.

This finding reflects period-dependent internal structure in ionization-energy profile space rather than:

- Joint clustering artifacts,
- Geometric out-of-distribution effects,
- Or a unique Period 6 anomaly.

The investigation therefore resolves as a model-selection result concerning manifold geometry, not as a period-specific physical anomaly.

---

## 7. Boundary Case: Period 2

Period 2 consistently exhibits \( R^2 \approx 0 \) under all k values tested.

This confirms that Period 2 lies outside the shared mixture structure observed in Periods 3–6. Importantly, this failure is itself informative: the mixture framework is not sufficiently flexible to spuriously fit arbitrary structure. Its inability to explain Period 2 reinforces that the regime–collapse association observed in Periods 3–6 is structured rather than an artifact of model flexibility.

---

## Summary

This investigation demonstrates:

1. Regime-conditioned collapse structure is statistically non-random (permutation \( p \le 5 \times 10^{-5} \)).
2. Period 6 degradation under k=3 is stable out-of-sample.
3. Period 6 profiles are not geometrically out-of-distribution.
4. Increasing regime resolution redistributes predictive power across periods.
5. Mixture resolution stabilizes at k ≈ 4.
6. No single coarse partition simultaneously optimizes predictive performance across Periods 3–6.
7. Period 2 remains outside the shared mixture manifold.

The central result is a structural one: the ionization-energy profile manifold exhibits period-dependent internal structure that cannot be simultaneously resolved under a single low-k regime partition.
