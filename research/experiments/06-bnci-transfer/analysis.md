# Protocol 06 analysis: zero-shot transfer to BNCI 2014-001 (CONFIRMATORY + one exploratory row)

Results: `results/bnci.md`. 4 classes, 9 people, session 1 → session 2 (a day gap), 5 seeds.

- **Prediction confirmed:** with Dreyer's coefficients unchanged, C1 +0.054 ± 0.005 and C3
  +0.052 ± 0.003 over the EEGNet experts (0.745 → 0.799 / 0.797), 5/5 seeds; NLL 0.681 →
  0.530 / 0.535.
- The plain average does *better* in accuracy here (+0.062, 5/5), though with worse NLL
  (0.590). The EXPLORATORY BNCI-fitted log-linear rule (leave-one-person-out on session 2)
  reaches +0.070 / NLL 0.490; the transferred weights capture ~77% of that.
- **Why the weights differ (exploratory):** across a day gap, the fine-tuned EEGNet expert
  is *over*-confident (mean confidence 0.79 at 0.745 accuracy), while ts is roughly
  calibrated (0.685 / 0.665). BNCI's optimal log-linear weights are ≈ 0.67 EEGNet / 0.83 ts,
  the reverse of Dreyer's within-session 0.83 / 0.42. On BNCI the classical expert is also
  relatively stronger (0.665 vs 0.745; on Dreyer 0.68 vs 0.90).
- Reliability weighting (C3) adds nothing over C1 here: all 9 people have usable ts
  reliability (0.31–0.77 on the 4-class scale, chance 0.25).

**Implication for the sealed phase** (cross-day, 3 classes): combining in log space is safe
and large across regimes (+0.014–0.024 within-session, +0.05 across days). The relative
weights, however, depend on which expert drifts more across days. Transferred coefficients
leave gain unclaimed. The sealed phase's calibration has several labelled sessions, so the
combiner can be fitted on a **leave-one-session-out** bank that matches the cross-day test
regime.
