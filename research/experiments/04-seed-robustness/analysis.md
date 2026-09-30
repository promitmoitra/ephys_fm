# Protocol 04 analysis: seed robustness (CONFIRMATORY)

Results: `results/seeds.md`. Combiners fitted once on the seed-0 cross-fitted dev bank,
applied unchanged to EEGNet experts from training seeds 0, 1, 2.

- **Prediction confirmed, and the gain is larger than on seed 0:** C3 − R0 = +0.014 / +0.020 /
  +0.024 (seeds 0 / 1 / 2), every person-bootstrap CI above zero, 10–15 people better vs
  5–7 worse. C1: +0.014 / +0.020 / +0.021.
- **Mean over 3 seeds:** EEGNet alone 0.898 / NLL 0.253 → C1 0.917 / 0.221, **C3 0.917 / 0.215**;
  plain average 0.914 / 0.332.
- Seed 0 had the strongest EEGNet experts (0.908 vs 0.893 / 0.894); the weaker the EEGNet
  experts, the more the classical expert adds, consistent with the dev folds (largest gain
  on the fold where EEGNet was weakest).
- The combiners' weights were fitted on seed-0 dev experts and transfer across seeds
  unchanged.

Decision: C3's adoption is upgraded from "one seed" to "robust over 3 seeds".
