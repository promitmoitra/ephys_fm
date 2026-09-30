# Research Findings (loop A: fingerprint)

## Research Question

Which per-window person-ID model gives the most accurate and best-calibrated p(person | window)
for the Track 2 evaluation participants, so that soft routing in the fingerprint mixture keeps
the oracle-ID gain, both within a session (the Dreyer simulation) and across days (the sealed
phase's likely regime)?

## Current Understanding

**A filter-bank covariance fingerprint solves person ID on this problem, and it is shippable.**
Per-band spatial covariance (OAS) → Riemannian tangent space → multinomial logistic regression
identifies the 21 Dreyer evaluation people from a single 4-s window at 0.996 on the hidden runs
R4–R6. The shipped EEGNet fingerprint scored 0.744. With it, the soft-routed mixture reaches the
oracle-ID ceiling (0.907 vs 0.9075; shipped 0.901). It is deterministic and needs no epoch
selection. It trains in about a minute on one CPU thread and is well calibrated (ECE 0.02–0.05)
without temperature scaling.

**Why it works, and why the broadband version did not transfer across days.** Identity is
carried by the spatial structure of oscillatory activity in the **alpha/beta range**. On BNCI
across days, 8–13 Hz alone identifies 9 people at 0.975. Delta (1–4 Hz) is equally
discriminative within a session (0.989 on Dreyer R3) but **day-specific** (0.710 across days):
it reflects eye movements, drifts and electrode state, which change between days. A broadband
covariance is dominated by that low-frequency power, which is why it reached only 0.872 across
days. Splitting into bands lets the classifier weight the stable bands: 6 bands give 0.974, and
18 × 2-Hz bands from 4–40 Hz (no delta) give 0.992 / NLL 0.035. The finer bank is consistent
with individual spectral peaks (e.g. the individual alpha frequency) being part of the
signature. H3.3 separates the "no delta" effect from the "finer resolution" effect.

**The EEGNet fingerprint was weaker than it looked.** Under the locked protocol (fixed 150
epochs, last epoch, no selection on R3), EEGNet reached 0.618 on R3 with NLL 2.26 and ECE 0.24
(seed 0). Its R3 accuracy swung between 0.59 and 0.72 from epoch 50 to 100. The shipped 0.857
came from picking the best epoch on R3 itself. With 80 windows per person, a CNN overfits the
identity task, while second-order statistics with a linear classifier do not.

## Key Results

| Benchmark | EEGNet (fixed budget) | ts_broad | ts_fb (6 bands) | fine18 |
|---|---|---|---|---|
| Dreyer R1–R2 → R3 (locked dev) | 0.618 / 2.256 (seed 0) | 0.987 / 0.107 | **1.000 / 0.027** | 1.000 / 0.026 |
| Dreyer R1 → R3 | (running) | 0.976 / 0.188 | 1.000 / 0.052 | 1.000 / 0.051 |
| BNCI day 1 → day 2 | (running) | 0.872 / 0.403 | 0.974 / 0.071 | **0.992 / 0.035** |
| **Dreyer R4–R6 (confirm-1)** | 0.744 (shipped, epoch on R3) | 0.990 / 0.080 | **0.996 / 0.044** | – |
| Soft mixture on R4–R6 | 0.901 (shipped) | 0.908 | **0.907** (oracle 0.9075) | – |

(balanced accuracy / NLL; 21-way on Dreyer, 9-way on BNCI)

- **H9, shippable:** each band's zero-phase filter on a fixed window is an exact linear operator,
  i.e. one T × T matrix. OAS, tangent space (eigh) and the folded scaler + LR run in torch. The
  torch module matches sklearn to 3.9e-6 (float32) with identical argmax, and is 5.7 MB.

## Patterns and Insights

- Within a session, almost any per-band covariance identifies a person (every single band
  scored ≥ 0.97 on Dreyer). Only across days do bands differ, so **cross-day data is the only
  informative dev signal left**. The locked R3 metric and D1 are saturated.
- What is stable across days is oscillatory spatial structure, not low-frequency power. The
  same logic predicts which features survive the sealed phase's day gaps.
- Calibration comes for free with a linear model on well-conditioned features. Temperature
  scaling (H4) is unnecessary for `ts_fb` within a session.

## Lessons and Constraints

- R4–R6 are never used for decisions. R3 is the dev set, with fixed epoch budgets and the last
  epoch. Confirmation happens only for pre-registered candidates (confirm-1 was committed
  before it ran).
- The R3 metric is saturated for covariance models. Decide on the cross-day benchmarks, and
  validate bank choices on held-out cross-day datasets (H3.3) to avoid overfitting BNCI's 9
  people.
- Don't add a broadband or delta covariance to a fingerprint meant to work across days
  (`fb_plus_broad` 0.963 < `fb6` 0.974 on BNCI).
- `pgrep -f` / `$!` match the launching shell. Wait on the Python PID from `ps -C python`.
- 4 CPU threads for loop A. Covariance models run on 1 thread
  (`OMP_NUM_THREADS=1`), EEGNet on 4.

## Open Questions

- Does the finer bank's cross-day gain hold on other datasets (H3.3 held-out: BNCI2015_001,
  Zhou2016)?
- The sealed phase has 47 channels (incl. 2 EOG, 2 EMG), probably 3 calibration days, and
  10 people. Should EOG/EMG channels be excluded from the fingerprint (they carry day-specific
  artifacts)? Does training on several calibration days make the fingerprint more robust?
- Is anything left for EEGNet + Riemann ensembles (H6) when Riemann is already at 0.99+?

## Optimization Trajectory

R3 dev: EEGNet 0.618 (fixed budget) → ts_broad 0.987 → ts_fb 1.000 (saturated).
BNCI cross-day: EEGNet 0.909 (old, best-epoch) → ts_broad 0.872 → ts_fb 0.974 → fine18 0.992.
Dreyer hidden (confirm-1): 0.744 → 0.996; mixture 0.901 → 0.907 (oracle 0.9075).
