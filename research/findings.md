# Findings: loop B (learned expert weights)

## Current understanding

**Adding each person's Riemannian tangent-space (ts) expert to their EEGNet expert helps,
if the ts expert is prevented from being over-confident.** On the hidden runs R4–R6
(seed 0, oracle ID), every log-linear combination fitted on calibration data beats EEGNet
alone by +0.012 to +0.014 (0.908 → 0.920–0.922; person-bootstrap 95% CI excludes zero), and
lowers NLL from 0.235 to 0.209–0.213. Soft-routed with loop A's filter-bank fingerprint, the
recommended combiner reaches **0.921** (EEGNet alone 0.907).

**Recommended combiner (C3):** reliability-weighted log-linear pooling,
log p ∝ 0.81·log p_eeg + (0.96 + 1.70·(rel_k − 0.5))·log p_ts + b, with a regularised ts
expert (8–30 Hz, 0.5–4 s, OAS → tangent space → logistic regression, C = 0.1) and rel_k =
person k's run-to-run ts accuracy on calibration data. **C1** (fixed log-linear weights
0.83 / 0.42 on the original ts) is statistically equivalent and simpler.

**Why a plain average was the wrong rule, and why it sometimes looked fine.** The original ts
expert is over-confident (mean confidence 0.755 at 0.680 accuracy). The fine-tuned EEGNet
expert is calibrated (0.883 / 0.887). A linear average therefore lets confident but
chance-level ts votes overrule EEGNet for people whose motor imagery ts cannot decode. Its
effect swings with how strong EEGNet happens to be on a run: −0.018 to +0.010 across the
three dev folds, +0.014 on R4–R6. Log-space pooling, regularising ts (C = 0.1 makes it
slightly under-confident), or weighting it by reliability all remove that failure mode.

**ts's usefulness is a stable person trait.** A person's ts accuracy from one calibration run
to another predicts their held-out ts accuracy almost perfectly across people (r = 0.94–0.99).
ts is near-perfect for ~6 of 21 people and at chance for ~9. Reliability weighting uses this
directly: learned ts weight ≈ 0.1 at chance-level reliability, ≈ 1.0 at 0.9.

## Key results

| Rule | Dev: cross-fitted R1–R3 (2,520 windows) | Test: R4–R6 (2,520) |
|---|---|---|
| EEGNet only | 0.875 / NLL 0.303 | 0.908 / 0.235 |
| Plain average EEGNet + ts | 0.867 / 0.371 | 0.921 / 0.325 |
| C1 log-linear (ts) | 0.877 / 0.287 | 0.922 / 0.212 |
| C3 reliability-weighted (ts_C0.1) | 0.883 / 0.278 | 0.921 / 0.209 |

## Patterns and insights

- Resolution matters more than combiner choice: a 40-window-per-person dev set (R3 only) gave
  the wrong sign for the ts effect. Even 2,520 cross-fitted windows give ±0.011 CIs, and all
  sensible combiners sit inside that band. The consistent signal across every split is the
  **NLL** improvement of log-space pooling.
- Confidence gating cannot help: where EEGNet is unsure, ts is even less accurate.
- CSP adds nothing (learned weight ≈ 0.02–0.04); filter banks, other windows and low
  frequencies do not make a better classical expert. Regularisation (calibration) is what
  matters.
- Routing is solved by loop A: with its fingerprint, soft-routed = oracle within 0.001.

## Lessons and constraints

- R4–R6 are never used for decisions. Confirm only pre-registered candidates; fix the adoption
  rule before the test run (checkpoint 2 did).
- Don't rank combiners on a 40-window-per-person dev set. Use the cross-fitted bank
  (`bank_xfit_seed0.npz`).
- Dev experts are trained on 2 calibration runs, test experts on 3.
- `cmd && ps ... && launch` fails silently when `ps` finds nothing (exit 1). Launch separately.

## Open questions

- Seed robustness: is the gain present for other EEGNet training seeds? (Protocol 04, running.)
- Integration: C3 needs per-person tangent-space + LR weights as torch tensors in
  `submission.py`. Loop A's H9 export shows the pattern works.
- Does this transfer to the sealed phase's data (cross-day, 3 classes, EOG/EMG channels)?
