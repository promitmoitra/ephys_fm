# H3.1: do covariance fingerprints survive drift? (harder dev benchmarks)

**Status:** CONFIRMATORY. Parent: H3.

## Why

H3's filter-bank Riemannian fingerprint scored 1.000 on R3 (NLL 0.027), which saturates the
inner-loop metric. The hidden runs R4–R6 are 1–3 runs farther from the training data than R3,
and the sealed phase's hidden sessions are probably other days. On BNCI, Riemann won within a
session (0.999) and lost across days (0.872 vs EEGNet 0.909, best-epoch). Two harder dev
benchmarks, neither touching R4–R6:

- **D1, Dreyer run distance:** train on R1 only (40 windows per person); test on R2 (gap 1)
  and R3 (gap 2). The drop from R2 to R3 measures decay with time within a session.
- **D2, BNCI cross-day:** 9 people, train on session 1 (all 6 runs, 2,592 windows), test on
  session 2 (a different day, 2,592 windows). 9-way.

## Mechanism → prediction

The covariance fingerprint captures stable within-session head/cap geometry. Across runs within
a session that geometry barely changes. Across days, cap re-placement changes it.
**Predictions:**
1. D1: `ts_fb` loses ≤ 0.02 from R2 to R3 and stays ≥ 0.95 on R3 even with one training run.
   EEGNet loses more, because 40 windows per person is little data for a CNN.
2. D2: `ts_broad` reproduces ≈ 0.87. `ts_fb` beats `ts_broad` (band-specific spectra add stable
   identity cues). Whether `ts_fb` beats EEGNet (fixed 150 epochs, last epoch) is open.
   Prior: no, because cross-day shifts the covariances themselves.

## What

Models: `eegnet` (H0 recipe: 150 epochs, last epoch, seeds 0/1/2, 4 threads), `ts_broad_C1`,
`ts_fb_C1` (H3 recipes; C=1 won on NLL for both feature sets). Same locked metrics.

## Decision use

- If `ts_fb` holds on D2 (≥ EEGNet), it becomes the lead candidate for confirmation.
- If it collapses on D2, the lead candidate is an ensemble (H6), and D2 becomes a required
  gate for every later hypothesis.
