# Checkpoint 2 analysis

Results: `results/confirm.md`.

- **No sign flip:** every candidate beats EEGNet alone on R4–R6 (oracle +0.012 to +0.014).
  For all log-linear rules the person-bootstrap 95% CI excludes zero
  (C1 +0.014 [+0.004, +0.025], C3 +0.014 [+0.003, +0.025]); the plain average's does not
  (+0.014 [−0.002, +0.031]), because it gains as much on average but more erratically.
- **NLL:** every log-linear rule beats EEGNet alone (0.209–0.213 vs 0.235); the plain
  average is much worse (0.325). The reliability-weighted and regularised variants
  (C2, C3: 0.209) are the best calibrated.
- **Soft routing:** with loop A's filter-bank fingerprint, C3 reaches 0.921 (vs 0.907 for
  EEGNet alone), i.e. the gain survives routing intact. With the packaged EEGNet
  fingerprint, 0.909 vs 0.901.
- The candidates are indistinguishable from each other on test (0.920–0.922), as on dev.

**Adoption (rule fixed in the protocol):** no candidate fell below R0, so the recommended
combiner is **C3: reliability-weighted log-linear pooling of each person's EEGNet expert
with their regularised tangent-space expert (ts_C0.1)**:
log p ∝ 0.81·log p_eeg + (0.96 + 1.70·(rel − 0.5))·log p_ts + b, with rel = the person's
calibration-run-to-run ts accuracy. **C1 (log-linear, fixed weights 0.83 / 0.42, base ts)**
is statistically equivalent and simpler to ship (no reliability fits); it is the fallback
if integration cost matters.

**Caveats.** One training seed for every EEGNet model; one dataset (Dreyer), where the
EEGNet decodes mostly a cue-locked response. The dev evidence alone was inconclusive
(+0.005 ± 0.01); the adoption rests on dev + test agreeing in sign and the mechanism
(calibration of the classical expert) being clear.
