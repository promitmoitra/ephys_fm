# Findings: loop B (learned expert weights)

## Current understanding

- **ts can add to EEGNet, but the evidence is noisy.** On the hidden runs R4–R6 (seed 0),
  combining each person's EEGNet expert with their Riemannian tangent-space (ts) expert lifts
  oracle-ID accuracy from 0.908 to 0.921. That holds for both the plain average and a
  log-linear stack. On the R3 dev bank (experts trained on R1–R2), the plain average instead
  *hurt* (0.887 → 0.869). With 40 windows per person, the dev bank cannot resolve effects
  of ±0.015, so a cross-fitted dev bank (all three calibration runs held out in turn) is
  being built.
- **Why a plain average is the wrong rule:** ts is over-confident (mean confidence 0.754 at
  0.681 accuracy), while the fine-tuned EEGNet is calibrated (0.883 / 0.887). A linear
  average lets chance-level but confident ts votes overrule EEGNet, and it wrecks NLL
  (0.325 vs 0.235 on R4–R6). Log-linear pooling with a bias (weights ≈ 0.83 EEGNet, 0.40 ts)
  keeps the accuracy gain and gives the best NLL of any rule (0.214).
- **ts is strongly person-specific:** R3 accuracy ≥ 0.92 for 6 people and ≈ chance for 9.
  Its value therefore probably lies in a per-person reliability weight, not a global one.
- **Routing is no longer the bottleneck:** loop A's filter-bank covariance fingerprint
  (0.996 on R4–R6) brings the soft mixture to the oracle ceiling, so oracle-ID expert
  quality is what loop B should improve.

## Patterns and insights

- Confidence gating cannot help: where EEGNet is unsure, ts is even less accurate
  (0.61 vs 0.69 on R3).
- CSP adds nothing (learned weight ≈ 0.04; including it in a plain average costs 0.12 on dev).

## Lessons and constraints

- R4–R6 are never used for decisions; confirm only pre-registered candidates.
- Dev experts are trained on 2 calibration runs, test experts on 3.
- A 40-window-per-person dev set gives a paired SE of ≈ 0.01 on rule differences, which is
  too coarse to rank combiners whose differences are 0.01–0.02.

## Open questions

- Is the dev/test disagreement noise, an R3-specific effect, or ts gaining more than EEGNet
  from a third training run? (Protocol 02 per-fold results.)
- Per-person reliability weights (H8) and a stronger classical expert (H9, filter-bank ts).
