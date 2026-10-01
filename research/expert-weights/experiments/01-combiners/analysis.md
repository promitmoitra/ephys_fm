# Protocol 01 analysis (CONFIRMATORY unless marked)

Full table: `results/sweep.md`. Dev bank, true ID, 840 windows (one window ≈ 0.0012).

**Sanity:** all experts above chance on R3; EEGNet experts 0.887 (vs 0.908 on R4–R6 with one
more calibration run), ts 0.681 (0.688), csp 0.645 (0.671). Pooled 0.860 → control 0.888.

## Results vs predictions

| Prediction | Outcome |
|---|---|
| Learned EEGNet weight in [0.5, 1), H1 ≥ R1 by ≤ 0.01 | Weight 0.915 (softmax(1.19, −1.19)); H1 = 0.887 vs R1 0.869: **+0.018**, but H1 only matches EEGNet alone (R0 0.887). Partly wrong: ts is nearly switched off. |
| EEGNet over-confident, T > 1 | **Refuted.** EEGNet is calibrated (mean confidence 0.883, accuracy 0.887); learned T < 1 (sharpen). The over-confident expert is **ts** (confidence 0.754, accuracy 0.681). CSP is far worse (NLL 1.31). |
| Per-person (H2) < global | **Refuted, marginally:** H2 0.891 ± 0.002 is the best number, +0.004 over H1 (5 people); within noise. |
| CSP gets small weight, R2 < R1 | Confirmed: learned CSP weight ≈ 0.04 (H7) or −0.05 (H7b); R2 0.750. |

**Headline:** on dev, the plain EEGNet + ts average *hurts* (−0.018 vs EEGNet alone, 10 people
worse / 6 better), the opposite sign of the one-seed R4–R6 result (+0.008). Every learned
global combiner recovers EEGNet-alone accuracy (0.880–0.888) and improves NLL slightly
(0.300 vs 0.312), but none beats EEGNet alone in accuracy.

## Mechanism (EXPLORATORY analysis on the same bank)

- Error overlap: both right 61.8%, EEGNet only 26.9%, ts only 6.3%, both wrong 5.0%. ts
  rarely rescues an EEGNet error.
- On windows where EEGNet is unsure (|p − 0.5| < 0.2, n = 122), EEGNet is still more accurate
  (0.69) than ts (0.61): a confidence gate cannot find a region where ts is better (H6 ≈ H1).
- ts is strongly **person-specific**: per-person R3 accuracy ≥ 0.92 for 6 people, ≈ chance
  (≤ 0.57) for 9. A plain average hands chance-level, over-confident ts votes to 9 people.
  This is why a global weight must be small, and why a per-person *reliability* weight is
  the natural next step (H8, CAWPE-style: estimate each person's ts accuracy on calibration
  data by within-calibration CV, not on R3's 20 fold windows).

## Candidates for checkpoint 1 (by the locked rule)

R1 (reference), best global accuracy **H5a** (0.888), best global NLL **H3** (0.300).
Dev expectation: none beats EEGNet alone by more than noise.
