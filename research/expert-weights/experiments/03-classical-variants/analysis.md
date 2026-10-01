# Protocol 03 analysis: classical variants (H9) and reliability weights (H8) (CONFIRMATORY)

Results: `results/variants.md`. Cross-fitted bank, true ID, LOPO.

## H9: variants

| Prediction | Outcome |
|---|---|
| ts_fb5 alone > ts alone | Barely: 0.684 vs 0.680 (ts_fb5_w0 0.697 is the best expert alone). The filter bank does not rescue the ~9 people at chance. |
| ts_broad / ts_low strongest alone, add least | **Refuted in part:** they are the *weakest* alone (0.649, 0.587) and add least (ts_low weight 0.19). |
| ts_C0.1 better calibrated at equal accuracy | **Supported, and then some:** mean confidence 0.607 at 0.668 accuracy (ts: 0.755 at 0.680), i.e. now slightly under-confident. That alone flips the plain average from −0.008 (ts) to **+0.007** (ts_C0.1, 12 / 6). |

By the locked rule (best H3-combined accuracy), the winner is **ts_C0.1**: H3 0.879 (+0.004,
**15 better / 6 worse**, the best people ratio of any rule), NLL 0.285. The variants are
within 0.004 of each other, so this is a choice among equals. What matters is regularisation
(calibration), not bands or windows.

## H8: reliability weights

- Each person's reliability (accuracy of a one-run model on another calibration run) predicts
  their held-out ts accuracy almost perfectly across people (r = 0.94–0.99). Whether ts
  works is a stable person trait.
- The learned slope is large and positive (w1 ≈ 1.7–2.1: ts weight ~0.14 at rel 0.5, ~1.0 at
  rel 0.9), as predicted.
- H8 beats H3 on every variant except ts_low (whose slope is ~0). On ts: 0.881 vs 0.876,
  **NLL 0.269 vs 0.288**, the lowest NLL of the whole loop. On the winning variant ts_C0.1:
  0.883 vs 0.879 (+0.004), NLL 0.278 vs 0.285. The locked criterion (≥ 0.005 over H3 on the
  winning variant) **narrowly fails** (+0.004); on base ts it passes (+0.005).
- Bootstrap 95% CIs vs EEGNet alone: H8-ts +0.006 [−0.005, +0.018]; H8-ts_C0.1 +0.008
  [−0.003, +0.019]; H3-ts_C0.1 +0.004 [−0.008, +0.014]; R1-ts_C0.1 +0.007 [−0.001, +0.015].

## Synthesis

Adding a per-person classical expert to the EEGNet expert is worth roughly **+0.005 ± 0.01**
on dev, and it is **positive only if the classical expert is not allowed to be over-confident**.
Three routes get there: regularise it (C = 0.1), down-weight it in log space (log-linear
pooling), or weight it by the person's calibration reliability (H8). Reliability weighting
is the most principled (it uses a stable person trait) and the best by NLL.

**Checkpoint 2 candidates (pre-registered in its protocol):** H4-ts (protocol 02 rule),
H3-ts_C0.1 (H9 rule), H8-ts_C0.1 and H8-ts (H8, best dev accuracy and best dev NLL).
