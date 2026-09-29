# EEGNet: pooled vs subject-routed heads (BNCI2014_001)

4-class MI, session 1 → session 2, balanced accuracy per 4-s window (chance 0.25). 5 seeds, 150 epochs, best epoch on session-1 run 5. Fingerprint EEGNet subject-ID accuracy on session 2: 0.886 ± 0.046 (chance 0.111).

| Arm | Balanced acc (mean ± SD) | Δ vs pooled, paired (mean ± SD) | Seeds > pooled |
|---|---|---|---|
| pooled | 0.692 ± 0.007 | — | — |
| multihead_oracle | 0.661 ± 0.009 | -0.030 ± 0.009 | 0/5 |
| multihead_hard | 0.640 ± 0.009 | -0.052 ± 0.010 | 0/5 |
| multihead_soft | 0.645 ± 0.009 | -0.047 ± 0.010 | 0/5 |
| multihead_uniform | 0.595 ± 0.013 | -0.096 ± 0.011 | 0/5 |

Runtime: 4735.6 s.
