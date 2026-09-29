# EEGNet: pooled vs per-subject fine-tuning with oracle ID (BNCI2014_001)

4-class MI, session 1 → session 2, balanced accuracy per 4-s window (chance 0.25). 5 seeds; pooled model 150 epochs, then 50 fine-tuning epochs per subject on its session-1 runs 0–4; `_sel` picks the epoch on the subject's run 5 (epoch 0 = pooled included), `_fixed` keeps the last.

| Arm | Balanced acc (mean ± SD) | Δ vs pooled, paired (mean ± SD) | Seeds > pooled |
|---|---|---|---|
| pooled | 0.692 ± 0.007 | — | — |
| ft_head_sel | 0.716 ± 0.005 | +0.024 ± 0.002 | 5/5 |
| ft_head_fixed | 0.728 ± 0.008 | +0.036 ± 0.007 | 5/5 |
| ft_all_sel | 0.722 ± 0.008 | +0.030 ± 0.010 | 5/5 |
| ft_all_fixed | 0.744 ± 0.005 | +0.053 ± 0.003 | 5/5 |

Runtime: 2316.0 s.
