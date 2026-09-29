# EEGNet: is per-subject fine-tuning real, and does it survive routing? (BNCI2014_001)

4-class MI, session 1 → session 2, balanced accuracy per 4-s window (chance 0.25), last-epoch weights. 5 seeds; pooled and fingerprint EEGNets 150 epochs; then 50 epochs of either continued pooled training on all subjects (`pooled_cont_*`, the control) or per-subject fine-tuning (`ft_*`). `head` = final layer only (lr 1e-3, BatchNorm frozen); `all` = all weights (lr 1e-4); `bn` = BatchNorm running statistics re-estimated only (AdaBN: no gradients, no labels, no epochs). Routing uses the fingerprint EEGNet: subject-ID accuracy on session 2 0.886 ± 0.046 (chance 0.111).

Δ columns are paired over seeds: mean ± SD (seeds better).

| Arm | Balanced acc | Δ vs pooled | Δ vs control (`pooled_cont_*`) |
|---|---|---|---|
| pooled | 0.692 ± 0.007 | — | — |
| pooled_cont_head | 0.686 ± 0.008 | -0.006 ± 0.004 (0/5) | — |
| ft_head_oracle | 0.728 ± 0.008 | +0.036 ± 0.007 (5/5) | +0.042 ± 0.011 (5/5) |
| ft_head_soft | 0.716 ± 0.009 | +0.025 ± 0.009 (5/5) | +0.030 ± 0.013 (5/5) |
| ft_head_hard | 0.715 ± 0.009 | +0.023 ± 0.009 (5/5) | +0.029 ± 0.013 (5/5) |
| pooled_cont_all | 0.699 ± 0.009 | +0.008 ± 0.003 (5/5) | — |
| ft_all_oracle | 0.744 ± 0.005 | +0.053 ± 0.003 (5/5) | +0.045 ± 0.004 (5/5) |
| ft_all_soft | 0.733 ± 0.004 | +0.041 ± 0.005 (5/5) | +0.034 ± 0.007 (5/5) |
| ft_all_hard | 0.732 ± 0.005 | +0.041 ± 0.006 (5/5) | +0.033 ± 0.008 (5/5) |
| pooled_cont_bn | 0.691 ± 0.005 | -0.000 ± 0.005 (2/5) | — |
| ft_bn_oracle | 0.691 ± 0.007 | -0.000 ± 0.005 (3/5) | -0.000 ± 0.003 (2/5) |
| ft_bn_soft | 0.690 ± 0.006 | -0.001 ± 0.007 (2/5) | -0.001 ± 0.003 (2/5) |
| ft_bn_hard | 0.689 ± 0.005 | -0.003 ± 0.007 (2/5) | -0.002 ± 0.003 (1/5) |

Runtime: 4557.9 s.
