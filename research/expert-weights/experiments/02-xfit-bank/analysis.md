# Protocol 02 analysis: cross-fitted dev bank (CONFIRMATORY)

Results: `results/sweep.md`. 2,520 windows; each calibration run held out in turn.

**Sanity:** EEGNet experts per fold 0.883 / 0.855 / 0.887 (R1 / R2 / R3 held out); pooled
0.857 / 0.830 / 0.860; the R2 fold is the hardest for every EEGNet model. ts 0.665 / 0.694 /
0.681.

| Prediction | Outcome |
|---|---|
| R1 − R0 negative on the R3 fold, ~0 or positive on R1/R2 folds; pooled between −0.018 and +0.014 | Folds: −0.015 / **+0.010** / −0.018; pooled **−0.008** (9 better / 11 worse). Partly: only the R2 fold (where EEGNet is weakest) gains. R3 is not special; R1 behaves like R3. |
| H3 ≥ R0 and lowest NLL of global combiners | H3 +0.001 (12 / 7), NLL 0.288 vs 0.303. Log-linear without bias (H4) ties: +0.002, NLL 0.287 (lowest). |
| Per-person weights (H2) no longer worse than global | **Supported:** H2 is the most accurate rule, +0.007 (5 better / 3 worse; most people tie), NLL 0.288. |

**Resolution.** Person-level bootstrap 95% CIs on the gain over EEGNet alone are about
±0.011 wide: R1 −0.008 [−0.028, +0.011], H2 +0.007 [−0.004, +0.019], H4 +0.002
[−0.009, +0.012]. Even with 2,520 windows, no rule's accuracy gain is resolved. Only the
NLL gain of log-linear pooling is consistent: R3 bank, cross-fitted bank and R4–R6 all show it.

**What this says about the dev/test disagreement:** the plain average's sign depends on the
fold (−0.018 to +0.010) and on the test set (+0.014). This is fold-to-fold variation of a
small effect: ts helps when the EEGNet expert is weak on that run (R2 fold) and hurts when
it is strong. The pooled dev estimate (−0.008) and the test estimate (+0.014) are both
inside that spread.

**Checkpoint 2 candidates by the locked rule:** best global accuracy H7b (0.878) vs H4 (0.877):
within 0.002, so the one with fewer parameters, **H4**; best NLL also **H4**.
