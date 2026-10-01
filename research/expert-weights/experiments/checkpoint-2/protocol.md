# Checkpoint 2: confirm protocols 02–03's candidates on R4–R6

Locked 2026-09-30, after the cross-fitted dev results and before any test-bank output for
these candidates. Each combiner is fitted once on the full cross-fitted dev bank (2,520
windows, true ID) and applied unchanged to the test bank (packaged EEGNet experts trained on
R1–R3; classical experts fitted on R1–R3; reliability from the 6 ordered run pairs of R1–R3).

| ID | Combiner | Classical expert | Rule that selected it |
|---|---|---|---|
| C1 | H4 log-linear | ts | protocol 02 (best accuracy after the tie-break, and best NLL) |
| C2 | H3 log-linear + bias | ts_C0.1 | protocol 03 H9 rule |
| C3 | H8 reliability-weighted log-linear | ts_C0.1 | H8 on the H9 winner (best dev accuracy, 0.883) |
| C4 | H8 reliability-weighted log-linear | ts | best dev NLL (0.269) |
| refs | R0 EEGNet only; R1 equal average (ts); H3 (ts, refit on the xfit bank) | | |

Reported: oracle-ID and soft-routed (packaged fingerprint; loop A's `ts_fb` fingerprint)
balanced accuracy and NLL; per-person paired comparison and a person-bootstrap 95% CI vs R0.

**Expectation from dev:** all candidates within ±0.01 of each other; gains over R0 of
roughly +0.005 to +0.013 (checkpoint 1: R1 and H3 +0.013); C3/C4 with the lowest NLL.

**What gets adopted.** The test set cannot choose among candidates whose dev differences are
unresolved. So the adoption decision is made now, on dev: if every candidate is ≥ R0 on
the test set (no sign flip), the recommended combiner is **C3** (H8 on ts_C0.1). It has the
best dev accuracy, a mechanism (reliability is a stable person trait) and a calibrated
expert. If any candidate falls below R0 on the test set, the recommendation is to ship
EEGNet alone plus the NLL-only improvement (none) and to report the classical expert as
unresolved.
