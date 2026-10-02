# H6: EEGNet + Riemannian fingerprint ensembles

**Status:** CONFIRMATORY. Depends on the saved per-window probabilities of H3.1.

## Mechanism → prediction

The two model families see different evidence: tangent-space LR sees only second-order spatial
statistics (per band), while EEGNet also sees waveform shape (the cue-locked response, eye
position). Their errors should be partly independent, and their failure regimes differ: Riemann
is near-perfect within a session, EEGNet held up better across days on BNCI.
**Prediction:** on D2 (BNCI cross-day) an equal-weight ensemble beats both members on NLL, and
matches or beats the better member on accuracy. On D1 (Dreyer run distance) it stays within
0.01 of `ts_fb` in accuracy.

## What (no fitting; combinations fixed in advance)

For each EEGNet seed s ∈ {0, 1, 2} and each benchmark D1_R1toR2, D1_R1toR3, D2_bnci_s1tos2:

| Arm | Combination |
|---|---|
| `lin` | (P_eegnet + P_ts_fb) / 2 |
| `geo` | normalize(sqrt(P_eegnet · P_ts_fb)) (product of experts, equal weights) |

Plus `lin3` / `geo3`, which also include `ts_broad`, as a check that adding a correlated member
does not help.

No weights are tuned (tuning on the test runs of D1/D2 would bias them). If equal weighting
shows promise, weights get learned out-of-run in a follow-up.

## Metrics / decision

The locked metrics, 3-seed means. Supported if `geo` or `lin` beats both members on D2 NLL
without losing more than 0.01 accuracy on D1_R1toR3 relative to `ts_fb`.
