# confirm-1: the filter-bank Riemannian fingerprint on the hidden runs

**Status:** CONFIRMATION (outer-loop checkpoint 1). Pre-registered before any R4–R6 contact.

## Candidates (fixed; no other variant is scored on R4–R6 in this checkpoint)

| Candidate | Recipe | Seeds | Role |
|---|---|---|---|
| `ts_fb_C1` | 6-band (1–4, 4–8, 8–13, 13–20, 20–30, 30–45 Hz) OAS covariance → tangent space → StandardScaler → LR C=1 | deterministic | lead |
| `ts_broad_C1` | broadband version | deterministic | reference |
| `eegnet` | shipped recipe, 150 epochs, **last** epoch (no selection) | 0, 1, 2 | baseline |

Each is refit on R1–R3 of the 21 evaluation people (2,520 windows) and scored on R4–R6 (2,520),
via `research/src/confirm.py`. Mixture: soft routing over the **fixed** EEGNet experts in
`outputs/track2_dreyer_sim/submission/mixture.pt`, with oracle, hard and uniform references.

## Dev evidence behind the choice

`ts_fb_C1`: R3 1.000 (NLL 0.027); R1 → R3 1.000; BNCI cross-day 0.974 (NLL 0.071), vs 0.872
for `ts_broad_C1`.

## Predictions

1. `ts_fb_C1` fingerprint balanced accuracy ≥ 0.97 on R4–R6, vs 0.744 for the shipped EEGNet
   fingerprint.
2. The soft mixture with `ts_fb_C1` reaches ≥ 0.905, within 0.003 of the oracle (0.908 seed 0),
   vs 0.901 shipped.
3. The EEGNet baseline recipe (last epoch) lands near the shipped fingerprint (0.70–0.80).

## Interpretation rule

Only predictions 1–2 are tested. The headroom on Dreyer is small (oracle − shipped = 0.007),
so a mixture gain within ±0.003 is not evidence either way. The fingerprint accuracy is the
primary confirmation.
