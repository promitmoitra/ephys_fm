# Checkpoint 1: confirm protocol 01's candidates on R4–R6

Locked 2026-09-30 after protocol 01's dev results, before looking at any test-bank combiner
output. Candidates follow protocol 01's locked rule:

| ID | Combiner | Why |
|---|---|---|
| R0 | EEGNet only | baseline (known: 0.901 soft / 0.908 oracle) |
| R1 | equal average eegnet+ts | reference (known: 0.909 soft / 0.921 oracle) |
| H5a | per-expert temperature, then equal average | best dev bal acc among global combiners (0.888) |
| H3 | log-linear + bias (stacking on log-probs) | best dev NLL among global combiners (0.300) |

Each learned combiner is fitted once on the full dev bank (all 840 R3 windows, true ID) and
applied unchanged to the test bank. Reported: balanced accuracy and NLL on R4–R6 under
oracle ID and soft-routed with the packaged EEGNet fingerprint (`fp`); per-person paired
comparison vs R0 and R1. If loop A's filter-bank fingerprint probabilities on R4–R6 are
available (`outputs/t2-fingerprint/confirm-1/ts_fb_C1_seed0_probs.npz`, read-only), also
soft-routed with it.

**Dev expectation (from protocol 01):** H5a and H3 ≈ R0 in accuracy (dev: no combiner beats
EEGNet alone); R1 below R0 on dev but above it on the one known test seed. Whatever the test
shows, it does not change the next direction on its own: the dev bank has 840 windows and
the test bank 2,520, with one seed each.
