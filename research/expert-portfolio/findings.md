# Findings: loop C (expert portfolio)

## Current understanding

**Neither new expert stream earns a place in the submission.** A second CNN (ShallowFBCSPNet)
looked useful on the data the decision was made on, and on the reused hidden runs. On both honest
checks, fresh people (sim2) and a day gap (BNCI 2014-001), it gave no detectable benefit. A frozen
REVE probe with mean pooling added nothing anywhere.

| Evidence | E+T+S − E+T (ShallowFBCSPNet added) | Status of the data |
|---|---|---|
| Dreyer cross-fitted dev bank (decides) | +0.011 (drop-one CI [+0.000, +0.017]); NLL 0.278 → 0.253 | decision data |
| Dreyer R4–R6, seeds 0–2 | +0.014 (3/3 seeds) | **upper bound**: reused by loops A and B |
| **Dreyer sim2, 14 fresh people, seeds 0–2** | **−0.005 [−0.016, +0.004]** | honest |
| **BNCI 2014-001, session 1 → 2, 5 seeds** | **−0.004 (5/5 seeds negative)** | honest, cross-day |

Under the locked rule ShallowFBCSPNet passed: dev (a) with amendment 1, BNCI (b) at −0.004 ≥
−0.005, and no veto. **The user decided not to ship it (2026-10-05)**, because both honest checks
are at or below zero. Its question moves to the identity-integration option B, where a second
trunk costs one shared network instead of 21 copies.

**The same honest check says the shipped C3 (loop B) adds little within a session.** On sim2, E+T
beats EEGNet alone by +0.001 [−0.004, +0.007], against +0.017 on the reused R4–R6. Its cross-day
gain on BNCI (+0.05, 5/5 seeds) still stands. The R4–R6 numbers in `track2/README.md` are upper
bounds.

## Patterns and insights

- **Reused hidden runs mislead in a consistent direction.** Every effect measured on R4–R6 here is
  larger than on fresh people. A fresh simulation (sim2: 14 people drawn from the training pool and
  locked before any training) costs about 1.5 h per seed and gave the honest picture.
- **ShallowFBCSPNet is a near-equal, different model** (dev alone 0.869 vs EEGNet 0.875). It reads
  the early cue-locked window less (early-only 0.749 vs 0.838). On Dreyer's decision data the
  two-CNN ensemble gains, but that gain does not survive fresh people or a day gap.
- **Cross-subject, without calibration, ShallowFBCSPNet is clearly stronger than EEGNet**: 0.844
  vs 0.795 (kit EEGNet) on the leakage-free comparison page (`track2/bench/README.md`). Its value
  is in the pooled model, not as a second per-person expert; option B's ShallowFBCSPNet trunk tests
  that directly.
- **REVE, frozen and mean-pooled, is weak** (0.581 alone on Dreyer). Averaging the embedding over
  channels most likely discards the electrode layout that carries left-vs-right imagery. A
  channel-preserving probe was not tested.

## Engineering outcomes (kept)

- `track2/models.py`: `make_model` (EEGNet or ShallowFBCSPNet, time constants scaled to the
  sampling rate).
- `track2/reve_parts.py` + `submission.ReveProbe`: a frozen REVE stream that loads **offline**
  from the staged cache (`local_files_only=True`; `HF_HUB_OFFLINE` alone is read too early) with
  shipped electrode positions.
- `submission.PortfolioCombiner` + config flags for extra streams. Old packages load unchanged.
- `train_mixture.py --eval-people` (sim2), and `--shallow-experts` / `--reve-probe` /
  `--combiner portfolio` packaging.
- `track2/bench/`: leakage-free benchmark-page solvers.
- 23 `unittest` tests in `track2/tests/`.

## Lessons and constraints

- R4–R6 are report-only upper bounds; honest Dreyer numbers come from sim2.
- Decide on dev, but confirm on data no decision has touched. The identity-integration spec
  (`docs/superpowers/specs/2026-10-05-identity-integration-design.md`) locks 8 holdout people
  before any loop starts.
- `PortfolioLogLinear` needed a class-bias fix for more than 2 classes (BNCI's 4). Every combiner
  should be tested with C ≥ 3.
- Background chains: `A=… && B=… && cmd &` backgrounds the whole list and loses the assignments.

## Open questions

- A REVE probe that keeps per-channel features (mean over time patches only), or REVE's
  pretrained attention pooling.
- ShallowFBCSPNet as a second trunk under option B (one shared network).
