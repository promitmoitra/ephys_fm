# Research log: loop B (learned expert weights)

## 2026-09-30

- **Bootstrap.** Read the handoff brief (`docs/handoff/track2-research-loops.md`) and the
  Track 2 code. Workspace under `research/`. 20-min heartbeat set.
- Wrote `research/src/bank.py`; 1-epoch smoke test passed end to end (6 min).
- Locked protocol 00 (banks and evaluation). Deviation: the test bank reuses the packaged
  seed-0 EEGNet experts so the confirmation baseline is exactly 0.901.
- 11:23 launched the dev bank (pooled on pool + R1–R2, 4 threads). Loop A is running
  three jobs at the same time, so expect it to be slow.
- Fixed a typo in protocol 00: R4–R6 has 2,520 windows (21 × 120), not 5,040.
- 11:31 test bank built; sanity check reproduces the references exactly (oracle eegnet 0.908, ts 0.688, csp 0.671; soft eegnet 0.901, eegnet+ts 0.909). Sealed until checkpoint 1.
- 12:20 dev bank done (sanity: pooled 0.860, control 0.888, eegnet 0.887, ts 0.681, csp 0.645 on R3).
  [Session paused on usage limit until 15:30.]
- Protocol 01 results: on dev, the plain EEGNet+ts average hurts (0.869 vs 0.887); learned
  combiners switch ts off. ts is over-confident (conf 0.754 vs acc 0.681), EEGNet calibrated.
  ts is person-specific (≥0.92 for 6 people, chance for 9).
- Loop A (read-only): filter-bank covariance fingerprint 0.996 on R4–R6; soft mixture at the
  oracle ceiling. Routing is solved; loop B should target oracle-ID expert quality.
- Checkpoint 1 (R1, H5a, H3 pre-registered): on R4–R6 ts helps: R1 0.921, H3 0.921 with the
  best NLL (0.214) vs R0 0.908. Dev and test disagree in sign → dev bank too small.
- **Outer loop 1: DEEPEN the evaluation.** Protocol 02: cross-fitted dev bank (each calibration
  run held out; 2,520 windows). 15:35 launched folds R1, R2 (2 threads each).
- 16:54 cross-fit folds done. [Session paused on usage limit until 20:30.]
- 20:31 xfit bank merged (2,520 windows). Protocol 02: plain average −0.008 pooled (per fold
  −0.015 / +0.010 / −0.018); log-linear +0.002, NLL 0.303 → 0.287; per-person linear +0.007.
  Bootstrap CIs ±0.011: nothing resolved in accuracy; the NLL gain is consistent.
- Protocol 03: bands/windows don't make a better classical expert; C = 0.1 calibrates ts and
  flips the plain average to +0.007. Reliability predicts held-out ts accuracy (r ≈ 0.99);
  H8 has the best NLL (0.269) and, on ts_C0.1, the best accuracy (0.883).
- Checkpoint 2 (C1–C4, adoption rule fixed in advance): all beat EEGNet alone on R4–R6 by
  +0.012–0.014 (CI > 0 for log-linear), NLL 0.209–0.213 vs 0.235. **Adopted C3.**
- **Outer loop 2: DEEPEN (robustness).** Protocol 04: seeds 1, 2 of the test-time EEGNet
  experts; 20:35 launched.
- 20:45 Experiment 05 (engineering): torch export of ts_C0.1 experts + C3 matches the
  research pipeline (|Δp| 2e-8, oracle 0.921); run the ts path in float64.
- 20:55 Protocol 06 locked: zero-shot transfer of the Dreyer coefficients to BNCI 2014-001
  (4-class, cross-session, 5 seeds). Classical stage done: ts 0.665, ts_C0.1 0.654 on
  session 2; reliability spread 0.31–0.77 (4-class scale). EEGNet stage running.
- 20:35–01:30 Seeds 1, 2 test banks; BNCI experts, 5 seeds. [Session paused on usage limit.]

## 2026-10-01

- Protocol 04: C3 − EEGNet = +0.014 / +0.020 / +0.024 (seeds 0 / 1 / 2), all CIs > 0; mean
  0.898 → 0.917. Protocol 06: Dreyer coefficients on BNCI +0.052 (C3) / +0.054 (C1), 5/5 seeds;
  plain average +0.062; BNCI-fitted (exploratory) +0.070. Across days the EEGNet expert is
  over-confident, so the best weights reverse.
- **Outer loop 3: CONCLUDE.** Updated `track2/README.md` (step 6, evidence, next steps) and the
  README experiments table; final report 002; PR.
