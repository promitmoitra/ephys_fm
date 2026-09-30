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
