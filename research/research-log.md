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
