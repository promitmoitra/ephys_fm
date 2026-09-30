# Research log: loop B (learned expert weights)

## 2026-09-30

- **Bootstrap.** Read the handoff brief (`docs/handoff/track2-research-loops.md`) and the
  Track 2 code. Workspace under `research/`. 20-min heartbeat set.
- Wrote `research/src/bank.py`; 1-epoch smoke test passed end to end (6 min).
- Locked protocol 00 (banks and evaluation). Deviation: the test bank reuses the packaged
  seed-0 EEGNet experts so the confirmation baseline is exactly 0.901.
- 11:23 launched the dev bank (pooled on pool + R1–R2, 4 threads). Loop A is running
  three jobs at the same time, so expect it to be slow.
