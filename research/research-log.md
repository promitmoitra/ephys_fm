# Research Log (loop A: fingerprint)

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-09-30 | bootstrap | Worktree `t2-fingerprint` on `exp/fingerprint-model`. Read the handoff, `track2/README.md`, `train_mixture.py`, the BNCI fingerprint results (EEGNet 0.909 / Riemann 0.872 across days, Riemann 0.999 within a session). Key observation: Dreyer has one session per person, so R1–R2 → R3 and R1–R3 → R4–R6 are **within-session, cross-run** problems, where BNCI's Riemannian fingerprint was near-perfect. Wrote the shared harness `research/src/fp.py` (locked metrics: bal acc, NLL, p_true, ECE, top-3, worst person) and split caches under `outputs/t2-fingerprint/cache/`. Protocols locked for H0 (EEGNet baseline, 3 seeds) and H3 (Riemannian, 4 configs). |
