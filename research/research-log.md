# Research Log (loop A: fingerprint)

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-09-30 | bootstrap | Worktree `t2-fingerprint` on `exp/fingerprint-model`. Read the handoff, `track2/README.md`, `train_mixture.py`, the BNCI fingerprint results (EEGNet 0.909 / Riemann 0.872 across days, Riemann 0.999 within a session). Key observation: Dreyer has one session per person, so R1–R2 → R3 and R1–R3 → R4–R6 are **within-session, cross-run** problems, where BNCI's Riemannian fingerprint was near-perfect. Wrote the shared harness `research/src/fp.py` (locked metrics: bal acc, NLL, p_true, ECE, top-3, worst person) and split caches under `outputs/t2-fingerprint/cache/`. Protocols locked for H0 (EEGNet baseline, 3 seeds) and H3 (Riemannian, 4 configs). |
| 2 | 2026-09-30 | inner-loop | H3 (R1–R2 → R3): `ts_broad_C1` 0.987 / NLL 0.107; **`ts_fb_C1` 1.000 / NLL 0.027**; C = 0.1 worse on NLL for both. Saturates the locked dev metric. |
| 3 | 2026-09-30 | inner-loop | H3.1 drift benchmarks (Riemann arms): Dreyer R1 → R2 0.996, R1 → R3 1.000 for `ts_fb` (no decay with run distance). **BNCI cross-day: `ts_fb` 0.974 / NLL 0.071 vs `ts_broad` 0.872.** Prediction 2 (Riemann below EEGNet across days) refuted: the filter bank fixes the cross-day weakness of the broadband covariance. EEGNet arms queued. |
| 4 | 2026-09-30 | inner-loop | H0 EEGNet baseline running (slow under contention); its R3 learning curve is unstable at fixed budget (0.70 @ 50, 0.59 @ 75, 0.72 @ 100 epochs), so the shipped best-epoch-on-R3 score hid high variance. |
| 5 | 2026-09-30 | inner-loop | **confirm-1** (pre-registered): on R4–R6, `ts_fb_C1` fingerprint 0.996 / NLL 0.044 (shipped EEGNet 0.744); soft mixture with the fixed experts 0.907 vs oracle 0.9075 (shipped 0.901). `ts_broad_C1` 0.990, mixture 0.908. Predictions 1–2 supported. |
| 6 | 2026-09-30 | outer-loop | Cycle 1. Direction: **DEEPEN + BROADEN**. On Dreyer the routing gap is closed (soft = oracle), so no fingerprint can lift the Dreyer mixture further. What remains for the sealed phase: (a) shippability (pure torch), (b) cross-day robustness (BNCI worst person 0.77), (c) understanding which bands carry stable identity (H3.2). EEGNet arms kept as pre-registered comparisons, lower priority. |
| 7 | 2026-09-30 | inner-loop | H9 torch export: filtfilt on a fixed window = one T × T matrix per band; OAS, tangent space and folded LR in torch. Matches sklearn to 3.9e-6 (float32), argmax identical on 840 windows, 5.7 MB state. Pass. |
