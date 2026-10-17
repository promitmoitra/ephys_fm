# Findings: 2 × 2 Complex Hermitian × 5/6-Band Fingerprint on the Cross-Day Harness

**Branch:** `exp/complex-fingerprint` · **Worktree:** `.claude/worktrees/t2-fp-complex`  
**Protocol:** [`experiments/00-protocol/protocol.md`](experiments/00-protocol/protocol.md) (committed in `0662fca` before any code or evaluation)  
**Results:** [`results/2x2_results.json`](results/2x2_results.json), [`results/2x2_results.md`](results/2x2_results.md)

---

## 1. Summary

1. **§1a (Complex Hermitian tangent space) replicates on the Track 2 cross-day harness (`complex_6b`):**
   - Across the 17 cross-day dev participants (`bnci2014` 6, `bnci2015` 8, `zhou2016` 3), `complex_6b` lifts person-ID balanced accuracy from **0.9329 to 0.9859** (**+0.0529**, 95% CI `[+0.0000, +0.1503]`, **3 better / 0 worse / 14 tied at 1.000**) and cuts person-ID NLL from **0.1720 to 0.0642 nats** (**−0.1078 nats**, 95% CI **`[+0.0026, +0.2682]`**, **13 better / 4 worse**, Clopper-Pearson win CI `[50.1%, 93.2%]`).
   - Specifically, every dev participant who was below 0.995 with `real_6b` improved:
     - `bnci2015` dev subject 5: **0.635 → 0.770** (NLL `1.278 → 0.670`); dev subject 7: **0.995 → 1.000**.
     - `zhou2016` dev subject 2: **0.240 → 1.000** (NLL `1.242 → 0.135`), lifting `zhou2016` cross-day ID from **0.7467 → 1.0000**.
   - As on PainLab and ds004148, the gain decomposes into both the $\rho = 0.05$ Hermitian trace-shrinkage estimator (`real_6b` 0.9329 → `herm_real_6b` 0.9676, +0.0347) and the imaginary phase-lag subspace (`herm_real_6b` 0.9676 → `complex_6b` 0.9859, **+0.0182**, 3 better / 0 worse).

2. **§1b (Dropping the 1–4 Hz delta band) does not help cue-locked motor-imagery windows (`real_5b`, `complex_5b`):**
   - Unlike resting-state recordings in PainLab, Track 2's windows are 0–4 s cue-locked motor-imagery trials where 1–4 Hz carries task/cue-locked spatial structure.
   - Dropping delta alone (`real_5b` vs `real_6b`) changes cross-day ID accuracy by only **+0.0015** while slightly worsening mean NLL (**0.1720 → 0.1770**), and hurts single-day calibration on `zhou2016` (`s0 → s2`: `0.9733 → 0.9311`; `s1 → s2`: `0.7933 → 0.7800`).
   - On top of the complex template, `complex_6b` beats `complex_5b` on cross-day ID accuracy (**0.9859 vs 0.9835**), NLL (**0.0642 vs 0.0714**), task accuracy (**0.8278 vs 0.8266**), and both single-day `zhou2016` slices (`0.9689 vs 0.9511` and `0.8978 vs 0.8844`).

3. **End-to-end Track 2 task decoding is ceiling-bounded by the +0.0070 oracle headroom:**
   - Because 14 of the 17 dev participants already had $\ge 0.99$ ID accuracy under `real_6b` (and on `zhou2016` subject 2, the other participants' experts decoded subject 2's motor imagery nearly as well as subject 2's own expert), oracle identity (`row 3`) only exceeds `real_6b` (`row 2`) by **+0.0070** (`[+0.0000, +0.0206]`).
   - Routing the banked C3 experts with `complex_6b` yields **0.8278** vs `real_6b`'s **0.8264** (**+0.0014**, 95% CI `[-0.0003, +0.0044]`, 4 better / 2 worse / 11 equal), recovering 20% of the oracle gap (including **+0.0233** task accuracy on `bnci2015` dev subject 5). On within-session `dreyer_sim2` (14 fresh people), task accuracy is unchanged (**0.9357**, $\Delta = 0.0000$).
   - Consequently:
     - **Criterion A (standalone Track 2 candidate rule `metrics.adopt`, requiring $\ge +0.005$ task gain):** **FAIL** (`+0.0014 < +0.005`).
     - **Criterion B (fingerprint replacement rule, Protocol 00 §5.2):** **PASS** for **`complex_6b`** (significant cross-day ID NLL reduction of −0.1078 nats with 95% CI excluding zero, non-negative task delta on dev and `sim2`, and pure-PyTorch `complex64` export parity of $\max |\Delta p| = 3.59 \times 10^{-6} < 10^{-4}$).

4. **Pure-PyTorch export (`ComplexFBFingerprint`):**
   - Folding the 4th-order zero-phase Butterworth band-pass filter and the discrete Hilbert transform into one `complex64` $(T \times T)$ matrix per band matches the SciPy + sklearn reference to **$\max |\Delta p| \le 3.59 \times 10^{-6}$** across all 4 datasets with **100.0% top-1 agreement**.
   - State-dict size is ~11.1 MB at 120 Hz ($T = 480$) and projects to ~192.5 MB at 500 Hz ($T = 2000$, 43 channels; well within the 15 GB storage limit, or ~11 MB if resampled to 120 Hz before filtering).
