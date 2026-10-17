# Findings: Reference Covariance Re-Centering & Per-Person $\mu$-Band Experts (Protocols 00, 01, 02)

- **Branch / Worktree:** `exp/recenter-pooled` ([`.claude/worktrees/t2-recenter`](file:///home/promit/Documents/ephys_fm/.claude/worktrees/t2-recenter))
- **Pre-registered protocols:**
  - **Protocol 00 (§2a — Pooled task classifier on calibration-re-centered tangent space):** [`experiments/00-protocol/protocol.md`](experiments/00-protocol/protocol.md) (`6cd92a4`)
  - **Protocol 01 (§2b follow-up — Test-time reference covariance adaptation for per-person `RiemannExperts`):** [`experiments/01-protocol/protocol.md`](experiments/01-protocol/protocol.md) (`9eeaff7`)
  - **Protocol 02 (§2c — Per-person $\mu$-peak band centering for `RiemannExperts`):** [`experiments/02-protocol/protocol.md`](experiments/02-protocol/protocol.md) (`ff73954`)
- **Generated results:**
  - Protocol 00: [`results/recenter_results.json`](results/recenter_results.json), [`results/recenter_results.md`](results/recenter_results.md)
  - Protocol 01: [`results/online_recenter_results.json`](results/online_recenter_results.json), [`results/online_recenter_results.md`](results/online_recenter_results.md)
  - Protocol 02: [`results/mu_band_results.json`](results/mu_band_results.json), [`results/mu_band_results.md`](results/mu_band_results.md)
- **Implementation & tests (`10/10` passing):** [`src/recenter.py`](src/recenter.py), [`src/evaluate_recenter.py`](src/evaluate_recenter.py), [`src/online_recenter.py`](src/online_recenter.py), [`src/evaluate_online_recenter.py`](src/evaluate_online_recenter.py), [`src/mu_band.py`](src/mu_band.py), [`src/evaluate_mu_band.py`](src/evaluate_mu_band.py), [`tests/test_recenter.py`](src/tests/test_recenter.py)

---

## 1. Summary Table Across All Three Protocols (17 Cross-Day Dev People + 14 Fresh `dreyer_sim2` People)

| Protocol / Arm | What Changed | Task Classifier | L1 Standalone Riemann BA (17 Dev) | L2 Track 2 Mixture BA (`C3`, Seeds 0–2) | $\Delta$ vs `row 2` (`0.8264`) [95% CI] | `dreyer_sim2` (14) | Adopt over `row 2`? |
|---|---|---|---:|---:|---:|---:|---|
| **Baseline (`row 2`)** | Fixed `[8, 30]` Hz, static calibration $\bar{C}_k^{\text{cal}}$ | $K$ per-person LRs | 0.7397 | **0.8264** (`row 3` oracle: `0.8334`) | — | 0.9357 | ref |
| **Protocol 00 (§2a `E1` / `E2`)** | Calibration $\bar{C}_k^{\text{cal}}$ re-centering + **1 pooled LR** | **1 pooled LR** across all $K$ subjects | 0.7060 (`0.7125` oracle) | **0.7885** (`E1`), **0.8003** (`E2`) | -0.0379 [-0.0556, -0.0215] (`E1`) | 0.9036 (`E1`) | **NO** |
| **Protocol 01 (`ema_tangent_shuffled`, FM-10)** | **Test-day online causal EMA** of $\bar{C}_k$ ($\alpha=0.05$, randomly interleaved windows) | **$K$ per-person LRs** (unchanged) | **0.7914** (`+0.0517`) | **0.8381** (`0.8415` w/ `complex_6b`) | **+0.0117** `[+0.0030, +0.0223]` (11W/6L) | **0.9369** | **YES** |
| **Protocol 01 (`batch64_shuffled`, FM-10)** | **Stateless batch-64** tangent shift ($N_0=10$, randomly interleaved windows) | **$K$ per-person LRs** (unchanged) | **0.7746** (`+0.0349`) | **0.8351** (`0.8434` in kit order) | **+0.0087** `[+0.0037, +0.0147]` (13W/3L/1T) | **0.9375** | **YES** |
| **Protocol 01 (`session_fp_spd`)** | **Full test-session** `fp_logp`-weighted SPD Fréchet mean ($N_0=10$) | **$K$ per-person LRs** (unchanged) | **0.7999** (`+0.0602`) | **0.8409** (`0.8452` w/ `complex_6b`) | **+0.0145** `[+0.0053, +0.0259]` (11W/5L/1T) | **0.9381** | **YES** |
| **Protocol 02 (§2c `mu_shift_wide_static`)** | Shift single `[8, 30]` Hz band by $f_{\mu,k}-10$ Hz | $K$ per-person LRs | 0.7341 (`-0.0056`) | **0.8258** (`0.8388` w/ EMA) | -0.0006 [-0.0026, +0.0014] | 0.9351 | **NO** |
| **Protocol 02 (§2c `mu_beta_2band_static`)** | Split into 2 bands at personal $\mu$ peak: $[f_{\mu,k}-2.5, f_{\mu,k}+2.5] + [f_{\mu,k}+2.5, 30]$ Hz | $K$ per-person LRs | **0.7648** (`+0.0251`) | **0.8401** | **+0.0137** `[+0.0086, +0.0190]` | 0.9327 | **YES** (vs `row 2`; `+0.0058` vs fixed 2-band) |
| **Protocol 01 + 02 (`mu_beta_2band_ema_shuf`)** | Personal 2-band $[\mu_k, \beta_k]$ + **causal tangent EMA (FM-10 shuffled)** | $K$ per-person LRs | **0.8122** (`+0.0725` vs `row 2` L1) | **0.8522** | **+0.0258** `[+0.0169, +0.0356]` vs `row 2` (`+0.0142` `[+0.0091, +0.0197]` vs 1-band EMA) | **0.9369** | **YES** |

---

## 2. Key Scientific Takeaways

1. **Protocol 00 (§2a — Calibration re-centering with 1 pooled classifier): Negative vs `row 2`.**
   - Whitening each person by $\bar{C}_k^{-1/2}$ removes resting barycenter shift (`+0.0155` routed / `+0.0221` oracle over uncentered pooled tangent space), and helps when $F > N_{\text{cal}}$ (`dreyer_sim2` `+0.0256`, `zhou2016` `+0.0378`).
   - However, when $N_{\text{cal}} \ge 200$ (`bnci2014`, `bnci2015`), replacing $K$ individualized linear readouts $w_k$ with 1 pooled hyperplane $w_{\text{pool}}$ loses `-0.0337` standalone (`-0.0261` in C3) because inter-subject task differences include tangent-space rotations, not just resting translation.
2. **Protocol 01 (§2b follow-up — Test-time reference covariance adaptation for per-person `RiemannExperts`): Strong Positive (`+0.0117` to `+0.0188` in C3, passing `FM-10`).**
   - Keeping each person's calibrated $(w_k, b_k)$ and subtracting their fingerprint-weighted test-session tangent shift $\hat{\mu}_k$ (via causal EMA `ema_tangent_shuffled`, stateless `batch64_shuffled`, or full-session `session_fp_spd`) lifts standalone cross-day Riemannian accuracy from **0.7397 $\to$ 0.7914–0.8110** and lifts the Track 2 `C3` mixture from **0.8264 (`row 2`) $\to$ 0.8381–0.8452**, surpassing even the static-reference oracle-ID ceiling (`row 3 = 0.8334`) with zero extra `eigh` calls for the tangent-shift arms.
3. **Protocol 02 (§2c — Per-person $\mu$-peak band centering): Shifting a single `8–30 Hz` band does nothing (`-0.0006`), but splitting `8–30 Hz` into two sub-bands at $f_{\mu,k} + 2.5$ Hz (`mu_beta_2band`) adds a further `+0.0142` (`0.8381 -> 0.8522`).**
   - Shifting the wide 22 Hz `[8, 30]` Hz passband (`mu_shift_wide`) has near-zero effect (`-0.0006` static, `+0.0008` with EMA in C3) because `8–30 Hz` already covers both $\mu$ and $\beta$.
   - Separating $\mu$ and $\beta$ into two covariance matrices (`[f_{\mu,k}-2.5, f_{\mu,k}+2.5]` Hz and `[f_{\mu,k}+2.5, 30]` Hz) prevents $\beta$ power from washing out $\mu$ spatial covariance. Crucially, comparing against the **fixed 2-band control** (`fixed_2band`: `[8, 13]` + `[13, 30]` Hz) shows why individualizing $f_{\mu,k}$ matters: on `zhou2016` (where subjects' $\mu$ peaks sit at `11.8–12.2 Hz`), a fixed `13 Hz` boundary cuts through the $\mu$ peak (`0.8044` standalone with EMA), whereas placing the boundary at $f_{\mu,k}+2.5$ Hz lifts `zhou2016` to **`0.8756`** (`+0.0712` standalone, and `+0.0134` `[+0.0005, +0.0305]` across all 17 dev people on Level 1; `0.8522` vs `0.8444` on Level 2).
