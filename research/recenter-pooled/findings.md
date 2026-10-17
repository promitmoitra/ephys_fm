# Findings: Reference Covariance Re-Centering for Track 2 Task Decoding (Protocols 00 & 01)

- **Branch / Worktree:** `exp/recenter-pooled` ([`.claude/worktrees/t2-recenter`](file:///home/promit/Documents/ephys_fm/.claude/worktrees/t2-recenter))
- **Pre-registered protocols:**
  - **Protocol 00 (§2a — Pooled task classifier on calibration-re-centered tangent space):** [`experiments/00-protocol/protocol.md`](experiments/00-protocol/protocol.md) (locked in `6cd92a4`)
  - **Protocol 01 (§2b follow-up — Test-time reference covariance adaptation for per-person `RiemannExperts`):** [`experiments/01-protocol/protocol.md`](experiments/01-protocol/protocol.md) (locked in `9eeaff7`)
- **Generated results:**
  - Protocol 00: [`results/recenter_results.json`](results/recenter_results.json), [`results/recenter_results.md`](results/recenter_results.md)
  - Protocol 01: [`results/online_recenter_results.json`](results/online_recenter_results.json), [`results/online_recenter_results.md`](results/online_recenter_results.md)
- **Implementation & tests (`8/8` passing):** [`src/recenter.py`](src/recenter.py), [`src/evaluate_recenter.py`](src/evaluate_recenter.py), [`src/online_recenter.py`](src/online_recenter.py), [`src/evaluate_online_recenter.py`](src/evaluate_online_recenter.py), [`tests/test_recenter.py`](tests/test_recenter.py)

---

## 1. Headline Contrast: Protocol 00 vs Protocol 01

| Experiment | What is Re-Centered | Task Classifier | 17-Person Cross-Day Standalone Riemann BA | 17-Person Cross-Day Track 2 Mixture BA (`C3`) | $\Delta$ vs `row 2` (`0.8264`) [95% CI] | `dreyer_sim2` (14) | Passes Track 2 Adopt Rule? |
|---|---|---|---:|---:|---:|---:|---|
| **Baseline (`row 2`)** | Calibration $\bar{C}_k^{\text{cal}}$ (static at test time) | $K$ per-person LRs | 0.7397 | **0.8264** (`row 3` oracle: `0.8334`) | — | 0.9357 | ref |
| **Protocol 00 (§2a)** | Calibration $\bar{C}_k^{\text{cal}}$ (static at test time) | **1 pooled LR** across all $K$ subjects | 0.7060 (`0.7125` oracle) | **0.7885** (`E1`), **0.8003** (`E2`), **0.8176** (`E3`) | -0.0379 [-0.0556, -0.0215] (`E1`) | 0.9036 (`E1`) | **NO** |
| **Protocol 01 (`ema_tangent_shuffled`, FM-10)** | **Test-day online causal EMA** of $\bar{C}_k$ ($\alpha=0.05$, randomly interleaved windows) | **$K$ per-person LRs** (unchanged from `row 2`) | **0.7914** (`+0.0517`) | **0.8381** (`0.8415` w/ `complex_6b`) | **+0.0117** `[+0.0030, +0.0223]` (11W/6L) | **0.9369** | **YES** |
| **Protocol 01 (`batch64_shuffled`, FM-10)** | **Stateless batch-64** tangent shift ($N_0=10$, randomly interleaved windows) | **$K$ per-person LRs** (unchanged from `row 2`) | **0.7746** (`+0.0349`) | **0.8351** (`0.8434` in kit order) | **+0.0087** `[+0.0037, +0.0147]` (13W/3L/1T) | **0.9375** | **YES** |
| **Protocol 01 (`session_fp_spd`)** | **Full test-session** `fp_logp`-weighted SPD Fréchet mean ($N_0=10$) | **$K$ per-person LRs** (unchanged from `row 2`) | **0.7999** (`+0.0602`) | **0.8409** (`0.8452` w/ `complex_6b`) | **+0.0145** `[+0.0053, +0.0259]` (11W/5L/1T) | **0.9381** | **YES** |

---

## 2. Protocol 00 (§2a): Why Calibration Re-Centering with One Pooled Classifier Fails vs `row 2`

1. **H2a.1 (Re-centering vs uncentered pooled): Positive (`+0.0155` routed, `+0.0221` oracle).**
   - Whitening each participant by their calibration Fréchet mean $\bar{C}_k^{-1/2}$ before fitting a single pooled `LogisticRegression(C=0.1)` lifts standalone pooled tangent-space accuracy from **0.6905** (`ts_global_pooled`) to **0.7060** (`ts_recenter_routed`) and **0.7125** (`ts_recenter_oracle`, 95% CI `[+0.0015, +0.0442]`).
   - Single-pass whitener interpolation $\bar{W}(x) = \sum_k p(k \mid x)\bar{C}_k^{-1/2}$ (`ts_recenter_whitener`, **0.7066**) matches the $K$-pass tangent mixture within **+0.0006** (`12/17` exact ties) at $1/K$ the eigendecomposition cost.
2. **H2a.2 (1 pooled LR vs $K$ per-person LRs on the same calibration data): Negative across the 17 cross-day dev people (`-0.0337`).**
   - While pooling across re-centered subjects helps when tangent feature dimension $F$ exceeds per-person calibration count (`dreyer_sim2` $F=378, N_{\text{cal}}\approx 120$: `0.7018 vs 0.6762`, **+0.0256**; `zhou2016` $N_{\text{cal}}\approx 100$: `0.7933 vs 0.7556`, **+0.0378**), it underfits when $N_{\text{cal}} \ge 200$ (`bnci2014`: `0.5903 vs 0.6672`; `bnci2015`: `0.7600 vs 0.7881`).
   - **Geometric cause:** Whitening by $\bar{C}_k^{-1/2}$ only translates each person's resting barycenter to $I_C$; it does not align subject-specific task dipole orientations (rotations $w_k \in T_I\mathcal{S}_{++}^C$).

---

## 3. Protocol 01 (§2b Follow-Up): Why Test-Time Reference Covariance Adaptation for Per-Person Experts Strongly Succeeds

### 3.1 Mechanism
In the shipped `RiemannExperts` (`row 2`), person $k$'s individualized linear classifier $(w_k, b_k)$ captures their task orientation in $T_{\bar{C}_k^{\text{cal}}}\mathcal{S}_{++}^C$, where calibration tangent vectors have zero mean ($\mathbb{E}_{\text{cal}_k}[f_k^{\text{cal}}(C)] = 0$). Across days, cap re-application and electrode impedance shifts move person $k$'s resting barycenter from $\bar{C}_k^{\text{cal}}$ to $\bar{C}_k^{\text{test}}$, inducing a systematic additive shift $\mu_k^{\text{test}} = \mathbb{E}_{\text{test}_k}[f_k^{\text{cal}}(C)] \neq 0$ in tangent space that biases the linear readout $w_k^\top f_k + b_k$.

By tracking $\hat{\mu}_k$ (or $\bar{C}_k^{(t)}$ on $\mathcal{S}_{++}^C$) at test time using the fingerprint responsibility weights $\gamma_{tk} = p(k \mid x_t)$ — **without touching $(w_k, b_k)$ and without using any task labels** — we re-center each person's test-day tangent vectors back to zero mean before applying their calibrated classifier $(w_k, b_k)$.

### 3.2 Key Results (`results/online_recenter_results.md`)
1. **Large standalone cross-day gain on `RiemannExperts` (Level 1):**
   - Static calibration reference (`static_cal_ref`): **0.7397** (`bnci2014` `0.6672`, `bnci2015` `0.7881`, `zhou2016` `0.7556`).
   - Causal online tangent EMA under **FM-10 random window shuffling** (`ema_tangent_shuffled`, $\alpha=0.05$): **0.7914** (**+0.0517** `[+0.0193, +0.0911]`, 13 better / 4 worse / 0 tied).
     - `bnci2014`: `0.6672` $\to$ **0.6904** (`+0.0232`)
     - `bnci2015`: `0.7881` $\to$ **0.8431** (`+0.0550`; Subject 5 jumps from **0.5500 $\to$ 0.8350**)
     - `zhou2016`: `0.7556` $\to$ **0.8556** (`+0.1000`)
   - Full-session fingerprint-weighted SPD Fréchet mean (`session_fp_spd`): **0.7999** (**+0.0602** `[+0.0250, +0.1052]`, 13W / 3L / 1T), and **0.8110** with `complex_6b` (`+0.0713` `[+0.0328, +0.1176]`), within `0.0064` of the true-ID oracle session ceiling (`0.8174`).
2. **Surpasses both `row 2` (`0.8264`) and the static-reference Oracle Ceiling `row 3` (`0.8334`) in the end-to-end Track 2 C3 mixture (Level 2, seeds 0–2):**
   - **Causal online tangent EMA (`ema_tangent_shuffled`, FM-10 shuffled):** **0.8381** (**+0.0117** `[+0.0030, +0.0223]` vs `row 2`, 11W / 6L; `dreyer_sim2`: **0.9369**, `+0.0012`).
   - **Causal online tangent EMA in dataset order (`ema_tangent_ordered`):** **0.8420** (**+0.0156** `[+0.0055, +0.0273]` vs `row 2`, 12W / 4L / 1T; `dreyer_sim2`: **0.9393**, `+0.0036`).
   - **Stateless batch-64 adaptation (`batch64_ordered` / `batch64_shuffled`, zero state across `predict(X)` calls):**
     - Kit batching (`batch64_ordered`): **0.8434** (**+0.0170** `[+0.0078, +0.0277]` vs `row 2`, 15W / 2L).
     - **FM-10 randomly mixed-person batches (`batch64_shuffled`):** **0.8351** (**+0.0087** `[+0.0037, +0.0147]` vs `row 2`, 13W / 3L / 1T; `dreyer_sim2`: **0.9375**).
   - **Combined with `complex_6b` (`complex6b_ema_tangent_shuffled` / `complex6b_session_fp_spd`):**
     - `complex6b_ema_tangent_shuffled`: **0.8415** (**+0.0151** `[+0.0042, +0.0280]` vs `row 2`).
     - `complex6b_session_fp_spd`: **0.8452** (**+0.0188** `[+0.0067, +0.0332]` vs `row 2`).
3. **Zero extra eigendecompositions required for tangent-space shift subtraction:**
   - Notice that `ema_tangent_shuffled` (**0.8381**) and `session_fp_tangent` (**0.8400**) match their exact SPD-manifold counterparts (`ema_spd_shuffled` **0.8376**, `session_fp_spd` **0.8409**) within `0.0005–0.0009` BA while requiring **zero additional matrix eigendecompositions** (`eigh`) at test time — they simply subtract the fingerprint-weighted running/batch mean $\hat{\mu}_k$ from `f = tangent(C[:, None], self.cref_isqrt[None])` inside `RiemannExperts.forward`!
