# Protocol 00: 2 × 2 Complex Hermitian × 5/6-Band Fingerprint on the Cross-Day Harness

Locked 2026-10-17 on branch `exp/complex-fingerprint` (worktree `.claude/worktrees/t2-fp-complex`)
before any experiment code or evaluation numbers exist on the Track 2 cross-day harness.

**Sources:**
- `docs/handoff/neuralprint-handoff.md` (§1a Complex Hermitian template, §1b Drop the delta band,
  and Methodological Rules FM-1 through FM-11).
- `docs/superpowers/specs/2026-10-05-identity-integration-design.md` (§1 Shared evaluation).
- `docs/handoff/identity-integration-loops.md` (Phase 0 harness banks and reference rows).

---

## 1. Question and Mechanism

The shipped Track 2 fingerprint (`submission.FBFingerprint`, from loop A) extracts per-band real OAS
covariances across 6 bands (1–45 Hz), projects each band into its Riemannian tangent space at the
calibration Fréchet mean, and fits a standardised multinomial logistic regression ($C = 1.0$).

Two results from NeuralPrint (`a87b47f`) bear directly on cross-day fingerprint accuracy:
1. **§1a — Complex Hermitian tangent space (`H_COMPLEX_TANGENT`, `H_COMPLEX_REP`):** Computing each
   band's covariance from the Hilbert analytic signal $Z_b = \operatorname{hilbert}(X_b)$ yields a
   complex Hermitian positive-definite matrix $\Gamma_b = \frac{1}{T} Z_b Z_b^H$ whose imaginary part
   retains non-zero-lag phase co-variation (immune to zero-lag volume conduction) that real
   covariance discards. At ~30 days, this lifted 159-way ID on PainLab from 74.8% to 83.9% (+5.4
   points from the imaginary part, +4.1 from the estimator) and 60-way ID on ds004148 from 46.7% to
   59.8% (+8.1 imaginary, +5.6 estimator).
2. **§1b — Dropping the delta band (`H_DELTA_HELDOUT`):** Loop A found delta (1–4 Hz) is largely
   day-specific (0.989 within-session vs 0.710 cross-day). Dropping delta (keeping 5 bands, 4–45 Hz)
   improved 16-way 30-day ID on PainLab (93.75% → 95.50%, NLL 0.292 → 0.229), though on Zhou 2016
   with a *single* calibration day it fell (0.790 → 0.735) and with two pooled calibration days was
   0.975 → 0.963.

This experiment tests the pre-registered $2 \times 2$ factorial (**Real OAS vs Complex Hermitian**
$\times$ **6 bands [1–45 Hz] vs 5 bands [4–45 Hz]**) on the Track 2 cross-day harness.

---

## 2. Experimental Arms

All arms use 4th-order zero-phase Butterworth band-pass filters (`scipy.signal.butter(4, band,
btype="bandpass", fs=sfreq, output="sos")` + `sosfiltfilt` along the 4-s window, $T = 480$ at
$120\text{ Hz}$), followed by `StandardScaler` and `LogisticRegression(C=1.0, max_iter=5000)` fitted
strictly on the enrolled dev participants' calibration windows (`d["calib"]`).

| Arm ID | Bands | Covariance & Tangent Projection | Features per window ($C$ channels) | Role |
|---|---|---|---|---|
| `real_6b` | 6 bands: `(1,4), (4,8), (8,13), (13,20), (20,30), (30,45)` Hz | Real OAS covariance $\to$ real Riemannian Fréchet mean $\bar{S}_b$ on calibration $\to$ $\log(\bar{S}_b^{-1/2} S_b \bar{S}_b^{-1/2})$ upper triangle (off-diag $\times \sqrt{2}$) | $6 \times \frac{C(C+1)}{2}$ | **Shipped anchor** (reproduces `bank["fp_logp"]`) |
| `real_5b` | 5 bands: `(4,8), (8,13), (13,20), (20,30), (30,45)` Hz | Same as `real_6b`, without delta (1–4 Hz) | $5 \times \frac{C(C+1)}{2}$ | **Candidate (§1b)** |
| `complex_6b` | 6 bands (1–45 Hz) | Analytic signal $Z_b = \operatorname{hilbert}(X_b)$ $\to$ $\Gamma_b = \frac{1}{T} Z_b Z_b^H$ $\to$ trace shrinkage $\Gamma_{\text{reg},b} = (1-\rho)\Gamma_b + \rho \frac{\operatorname{tr}(\operatorname{Re}(\Gamma_b))}{C} I$ ($\rho = 0.05$) $\to$ complex HPD Fréchet mean $\bar{\Gamma}_b$ on calibration (`max_iter=50, tol=1e-5, eps=0.5`) $\to$ $L_b = \log(\bar{\Gamma}_b^{-1/2} \Gamma_{\text{reg},b} \bar{\Gamma}_b^{-1/2})$ vectorized as $[\operatorname{diag}(\operatorname{Re}(L_b)), \sqrt{2}\operatorname{triu}_1(\operatorname{Re}(L_b)), \sqrt{2}\operatorname{triu}_1(\operatorname{Im}(L_b))]$ | $6 \times C^2$ | **Candidate (§1a)** |
| `complex_5b` | 5 bands (4–45 Hz) | Same as `complex_6b`, without delta (1–4 Hz) | $5 \times C^2$ | **Candidate (§1a + §1b)** |
| `herm_real_6b` | 6 bands (1–45 Hz) | $\operatorname{Re}(\Gamma_{\text{reg},b})$ whitened by $\operatorname{Re}(\bar{\Gamma}_b)$ (real tangent vectorization) | $6 \times \frac{C(C+1)}{2}$ | **Diagnostic control** (isolates $\rho=0.05$ estimator vs $\operatorname{Im}(L_b)$ phase lag) |
| `herm_real_5b` | 5 bands (4–45 Hz) | Same as `herm_real_6b`, without delta | $5 \times \frac{C(C+1)}{2}$ | **Diagnostic control** |

---

## 3. Datasets, Splits, and Leakage Controls (FM-1, FM-2, FM-8)

1. **Primary cross-day cohort (17 dev people; `research/integration-harness/src/data.py`):**
   - `bnci2014`: $K = 6$ dev people, 22 ch, 4 classes, session 0 $\to$ session 1.
   - `bnci2015`: $K = 8$ dev people, 13 ch, 2 classes, session 0 $\to$ session 1.
   - `zhou2016`: $K = 3$ dev people, 14 ch, 3 classes, sessions 0+1 $\to$ session 2.
   - **Never load the 8 locked holdout people** (`part="dev"` only).
2. **Secondary within-session check (14 fresh people):**
   - `dreyer_sim2` (`data.load_dreyer_sim2()`): $K = 14$ people, 27 ch, 2 classes, R1–R3 $\to$ R4–R6.
   - Reused Dreyer R4–R6 (subjects 61–81) are **not** loaded.
3. **Secondary single-day calibration diagnostic on `zhou2016` (report-only):**
   - Because §1b noted a single-day counter-result on Zhou 2016 (0.790 $\to$ 0.735), also report ID
     accuracy and NLL on `zhou2016` dev people when calibrated on session 0 alone ($\to$ session 2)
     and on session 1 alone ($\to$ session 2).
4. **Strict calibration-only fitting (FM-1, FM-2):**
   - Every arm sees the exact same calibration windows (`d["calib"]`). Reference Fréchet means,
     feature scalers, and logistic regression weights are fitted on `d["calib"]` only, with zero access
     to test windows.

---

## 4. Pure-PyTorch Export Specification (§1a Export)

In `submission.py`, inference runs in pure PyTorch without SciPy or pyriemann:
- On a fixed window length $T$, both `sosfiltfilt` and `scipy.signal.hilbert` along time are linear
  operators on $\mathbb{R}^T$:
  - $M_b = \operatorname{sosfiltfilt}(\text{sos}_b, I_T) \in \mathbb{R}^{T \times T}$
  - $H_T = \operatorname{hilbert}(I_T, \text{axis}=0) \in \mathbb{C}^{T \times T}$
  - Their product $F_b = (M_b H_T).astype(\text{complex64}) \in \mathbb{C}^{T \times T}$ maps real
    input $X \in \mathbb{R}^{B \times C \times T}$ directly to the analytic band-passed signal
    $Z_b = X_{\mathbb{C}} F_b \in \mathbb{C}^{B \times C \times T}$ in one matrix multiplication per
    band.
- `torch.linalg.eigh` supports complex Hermitian tensors (`complex64` / `complex128`), so tangent
  projection $L_b = \log(\bar{\Gamma}_b^{-1/2} \Gamma_{\text{reg},b} \bar{\Gamma}_b^{-1/2})$,
  vectorization, and the folded linear classifier (`weight / scale`, `intercept - weight @ mean`) run
  in pure PyTorch (`ComplexFBFingerprint`).
- **Export parity gate:** Across all test windows of the dev datasets, the exported PyTorch module
  (`complex64` buffers) must match the reference pipeline's predicted probabilities to
  $\max |\Delta p| < 10^{-4}$ (and $< 10^{-6}$ in `complex128`/`float64`), with identical top-1
  predictions ($\ge 99.9\%$ agreement).
- **Size & speed diagnostic:** Report serialized state-dict size and CPU inference time at $T = 480$
  (120 Hz) and projected buffer size at $T = 2000$ (500 Hz, 4 s, 43 EEG channels).

---

## 5. Metrics and Pre-Registered Decision Rule (FM-11)

### 5.1 Metrics reported per arm

1. **Cross-day person identification (17 dev people):**
   - Per-person test-session recall $r_{m,i}$ (mean over a dataset's $K$ people = balanced ID
     accuracy) and per-person mean test-session negative log-likelihood $\ell_{m,i} = -\frac{1}{N_i}\sum_{x \in \text{test}_i} \log p_m(k = i \mid x)$.
   - Paired per-person differences against `real_6b` across all 17 dev people:
     $\Delta r_i = r_{m,i} - r_{\text{real\_6b},i}$ and
     $\Delta \ell_i = \ell_{m,i} - \ell_{\text{real\_6b},i}$ (negative $\Delta \ell_i$ = better NLL),
     with 5,000-resample person-bootstrap 95% CIs (`rng = np.random.default_rng(0)`), better/worse/tied
     person counts, and exact Clopper-Pearson 95% CIs on non-tied win fractions ($N \le 17$, FM-11).
2. **End-to-end Track 2 task decoding balanced accuracy (17 dev people, seeds 0–2):**
   - Using the read-only harness banks (`outputs/t2-int-harness/bank_{name}_seed{0,1,2}.npz`), route
     the banked per-person C3 expert log-probabilities (`metrics.c3_logp(bank)`) with each arm's
     `fp_logp` via `metrics.route(fp_logp, metrics.c3_logp(bank))`.
   - Compute seed-mean per-person balanced accuracy (`metrics.seed_mean_scores`) and the pooled
     17-person paired comparison (`metrics.pooled_comparison`) against `row 2` (`real_6b`) and
     `row 3` (oracle identity).
   - Also report accuracy split by windows where `real_6b` was right vs wrong (routing diagnostic).
3. **Within-session check (`dreyer_sim2`, 14 fresh people, seed 0):**
   - Person-ID balanced accuracy, person-ID NLL, and end-to-end task balanced accuracy delta vs
     `row 2`.

### 5.2 Decision rule

**Context from Phase 0 (`research/integration-harness/results/references.md`):** On the 17 dev people,
`real_6b` already achieves 1.000 (`bnci2014`, 6 people), 0.953 (`bnci2015`, 8 people), and 0.747
(`zhou2016`, 3 people); 13 of the 17 people have zero fingerprint errors, so oracle identity
(`row 3`) beats `row 2` by only **+0.0070** (`[+0.0000, +0.0206]`, 4 better / 1 worse). We therefore
evaluate both:

- **Criterion A — Track 2 standalone dev candidate bar (`metrics.adopt` from the spec):**
  Pooled 17-person task balanced-accuracy gain over `row 2` $\ge +0.005$ with 95% bootstrap CI lower
  bound $> 0$, and `dreyer_sim2` task delta $\ge -0.010$.
- **Criterion B — Fingerprint replacement bar (for upgrading `FBFingerprint` / harness banks):**
  A candidate arm $m \in \{\text{real\_5b}, \text{complex\_6b}, \text{complex\_5b}\}$ is adopted as
  the better cross-day fingerprint if all three hold on the 17 dev people:
  1. **Cross-day ID improvement:** Pooled 17-person paired ID accuracy gain $\operatorname{mean}(\Delta r) \ge +0.010$
     with 95% bootstrap CI lower bound $> 0$, **or** pooled 17-person paired ID NLL reduction
     $\operatorname{mean}(-\Delta \ell) \ge +0.020$ nats with 95% bootstrap CI lower bound $> 0$.
  2. **End-to-end task non-degradation:** Pooled 17-person paired task balanced-accuracy difference
     vs `row 2` has mean $\ge 0.000$ and 95% bootstrap CI lower bound $\ge -0.002$, and `dreyer_sim2`
     task delta $\ge -0.002$.
  3. **Export parity:** Pure-PyTorch `complex64`/`float32` export matches reference probabilities to
     $\max |\Delta p| < 10^{-4}$.

**Tie-breaking among qualifying candidates:** Lowest mean cross-day person-ID NLL on the 17 dev people
(if tied within $0.005$ nats, prefer higher cross-day ID accuracy, then fewer parameters:
`real_5b < real_6b < complex_5b < complex_6b`). If no candidate passes Criterion B, keep `real_6b`
unchanged and report a null result.
