# Protocol 00: Fingerprint-Routed Covariance Re-Centering for Task Decoding (§2a)

Locked 2026-10-17 on branch `exp/recenter-pooled` (worktree `.claude/worktrees/t2-recenter`) before
any re-centering code or evaluation numbers exist on the Track 2 cross-day harness.

**Sources:**
- `NEURALPRINT_HANDOFF.md` (§2a Fingerprint-routed re-centering, and Methodological Rules FM-1
  through FM-11).
- `docs/superpowers/specs/2026-10-05-identity-integration-design.md` (§1 Shared evaluation,
  candidate adoption rule).
- `docs/handoff/identity-integration-loops.md` (Phase 0 harness banks and reference rows).

---

## 1. Question and Mechanism

`research/literature/survey.md` previously dismissed Riemannian covariance re-centering (Zanini et
al. 2018; Rodrigues et al. 2019) because `predict(X)` receives no participant IDs at test time.
However, the fingerprint supplies $p(\text{person } k \mid x)$ per window.

Instead of routing $K$ separate per-person task classifiers (each fitted on only that person's
80–200 calibration windows), **§2a routes the reference covariance $\bar{C}_k$**:
- Each person $k$'s calibration covariances have Fréchet mean $\bar{C}_k$ (whitener
  $W_k = \bar{C}_k^{-1/2}$).
- Whitening person $k$'s covariances by $W_k$ moves every person's calibration barycenter to the
  identity matrix $I_C$, so $\operatorname{tangent}(C, W_k) = \operatorname{vec}(\log(W_k C W_k))$
  places all $K$ participants into a shared, zero-centered tangent space $T_I \mathcal{S}_{++}^C$.
- A **single pooled task classifier** can then be trained on all calibration participants'
  re-centered tangent vectors pooled ($K \times$ more windows than a single person's expert).
- At test time, without true IDs, the window is re-centered using the fingerprint posterior
  $p(k \mid x)$ — either by routing across the $K$ per-person tangent projections or by whitening
  with the convex combination $\bar{W}(x) = \sum_k p(k \mid x) W_k$ (which requires only one
  eigendecomposition per window) — and scored by the single pooled classifier.

Per **Rule FM-2**, every arm sees the exact same calibration data (`d["calib"]` on cross-day
datasets; `d["pool"] | d["calib"]` on `dreyer_sim2`), and we test both:
1. **Level 1 — Standalone tangent-space task decoding** (comparing pooled uncentered, pooled
   re-centered routed, pooled re-centered with oracle ID ceiling, and per-person Riemannian experts
   on the same calibration windows).
2. **Level 2 — End-to-end Track 2 mixture decoding** (testing whether replacing or regularizing the
   per-person experts with the pooled re-centered model beats `row 2`, the shipped E+T C3 mixture).

---

## 2. Experimental Arms

### Level 1: Standalone Tangent-Space Task Decoders (no EEGNet)

All 8–30 Hz arms use the shipped `RiemannExperts` preprocessing (`track2/riemann_parts.py`: 4th-order
zero-phase Butterworth 8–30 Hz, 0.5–4.0 s crop, OAS covariance).

| Arm ID | Reference Covariance | Task Classifier | Identity at Test Time | Role |
|---|---|---|---|---|
| `ts_global_pooled` | 1 global Fréchet mean $\bar{C}_{\text{global}}$ over all calibration windows | 1 pooled `LogisticRegression(C=0.1)` on all calibration windows | None (identity-blind) | Uncentered pooled baseline |
| `ts_recenter_routed` | Per-person calibration Fréchet mean $\bar{C}_k$ ($W_k = \bar{C}_k^{-1/2}$) | 1 pooled `LogisticRegression(C=0.1)` fitted on all people's re-centered calibration tangent vectors | Soft-routed by `bank["fp_logp"]` (`real_6b`) | **§2a core standalone** (tangent mixture $\sum_k p(k\mid x) p_{\text{pool}}(c \mid v_k(x))$) |
| `ts_recenter_whitener` | Interpolated whitener $\bar{W}(x) = \sum_k p(k\mid x) W_k$ (1 `eigh` / window) | Same 1 pooled `LogisticRegression(C=0.1)` as `ts_recenter_routed` | Soft-routed whitener via `bank["fp_logp"]` | **§2a single-pass whitener variant** |
| `ts_recenter_oracle` | Per-person calibration Fréchet mean $\bar{C}_{k_{\text{true}}}$ | Same 1 pooled `LogisticRegression(C=0.1)` as `ts_recenter_routed` | **Oracle identity** $k_{\text{true}}$ | **§2a standalone ceiling** |
| `ts_recenter_fb_routed` | Per-band per-person $\bar{C}_{b,k}$ across 5 bands `(4,8),(8,13),(13,20),(20,30),(30,45)` Hz (0.5–4.0 s) | `StandardScaler` + 1 pooled `LogisticRegression(C=0.1)` on all calibration windows | Soft-routed by `bank["fp_logp"]` | **§2a multi-band pooled** |
| `ts_recenter_fb_oracle` | Same 5-band per-person $\bar{C}_{b,k}$ | Same 1 pooled 5-band classifier | **Oracle identity** $k_{\text{true}}$ | **§2a multi-band ceiling** |
| `ts_per_person_routed` | Per-person calibration Fréchet mean $\bar{C}_k$ | $K$ separate per-person `LogisticRegression(C=0.1)` (`bank["riemann_logp"]`) | Soft-routed by `bank["fp_logp"]` | **FM-2 per-person retraining baseline** |
| `ts_per_person_oracle` | Per-person calibration Fréchet mean $\bar{C}_{k_{\text{true}}}$ | $K$ separate per-person `LogisticRegression(C=0.1)` | **Oracle identity** $k_{\text{true}}$ | Per-person Riemannian ceiling |

### Level 2: End-to-End Track 2 Integration Arms (compared against `row 2`)

All Level 2 arms use the shipped C3 combiner (`metrics.c3_logp` coefficients from
`track2/riemann_parts.py`, zero test-set refitting) and are compared directly against `row 2`
(`current`, shipped E+T soft-routed by `fp_logp`) and `row 3` (`oracle` identity):

| Arm ID | Neural Stream | Riemannian Stream | Routing | Role |
|---|---|---|---|---|
| `E1_fully_pooled_recenter` | 1 `pooled_eegnet` (`bank["pooled_logp"]`) | 1 pooled re-centered `ts_recenter_routed` (combined with `pooled_logp` via C3 using dataset-mean calibration reliability) | `fp_logp` routes only $\bar{C}_k^{-1/2}$ | **Pure §2a pooled alternative to Options B/C** (zero per-person classifiers) |
| `E1_oracle` | 1 `pooled_eegnet` | `ts_recenter_oracle` | Oracle ID $k_{\text{true}}$ | Ceiling for `E1` |
| `E2_eeg_experts_plus_recenter` | $K$ per-person EEGNet experts (`bank["eeg_logp"]`) | 1 pooled re-centered model (`v_k(x)` scored by shared $w_{\text{pool}}$, replacing `bank["riemann_logp"]` in C3) | Soft-routed by `bank["fp_logp"]` | Tests replacing the $K$ per-person Riemannian experts with 1 pooled re-centered model |
| `E2_oracle` | $K$ per-person EEGNet experts | Same 1 pooled re-centered model | Oracle ID $k_{\text{true}}$ | Ceiling for `E2` |
| `E3_eeg_experts_plus_shrunk_riemann` | $K$ per-person EEGNet experts (`bank["eeg_logp"]`) | Equal log-pool of per-person `riemann_logp` and pooled re-centered `recenter_logp` (`0.5 * riemann_logp + 0.5 * recenter_logp` per person $k$ before C3) | Soft-routed by `bank["fp_logp"]` | Hierarchical shrinkage of noisy 80-window per-person Riemannian experts toward the pooled re-centered decision boundary |
| `E3_oracle` | $K$ per-person EEGNet experts | Same shrunk Riemannian log-probabilities | Oracle ID $k_{\text{true}}$ | Ceiling for `E3` |

---

## 3. Datasets, Splits, and Leakage Controls (FM-1, FM-2, FM-8)

1. **Primary cross-day cohort (17 dev people; `research/integration-harness/src/data.py`):**
   - `bnci2014` ($K=6$), `bnci2015` ($K=8$), `zhou2016` ($K=3$), seeds 0–2 from
     `outputs/t2-int-harness/bank_{name}_seed{0,1,2}.npz`.
   - The 8 locked holdout people (`2026-10-05-identity-integration-holdout.json`) are **never loaded**.
2. **Secondary within-session cohort (14 fresh people):**
   - `dreyer_sim2` (`data.load_dreyer_sim2()`, seed 0 bank).
   - For `dreyer_sim2`, `d["pool"] | d["calib"]` contains both the training-pool subjects and the
     14 evaluation subjects' R1–R3 calibration windows. Each subject $s$ in `pool | calib` is
     re-centered by their own training/calibration Fréchet mean $\bar{C}_s$ when fitting the pooled
     re-centered classifier, so `ts_recenter_*` sees the exact same windows as the pooled EEGNet
     (Rule FM-2). At test time (`d["test"]`, R4–R6 of the 14 evaluation subjects), only the 14
     evaluation subjects' calibration references $\bar{C}_k$ ($k = 0 \dots 13$) are routed by the
     14-way fingerprint.
3. **Zero test-set leakage (FM-1, FM-8):**
   - All Fréchet means $\bar{C}_k$, tangent projections, scalers, and `LogisticRegression(C=0.1)`
     models are fitted strictly on calibration/training windows. No hyperparameter is tuned on test
     sessions.

---

## 4. Metrics and Pre-Registered Decision Rules (FM-11)

1. **Level 1 (Standalone Tangent-Space Decoding, 17 dev people):**
   - Per-person test-session balanced accuracy and paired differences (`metrics.pooled_comparison`,
     5,000 bootstrap resamples, rng seed 0) + Clopper-Pearson win CIs (FM-11):
     - **H2a.1 (Re-centering vs uncentered pooled):** `ts_recenter_routed − ts_global_pooled`
       (does routing the reference covariance improve a pooled tangent-space decoder?).
     - **H2a.2 (Pooled re-centered vs per-person retraining, FM-2):**
       `ts_recenter_routed − ts_per_person_routed` and `ts_recenter_oracle − ts_per_person_oracle`
       (does pooling across re-centered subjects beat fitting separate per-person classifiers on the
       same calibration data?).
2. **Level 2 (End-to-End Track 2 Candidate Rule against `row 2`, 17 dev people, seeds 0–2):**
   - Seed-mean per-person balanced accuracy (`metrics.seed_mean_scores`) and paired comparison
     against `row 2` (`current`) and `row 3` (`oracle`).
   - **Track 2 Dev Candidate Rule (`metrics.adopt`):** An integration arm (`E1`, `E2`, or `E3`)
     becomes a Track 2 dev candidate if and only if its pooled 17-person gain over `row 2` is
     $\ge +0.005$ with 95% person-bootstrap CI lower bound $> 0$, and its `dreyer_sim2` delta vs
     `row 2` is $\ge -0.010$.
