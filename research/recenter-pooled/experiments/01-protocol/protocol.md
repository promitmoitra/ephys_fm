# Protocol 01: Test-Time Reference Covariance Adaptation for Per-Person Task Experts (§2b Follow-Up)

Locked 2026-10-17 on branch `exp/recenter-pooled` (worktree `.claude/worktrees/t2-recenter`) before
any online/test-time reference adaptation code or numbers exist on the Track 2 cross-day harness.

**Note on Handoff Numbering:**
- In `NEURALPRINT_HANDOFF.md`, §2b denotes Track 3 sleep-onset state features (`specparam`), §2c
  denotes per-person $\mu$-band centering, and §2a / §3 (lines 89–107, 139–141) cover covariance
  re-centering and the `FM-10` window-order control.
- Protocol 00 (`experiments/00-protocol/protocol.md`) tested §2a (calibration-time re-centering with
  a *single pooled* task classifier $w_{\text{pool}}$) and showed that replacing the $K$ per-person
  task classifiers $w_k$ loses `-0.0337` on the 17 cross-day dev people because different subjects
  have distinct task dipole orientations $w_k \in T_{\bar{C}_k}\mathcal{S}_{++}^C$.
- **Protocol 01** tests the direct corollary requested by the user: **keep each person $k$'s
  calibrated task classifier $(w_k, b_k)$ from `RiemannExperts` (`row 2`), and adapt person $k$'s
  reference covariance $\bar{C}_k$ during the test session using the fingerprint posterior
  $p(k \mid x)$ to counter day-to-day electrode/impedance drift.**

---

## 1. Mechanism and Mathematical Formulation

In the shipped `RiemannExperts` (`track2/submission.py`), person $k$'s linear classifier $(w_k, b_k)$
is fitted on calibration-day tangent vectors:
$$f_k^{\text{cal}}(C) = \operatorname{vec}\!\left(\log\!\left((\bar{C}_k^{\text{cal}})^{-1/2} C (\bar{C}_k^{\text{cal}})^{-1/2}\right)\right)$$
where $\bar{C}_k^{\text{cal}}$ is person $k$'s calibration-day Fréchet mean, so
$\mathbb{E}_{\text{cal}_k}[f_k^{\text{cal}}(C)] = 0$.

On a subsequent test day, cap re-application and impedance drift move person $k$'s session
barycenter from $\bar{C}_k^{\text{cal}}$ to $\bar{C}_k^{\text{test}} \neq \bar{C}_k^{\text{cal}}$
(accounting for 6–26% of covariance variance across days). Evaluating $(w_k, b_k)$ around the stale
reference $\bar{C}_k^{\text{cal}}$ introduces a systematic day-to-day shift
$\mu_k^{\text{test}} = \mathbb{E}_{\text{test}_k}[f_k^{\text{cal}}(C)] \neq 0$.

Without test-time subject IDs, each test window $x_t$ carries a fingerprint responsibility weight
$\gamma_{tk} = p(k \mid x_t) = \exp(\text{fp\_logp}_{t,k})$. We can adapt person $k$'s reference
center in two ways:
1. **Tangent-space drift subtraction (zero extra `eigh` calls):**
   $$\tilde{f}_k(C_t) = f_k^{\text{cal}}(C_t) - \hat{\mu}_k^{(t)}$$
   where $\hat{\mu}_k^{(t)}$ tracks person $k$'s test-session tangent mean weighted by $\gamma_{tk}$.
2. **SPD manifold reference covariance update ($W_k^{(t)} = (\bar{C}_k^{(t)})^{-1/2}$):**
   Update $\bar{C}_k^{(t)}$ on $\mathcal{S}_{++}^C$ (via log-Euclidean EMA
   $\log \bar{C}_k^{(t)} = (1 - \alpha \gamma_{tk})\log \bar{C}_k^{(t-1)} + \alpha \gamma_{tk}\log C_t$
   or Riemannian Fréchet mean) and re-project $\operatorname{tangent}(C_t, (\bar{C}_k^{(t)})^{-1/2})$.

---

## 2. Pre-Registered Experimental Arms and `FM-10` Window-Order Controls

Per `NEURALPRINT_HANDOFF.md` §3 (`FM-10`), `benchopt` warm-up test loaders preserve dataset/block
order (batches of 64), whereas sealed-phase window ordering is unknown. Therefore, every online or
batch adaptation arm is paired with a **mandatory `FM-10` random-shuffle control** (averaging over 3
random window permutations `seed in {0, 1, 2}` where all $K$ participants' test windows are randomly
interleaved).

Each arm is evaluated at **Level 1 (standalone `RiemannExperts` task balanced accuracy)** and
**Level 2 (combined with `bank["eeg_logp"]` via the shipped `C3` combiner and routed by `fp_logp`,
compared directly against `row 2` across seeds 0–2)**:

| Arm ID | Reference $\bar{C}_k$ at Test Time | Test-Time Access / Ordering | Hyperparameters (Pre-Registered, Fixed) | Role |
|---|---|---|---|---|
| `static_cal_ref` (`row 2` baseline) | Fixed calibration $\bar{C}_k^{\text{cal}}$ | Single-window (stateless) | None | Shipped `RiemannExperts` (`row 2` when combined with `eeg_logp` via C3) |
| `ema_tangent_ordered` | Causal online EMA of tangent mean $\hat{\mu}_k^{(t)}$ weighted by $\gamma_{tk} = p(k\mid x_t)$ | Causal 1-by-1 in dataset order | $\alpha = 0.05$ | Online test-time drift tracking (tangent space, zero extra `eigh`) |
| `ema_tangent_shuffled` (**FM-10 control**) | Same causal online EMA $\hat{\mu}_k^{(t)}$ weighted by $\gamma_{tk}$ | Causal 1-by-1 on **randomly interleaved** test windows (3 perm seeds) | $\alpha = 0.05$ | **FM-10 order-invariance check** for causal tangent EMA |
| `ema_spd_ordered` | Causal online Log-Euclidean EMA of $\bar{C}_k^{(t)}$ on $\mathcal{S}_{++}^C$: $\log \bar{C}_k^{(t)} = (1-\alpha\gamma_{tk})\log \bar{C}_k^{(t-1)} + \alpha\gamma_{tk}\log C_t$ | Causal 1-by-1 in dataset order | $\alpha = 0.05$ | Exact manifold reference covariance EMA $W_k^{(t)} = (\bar{C}_k^{(t)})^{-1/2}$ |
| `ema_spd_shuffled` (**FM-10 control**) | Same manifold EMA of $\bar{C}_k^{(t)}$ | Causal 1-by-1 on **randomly interleaved** test windows (3 perm seeds) | $\alpha = 0.05$ | **FM-10 order-invariance check** for manifold EMA |
| `batch64_ordered` | Stateless within-batch shift $\hat{\mu}_k(B) = \frac{\sum_{i\in B} \gamma_{ik} f_k(C_i)}{N_0 + \sum_{i\in B} \gamma_{ik}}$ | Stateless batches of 64 (dataset order) | $N_0 = 10$ | Matches stateless `predict(X)` contract on kit batch size 64 |
| `batch64_shuffled` (**FM-10 control**) | Same stateless within-batch shift $\hat{\mu}_k(B)$ | Stateless batches of 64 on **randomly interleaved** test windows (3 perm seeds) | $N_0 = 10$ | **FM-10 check** when each batch of 64 mixes all $K$ people randomly |
| `session_fp_tangent` | Full test-session fingerprint-weighted tangent mean $\hat{\mu}_k = \frac{\sum_i \gamma_{ik} f_k(C_i)}{N_0 + \sum_i \gamma_{ik}}$ | Full test session (order-invariant by construction) | $N_0 = 10$ | Offline/transductive fingerprint-weighted session centering |
| `session_fp_spd` | Full test-session fingerprint-weighted Riemannian Fréchet mean $\bar{C}_k^{\text{test, fp}}$ (`pyriemann.utils.mean.mean_riemann` with `sample_weight`) | Full test session (order-invariant by construction) | $N_0 = 10$ prior weight on $\bar{C}_k^{\text{cal}}$ | Exact SPD Fréchet mean adaptation using fingerprint weights |
| `oracle_session_spd` (**Ceiling**) | True-ID test-session Riemannian Fréchet mean $\bar{C}_k^{\text{test, oracle}}$ per person $k$ | Oracle test-session $\bar{C}_k^{\text{test}}$ | None | **Exact ceiling of unsupervised test-day reference adaptation** for per-person `RiemannExperts` |

Additionally, we evaluate `ema_tangent_shuffled` and `session_fp_spd` when both the reference
adaptation weights $\gamma_{tk}$ and the final mixture routing use the **`complex_6b` fingerprint**
from PR #15 (`complex6b_ema_tangent_shuffled`, `complex6b_session_fp_spd`).

---

## 3. Datasets, Splits, and Decision Rules (FM-1, FM-2, FM-10, FM-11)

1. **Datasets:**
   - **Primary cross-day cohort (17 dev people):** `bnci2014` (6), `bnci2015` (8), `zhou2016` (3),
     seeds 0–2 from `outputs/t2-int-harness/bank_{name}_seed{0,1,2}.npz`.
   - **Secondary within-session cohort (14 fresh people):** `dreyer_sim2` (`seed 0`).
   - **Locked holdout (8 people):** Never loaded.
2. **Pre-Registered Hypotheses and Adoption Criteria:**
   - **H2b.1 (Standalone per-person Riemannian cross-day drift reduction):** Does adapting
     $\bar{C}_k$ at test time (`ema_tangent_shuffled`, `ema_spd_shuffled`, `session_fp_spd`, and the
     ceiling `oracle_session_spd`) improve standalone `RiemannExperts` task balanced accuracy over
     `static_cal_ref` on the 17 cross-day dev people?
   - **H2b.2 (End-to-end Track 2 mixture gain over `row 2`):** Does replacing `bank["riemann_logp"]`
     with its drift-adapted counterpart inside the shipped `C3` combiner beat `row 2` (`0.8264`) by
     the Track 2 adoption bar (`metrics.adopt`: $\ge +0.005$ on the 17 dev people with 95%
     person-bootstrap CI lower bound $> 0$, `dreyer_sim2` delta $\ge -0.010$, **and** passing the
     `FM-10` shuffled-order check)?
