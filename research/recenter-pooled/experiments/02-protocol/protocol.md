# Protocol 02: Per-Person $\mu$-Peak Band Centering for `RiemannExperts` (Handoff §2c)

Locked 2026-10-17 on branch `exp/recenter-pooled` (worktree `.claude/worktrees/t2-recenter`) before
any per-person $\mu$-band task-decoding code or evaluation numbers exist on the Track 2 harness.

**Sources:**
- `NEURALPRINT_HANDOFF.md` (§2c Per-person $\mu$ band for the Riemannian expert, and Rules FM-1 to
  FM-11).
- `/home/promit/Documents/neuralprint/experiments/h-alpha-peak/code/run_experiment.py` (DPSS
  multitaper power spectrum `NW=2, K=3` tapers, $7\text{--}14$ Hz peak range).

---

## 1. Question and Mechanism

`NEURALPRINT_HANDOFF.md` §2c notes that the $\alpha/\mu$ peak frequency ($f_{\mu,k} \in [7, 14]$ Hz)
is the single most stable spectral trait across cap changes ($\text{ICC}(2,1) = 0.87$ across a month
on `ds004148`). Meanwhile, the shipped `RiemannExperts` (`track2/riemann_parts.py`) uses a fixed
`8–30 Hz` bandpass filter for every participant (implicitly assuming a canonical $10\text{ Hz}$
$\mu$ peak with a $[-2, +20]\text{ Hz}$ passband). If a participant's sensorimotor $\mu$ peak sits
at $7.5\text{ Hz}$, an $8\text{--}30\text{ Hz}$ filter attenuates the lower half of their $\mu$
desynchronization; if it sits at $12\text{ Hz}$, $8\text{--}10\text{ Hz}$ contributes mostly background
$\theta/\text{low-}\alpha$ noise.

We test whether estimating each participant $k$'s $\mu$ peak frequency $f_{\mu,k}$ strictly from
their calibration windows (`d["calib"]`, Rule FM-1) and centering their `RiemannExperts` passband on
$f_{\mu,k}$ improves cross-day task decoding — both with static calibration references and on top of
Protocol 01's test-time reference adaptation (`ema_tangent_shuffled` and `session_fp_spd`).

---

## 2. Calibration-Only Individual $\mu$-Peak Estimator (`estimate_mu_peak`)

For each participant $k \in \{0 \dots K-1\}$ using only their calibration windows (`0.5–4.0 s` crop):
1. Compute the DPSS multitaper power spectrum (`NW = 2.0, K = 3` tapers, channel mean removed)
   averaged across participant $k$'s calibration windows and channels: $\bar{P}_k(f)$.
2. Fit the $1/f$ aperiodic log-log line $\log_{10} \bar{P}_k(f) \approx b_k - \chi_k \log_{10} f$ by
   ordinary least squares over the flanking frequencies $f \in [2.0, 6.0] \cup [30.0, 45.0]\text{ Hz}$.
3. Compute the $1/f$-detrended residual spectrum $R_k(f) = \log_{10} \bar{P}_k(f) - (b_k - \chi_k \log_{10} f)$
   on $f \in [7.0, 14.0]\text{ Hz}$.
4. If $\max_{f \in [7, 14]} R_k(f) > 0.05$, refine the argmax by 3-point parabolic interpolation to
   obtain $f_{\mu,k} \in [7.0, 14.0]\text{ Hz}$; otherwise fall back to the canonical
   $f_{\mu,k} = 10.0\text{ Hz}$.

---

## 3. Pre-Registered Experimental Arms

All arms use 4th-order zero-phase Butterworth filtering, `0.5–4.0 s` window crop, OAS covariance,
and per-person `LogisticRegression(C=0.1)`:

| Arm ID | Per-Person Passband(s) for Participant $k$ ($\Delta f_k = f_{\mu,k} - 10.0\text{ Hz}$) | Test-Time Reference $\bar{C}_k$ | Role |
|---|---|---|---|
| `fixed_8_30_static` | Fixed `[8.0, 30.0]` Hz | Static $\bar{C}_k^{\text{cal}}$ | Shipped `RiemannExperts` baseline (`row 2` in C3) |
| `mu_shift_wide_static` | Shifted宽band $[\operatorname{clip}(8+\Delta f_k, 5.5, 11.0),\; \operatorname{clip}(30+\Delta f_k, 26.0, 34.0)]$ Hz | Static $\bar{C}_k^{\text{cal}}$ | **§2c core:** shifts `[8, 30]` Hz by $f_{\mu,k} - 10$ Hz |
| `mu_lower_edge_static` | Lower-edge anchored $[\operatorname{clip}(f_{\mu,k}-2.0, 5.5, 11.0),\; 30.0]$ Hz | Static $\bar{C}_k^{\text{cal}}$ | **§2c lower-edge:** shifts only the $\mu$ lower cutoff |
| `mu_beta_2band_static` | 2-band $[\operatorname{clip}(f_{\mu,k}-2.5, 5.0, 11.0),\; f_{\mu,k}+2.5]$ Hz + $[f_{\mu,k}+2.5,\; 30.0]$ Hz | Static $\bar{C}_k^{\text{cal}}$ | **§2c 2-band:** separates individualized $\mu$ and $\beta$ covariances |
| `fixed_8_30_ema_shuf` | Fixed `[8.0, 30.0]` Hz | Causal tangent EMA ($\alpha=0.05$, **FM-10 shuffled**) | Protocol 01 online baseline |
| `mu_shift_wide_ema_shuf` | Shifted $[\operatorname{clip}(8+\Delta f_k, 5.5, 11.0),\; \operatorname{clip}(30+\Delta f_k, 26.0, 34.0)]$ Hz | Causal tangent EMA ($\alpha=0.05$, **FM-10 shuffled**) | Tests §2c + Protocol 01 causal shuffled EMA |
| `mu_lower_edge_ema_shuf` | Lower-edge $[\operatorname{clip}(f_{\mu,k}-2.0, 5.5, 11.0),\; 30.0]$ Hz | Causal tangent EMA ($\alpha=0.05$, **FM-10 shuffled**) | Tests §2c lower-edge + Protocol 01 causal shuffled EMA |
| `mu_beta_2band_ema_shuf` | 2-band individualized $\mu + \beta$ | Causal tangent EMA ($\alpha=0.05$, **FM-10 shuffled**) | Tests §2c 2-band + Protocol 01 causal shuffled EMA |
| `mu_shift_wide_sess_spd` | Shifted $[\operatorname{clip}(8+\Delta f_k, 5.5, 11.0),\; \operatorname{clip}(30+\Delta f_k, 26.0, 34.0)]$ Hz | Full-session `fp_logp`-weighted SPD Fréchet mean | Tests §2c + Protocol 01 SPD session centering |

---

## 4. Evaluation and Decision Rule (FM-11)

- Evaluated on the **17 cross-day dev people** (`bnci2014` 6, `bnci2015` 8, `zhou2016` 3, seeds 0–2)
  and **14 fresh `dreyer_sim2` people** (seed 0). Never loads the 8 locked holdout people.
- **Decision rule:** A §2c $\mu$-band arm is adopted over its fixed `8–30 Hz` counterpart if and only
  if its paired 17-person difference (`mu_* − fixed_8_30_*`) has mean $\ge +0.005$ with 95%
  person-bootstrap CI lower bound $> 0$ and `dreyer_sim2` delta $\ge -0.010$.
