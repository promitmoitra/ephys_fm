# Protocol 01 Results: Test-Time Reference Covariance Adaptation for Per-Person Task Experts

Generated automatically by `research/recenter-pooled/src/evaluate_online_recenter.py`.
Evaluation uses the 17 cross-day dev people (`bnci2014` 6, `bnci2015` 8, `zhou2016` 3) and
14 fresh `dreyer_sim2` people. The 8 locked holdout people were never loaded.

## 1. Level 1 — Standalone Per-Person `RiemannExperts` Task Decoding (Test Balanced Accuracy)

| Arm | Test-Time $\bar{C}_k$ Adaptation | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | $\Delta$ vs `static_cal_ref` (95% CI) | B/W/T | `dreyer_sim2` (14) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `static_cal_ref` | Fixed calibration $\bar{C}_k^{\text{cal}}$ (`row 2` Riemann stream) | 0.6672 | 0.7881 | 0.7556 | **0.7397** | — | — | 0.6762 |
| `ema_tangent_ordered` | Causal tangent EMA ($\alpha=0.05$, dataset order) | 0.6858 | 0.8644 | 0.8622 | **0.8010** | +0.0612 [+0.0265, +0.1034] | 13/3/1 | 0.6946 |
| `ema_tangent_shuffled` | Causal tangent EMA ($\alpha=0.05$, **FM-10 shuffled**) | 0.6904 | 0.8431 | 0.8556 | **0.7914** | +0.0517 [+0.0193, +0.0911] | 13/4/0 | 0.6845 |
| `ema_spd_ordered` | Causal SPD Log-Euclidean EMA ($\alpha=0.05$, dataset order) | 0.6852 | 0.8606 | 0.8756 | **0.8013** | +0.0616 [+0.0265, +0.1026] | 11/4/2 | 0.6946 |
| `ema_spd_shuffled` | Causal SPD Log-Euclidean EMA ($\alpha=0.05$, **FM-10 shuffled**) | 0.6875 | 0.8456 | 0.8600 | **0.7924** | +0.0526 [+0.0181, +0.0949] | 11/6/0 | 0.6863 |
| `batch64_ordered` | Stateless batch-64 tangent shift ($N_0=10$, dataset order) | 0.6991 | 0.8644 | 0.8956 | **0.8115** | +0.0718 [+0.0351, +0.1169] | 14/1/2 | 0.6911 |
| `batch64_shuffled` | Stateless batch-64 tangent shift ($N_0=10$, **FM-10 shuffled**) | 0.6782 | 0.8206 | 0.8444 | **0.7746** | +0.0349 [+0.0130, +0.0595] | 11/3/3 | 0.6929 |
| `session_fp_tangent` | Full-session `fp_logp`-weighted tangent shift ($N_0=10$) | 0.6916 | 0.8556 | 0.8600 | **0.7985** | +0.0588 [+0.0232, +0.1046] | 12/4/1 | 0.6887 |
| `session_fp_spd` | Full-session `fp_logp`-weighted SPD Fréchet mean ($N_0=10$) | 0.6950 | 0.8569 | 0.8578 | **0.7999** | +0.0602 [+0.0250, +0.1052] | 13/3/1 | 0.6899 |
| `oracle_session_spd` | Oracle test-session $\bar{C}_k^{\text{test}}$ (routed by `fp_logp`) | 0.6950 | 0.8613 | 0.8689 | **0.8039** | +0.0642 [+0.0285, +0.1085] | 12/2/3 | 0.6881 |
| `oracle_session_spd_oracle_id` | Oracle test-session $\bar{C}_k^{\text{test}}$ + **oracle ID** | 0.6985 | 0.8781 | 0.8933 | **0.8174** | +0.0777 [+0.0356, +0.1266] | 13/1/3 | 0.6875 |
| `complex6b_static_cal_ref` | Fixed $\bar{C}_k^{\text{cal}}$ routed by `complex_6b` | 0.6661 | 0.7888 | 0.7400 | **0.7369** | -0.0029 [-0.0083, +0.0006] | 2/4/11 | 0.6762 |
| `complex6b_ema_tangent_shuffled` | Causal tangent EMA (**FM-10 shuffled**) w/ `complex_6b` | 0.6892 | 0.8538 | 0.8867 | **0.8015** | +0.0618 [+0.0269, +0.1024] | 11/2/4 | 0.6863 |
| `complex6b_session_fp_spd` | Full-session SPD Fréchet mean w/ `complex_6b` | 0.6979 | 0.8650 | 0.8933 | **0.8110** | +0.0713 [+0.0328, +0.1176] | 14/2/1 | 0.6917 |

## 2. Level 2 — End-to-End Track 2 Integration (`EEGNet` + Adapted `RiemannExperts` via `C3`, Seeds 0–2)

| Arm | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | $\Delta$ vs `row 2` (95% CI) | B/W/T vs `row 2` | `dreyer_sim2` (14) | Adopt? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `row1_pooled_eegnet` | 0.6829 | 0.8156 | 0.7956 | **0.7652** | — | — | 0.9304 | ref |
| `row2_current` | 0.7870 | 0.8467 | 0.8511 | **0.8264** | — | — | 0.9357 | ref |
| `row3_oracle` | 0.7870 | 0.8610 | 0.8526 | **0.8334** | — | — | 0.9363 | ref |
| `ema_tangent_ordered` | 0.7884 | 0.8683 | 0.8793 | **0.8420** | +0.0156 [+0.0055, +0.0273] | 12/4/1 | 0.9393 | **YES** |
| `ema_tangent_shuffled` | 0.7882 | 0.8642 | 0.8681 | **0.8381** | +0.0117 [+0.0030, +0.0223] | 11/6/0 | 0.9369 | **YES** |
| `ema_spd_ordered` | 0.7899 | 0.8690 | 0.8815 | **0.8433** | +0.0169 [+0.0066, +0.0288] | 12/4/1 | 0.9399 | **YES** |
| `ema_spd_shuffled` | 0.7855 | 0.8646 | 0.8696 | **0.8376** | +0.0112 [+0.0014, +0.0233] | 10/7/0 | 0.9369 | **YES** |
| `batch64_ordered` | 0.7915 | 0.8683 | 0.8807 | **0.8434** | +0.0170 [+0.0078, +0.0277] | 15/2/0 | 0.9369 | **YES** |
| `batch64_shuffled` | 0.7907 | 0.8569 | 0.8659 | **0.8351** | +0.0087 [+0.0037, +0.0147] | 13/3/1 | 0.9375 | **YES** |
| `session_fp_tangent` | 0.7899 | 0.8671 | 0.8681 | **0.8400** | +0.0136 [+0.0047, +0.0251] | 11/5/1 | 0.9375 | **YES** |
| `session_fp_spd` | 0.7897 | 0.8677 | 0.8719 | **0.8409** | +0.0145 [+0.0053, +0.0259] | 11/5/1 | 0.9381 | **YES** |
| `oracle_session_spd` | 0.7897 | 0.8685 | 0.8741 | **0.8417** | +0.0153 [+0.0056, +0.0275] | 11/4/2 | 0.9387 | **YES** |
| `oracle_session_spd_oracle_id` | 0.7899 | 0.8852 | 0.8867 | **0.8518** | +0.0254 [+0.0080, +0.0485] | 11/4/2 | 0.9393 | **YES** |
| `complex6b_static_cal_ref` | 0.7872 | 0.8496 | 0.8511 | **0.8278** | +0.0014 [-0.0003, +0.0044] | 4/2/11 | 0.9357 | no |
| `complex6b_ema_tangent_shuffled` | 0.7882 | 0.8665 | 0.8815 | **0.8415** | +0.0151 [+0.0042, +0.0280] | 10/6/1 | 0.9375 | **YES** |
| `complex6b_session_fp_spd` | 0.7892 | 0.8725 | 0.8844 | **0.8452** | +0.0188 [+0.0067, +0.0332] | 11/6/0 | 0.9393 | **YES** |

## 3. Mandatory `FM-10` Window-Order Controls (Dataset Order vs Random Interleaving Across Subjects)

| Order Comparison (17 Cross-Day Dev People) | Paired $\Delta$ BA (Ordered − Shuffled) | 95% Bootstrap CI | Better / Worse / Tied |
|---|---:|---:|---:|
| Level 1: `ema_tangent_ordered − ema_tangent_shuffled` | +0.0095 | [-0.0005, +0.0204] | 10 / 6 / 1 |
| Level 1: `ema_spd_ordered − ema_spd_shuffled` | +0.0090 | [-0.0003, +0.0197] | 10 / 6 / 1 |
| Level 1: `batch64_ordered − batch64_shuffled` | +0.0370 | [+0.0188, +0.0607] | 15 / 1 / 1 |
| Level 2: `ema_tangent_ordered − ema_tangent_shuffled` | +0.0040 | [+0.0005, +0.0079] | 11 / 3 / 3 |
| Level 2: `ema_spd_ordered − ema_spd_shuffled` | +0.0057 | [+0.0024, +0.0092] | 13 / 3 / 1 |
| Level 2: `batch64_ordered − batch64_shuffled` | +0.0083 | [+0.0033, +0.0137] | 12 / 4 / 1 |

## 4. Per-Person Test Balanced Accuracy on 17 Dev People (Level 1 Standalone & Level 2 C3 Mixture)

| Dataset | Person | L1 `static_cal_ref` | L1 `ema_tangent_shuffled` | L1 `session_fp_spd` | L1 `oracle_session_spd` | L2 `row2_current` | L2 `ema_tangent_shuffled` | L2 `session_fp_spd` | L2 `oracle_session_spd` | L2 `complex6b_session_fp_spd` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bnci2014` | `1` | 0.8090 | 0.7917 | 0.7986 | 0.8056 | 0.8924 | 0.8889 | 0.8912 | 0.8900 | 0.8900 |
| `bnci2014` | `4` | 0.6806 | 0.6840 | 0.6875 | 0.6840 | 0.7396 | 0.7373 | 0.7350 | 0.7326 | 0.7326 |
| `bnci2014` | `5` | 0.4688 | 0.4826 | 0.4688 | 0.4653 | 0.7153 | 0.7153 | 0.7188 | 0.7199 | 0.7222 |
| `bnci2014` | `6` | 0.5417 | 0.5451 | 0.5556 | 0.5556 | 0.6157 | 0.6123 | 0.6157 | 0.6169 | 0.6146 |
| `bnci2014` | `7` | 0.7153 | 0.8229 | 0.8368 | 0.8368 | 0.8889 | 0.9062 | 0.9086 | 0.9109 | 0.9086 |
| `bnci2014` | `8` | 0.7882 | 0.8160 | 0.8229 | 0.8229 | 0.8704 | 0.8692 | 0.8692 | 0.8681 | 0.8669 |
| `bnci2015` | `1` | 0.9450 | 0.9900 | 0.9900 | 0.9900 | 0.9717 | 0.9950 | 0.9967 | 0.9967 | 0.9967 |
| `bnci2015` | `2` | 0.9750 | 0.9600 | 0.9650 | 0.9750 | 0.9733 | 0.9767 | 0.9783 | 0.9750 | 0.9767 |
| `bnci2015` | `3` | 0.9100 | 0.9500 | 0.9550 | 0.9550 | 0.9683 | 0.9667 | 0.9650 | 0.9667 | 0.9650 |
| `bnci2015` | `4` | 0.9350 | 0.9250 | 0.9400 | 0.9350 | 0.9167 | 0.9200 | 0.9233 | 0.9167 | 0.9183 |
| `bnci2015` | `5` | 0.5500 | 0.8350 | 0.8850 | 0.8750 | 0.6967 | 0.7717 | 0.7800 | 0.7883 | 0.7817 |
| `bnci2015` | `6` | 0.5200 | 0.5850 | 0.5950 | 0.6400 | 0.6567 | 0.6867 | 0.6900 | 0.6933 | 0.7283 |
| `bnci2015` | `7` | 0.9300 | 0.9100 | 0.9250 | 0.9300 | 0.9300 | 0.9183 | 0.9233 | 0.9300 | 0.9283 |
| `bnci2015` | `12` | 0.5400 | 0.5900 | 0.6000 | 0.5900 | 0.6600 | 0.6783 | 0.6850 | 0.6817 | 0.6850 |
| `zhou2016` | `2` | 0.7133 | 0.8667 | 0.8867 | 0.8933 | 0.8844 | 0.8933 | 0.8933 | 0.8956 | 0.8978 |
| `zhou2016` | `3` | 0.8333 | 0.8867 | 0.8733 | 0.9067 | 0.8311 | 0.8356 | 0.8422 | 0.8578 | 0.8467 |
| `zhou2016` | `4` | 0.7200 | 0.8133 | 0.8133 | 0.8067 | 0.8378 | 0.8756 | 0.8800 | 0.8689 | 0.9089 |
