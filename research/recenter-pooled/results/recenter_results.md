# Handoff §2a Results: Fingerprint-Routed Covariance Re-Centering

Generated automatically by `research/recenter-pooled/src/evaluate_recenter.py`.
Evaluation uses the 17 cross-day dev people (`bnci2014` 6, `bnci2015` 8, `zhou2016` 3) and
14 fresh `dreyer_sim2` people. The 8 locked holdout people were never loaded.

## 1. Level 1 — Standalone Tangent-Space Task Decoding (Test Balanced Accuracy)

| Arm | Reference Covariance | Task Classifier | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | `dreyer_sim2` (14) |
|---|---|---|---:|---:|---:|---:|---:|
| `ts_global_pooled` | 1 global $\bar{C}_{\text{global}}$ | 1 pooled LR (8–30 Hz) | 0.5347 | 0.7612 | 0.8133 | **0.6905** | 0.6869 |
| `ts_recenter_routed` | Per-person $\bar{C}_k$ (`fp_logp` mixture) | 1 pooled LR (8–30 Hz) | 0.5903 | 0.7600 | 0.7933 | **0.7060** | 0.6911 |
| `ts_recenter_whitener` | Interpolated $\bar{W}(x)=\sum_k p_k \bar{C}_k^{-1/2}$ | 1 pooled LR (8–30 Hz) | 0.5897 | 0.7600 | 0.7978 | **0.7066** | 0.6905 |
| `ts_recenter_oracle` | Oracle $\bar{C}_{k_{\text{true}}}$ | 1 pooled LR (8–30 Hz) | 0.5903 | 0.7631 | 0.8222 | **0.7125** | 0.6899 |
| `ts_recenter_fb_routed` | 5-band per-person $\bar{C}_{b,k}$ (`fp_logp`) | 1 pooled LR (4–45 Hz) | 0.5596 | 0.7344 | 0.7756 | **0.6800** | 0.6958 |
| `ts_recenter_fb_oracle` | 5-band oracle $\bar{C}_{b,k_{\text{true}}}$ | 1 pooled LR (4–45 Hz) | 0.5590 | 0.7375 | 0.7711 | **0.6804** | 0.6946 |
| `ts_per_person_routed` | Per-person $\bar{C}_k$ (`fp_logp` mixture) | $K$ per-person LRs (8–30 Hz) | 0.6672 | 0.7881 | 0.7556 | **0.7397** | 0.6762 |
| `ts_per_person_oracle` | Oracle $\bar{C}_{k_{\text{true}}}$ | $K$ per-person LRs (8–30 Hz) | 0.6661 | 0.7931 | 0.7333 | **0.7377** | 0.6774 |

*Note on `dreyer_sim2` training pool:* In the main table above, `ts_recenter_*` fits its pooled LR on `pool | calib` (all 73 subjects re-centered by their own $\bar{C}_s$, matching `pooled_eegnet` per FM-2). If fitted on only the 14 evaluation subjects' R1–R3 calibration windows (`d['calib']`), `ts_recenter_routed` scores **0.7018** and `ts_recenter_oracle` scores **0.7024**.

### Level 1 Paired Comparisons on 17 Cross-Day Dev People (5,000 Bootstrap Resamples, Seed 0)

| Comparison | Paired $\Delta$ BA | 95% Bootstrap CI | Better / Worse / Tied | Non-Tied Win 95% CI |
|---|---:|---:|---:|---:|
| `ts_recenter_routed − ts_global_pooled` (**H2a.1: Does re-centering help pooled TS?**) | +0.0155 | [-0.0068, +0.0395] | 9 / 6 / 2 | [0.323, 0.837] |
| `ts_recenter_oracle − ts_global_pooled` (**H2a.1 oracle ceiling**) | +0.0221 | [+0.0015, +0.0442] | 10 / 5 / 2 | [0.384, 0.882] |
| `ts_recenter_whitener − ts_recenter_routed` (1-pass whitener vs tangent mixture) | +0.0006 | [-0.0026, +0.0045] | 2 / 3 / 12 | [0.053, 0.853] |
| `ts_recenter_routed − ts_per_person_routed` (**H2a.2: 1 pooled LR vs $K$ per-person LRs**) | -0.0337 | [-0.0652, +0.0013] | 5 / 12 / 0 | [0.103, 0.560] |
| `ts_recenter_oracle − ts_per_person_oracle` (**H2a.2 under oracle ID**) | -0.0252 | [-0.0606, +0.0146] | 5 / 11 / 1 | [0.110, 0.587] |
| `ts_recenter_fb_routed − ts_recenter_routed` (5-band vs single-band re-centered) | -0.0260 | [-0.0621, +0.0109] | 6 / 11 / 0 | [0.142, 0.617] |
| `ts_recenter_fb_routed − ts_per_person_routed` (5-band pooled vs per-person 8–30 Hz) | -0.0598 | [-0.1055, -0.0134] | 5 / 12 / 0 | [0.103, 0.560] |
| `ts_recenter_oracle − ts_recenter_routed` (routing headroom of re-centered TS) | +0.0066 | [-0.0011, +0.0181] | 4 / 4 / 9 | [0.157, 0.843] |

## 2. Level 2 — End-to-End Track 2 Integration Arms (Seeds 0–2 Mean)

| Arm | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | $\Delta$ vs `row 1` (95% CI) | $\Delta$ vs `row 2` (95% CI) | B/W/T vs `row 2` | `dreyer_sim2` (14) | Adopt? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `row1_pooled_eegnet` | 0.6829 | 0.8156 | 0.7956 | **0.7652** | — | — | — | 0.9304 | ref |
| `row2_current` | 0.7870 | 0.8467 | 0.8511 | **0.8264** | — | — | — | 0.9357 | ref |
| `row3_oracle` | 0.7870 | 0.8610 | 0.8526 | **0.8334** | — | — | — | 0.9363 | ref |
| `E1_fully_pooled_recenter` | 0.7382 | 0.8054 | 0.8437 | **0.7885** | +0.0232 [-0.0010, +0.0460] | -0.0379 [-0.0556, -0.0215] | 3/14/0 | 0.9036 | no |
| `E1_oracle` | 0.7384 | 0.8098 | 0.8615 | **0.7937** | +0.0285 [+0.0062, +0.0499] | -0.0327 [-0.0516, -0.0147] | 4/13/0 | 0.9036 | no |
| `E2_eeg_experts_plus_recenter` | 0.7568 | 0.8194 | 0.8363 | **0.8003** | +0.0350 [+0.0114, +0.0581] | -0.0261 [-0.0397, -0.0145] | 5/12/0 | 0.9167 | no |
| `E2_oracle` | 0.7568 | 0.8296 | 0.8630 | **0.8098** | +0.0445 [+0.0246, +0.0642] | -0.0166 [-0.0323, -0.0018] | 6/11/0 | 0.9149 | no |
| `E3_eeg_experts_plus_shrunk_riemann` | 0.7768 | 0.8356 | 0.8511 | **0.8176** | +0.0524 [+0.0291, +0.0740] | -0.0088 [-0.0174, -0.0017] | 6/11/0 | 0.9292 | no |
| `E3_oracle` | 0.7772 | 0.8494 | 0.8630 | **0.8263** | +0.0611 [+0.0426, +0.0793] | -0.0001 [-0.0119, +0.0130] | 9/8/0 | 0.9298 | no |

## 3. Per-Person Test Balanced Accuracy (17 Dev People)

| Dataset | Person | `ts_global_pooled` | `ts_recenter_routed` | `ts_recenter_oracle` | `ts_per_person_routed` | `row1_pooled_eegnet` | `row2_current` | `E1_fully_pooled_recenter` | `E2_eeg_experts_plus_recenter` | `E3_eeg_experts_plus_shrunk_riemann` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bnci2014` | `1` | 0.7639 | 0.7951 | 0.7951 | 0.8090 | 0.7616 | 0.8924 | 0.8530 | 0.8762 | 0.8877 |
| `bnci2014` | `4` | 0.4410 | 0.5000 | 0.5035 | 0.6806 | 0.6285 | 0.7396 | 0.6736 | 0.6921 | 0.7245 |
| `bnci2014` | `5` | 0.2778 | 0.4236 | 0.4201 | 0.4688 | 0.6516 | 0.7153 | 0.6458 | 0.6771 | 0.7060 |
| `bnci2014` | `6` | 0.3889 | 0.4444 | 0.4444 | 0.5417 | 0.5231 | 0.6157 | 0.5590 | 0.5868 | 0.5995 |
| `bnci2014` | `7` | 0.6424 | 0.6111 | 0.6111 | 0.7153 | 0.7778 | 0.8889 | 0.8507 | 0.8484 | 0.8704 |
| `bnci2014` | `8` | 0.6944 | 0.7674 | 0.7674 | 0.7882 | 0.7546 | 0.8704 | 0.8472 | 0.8600 | 0.8727 |
| `bnci2015` | `1` | 0.9850 | 0.9550 | 0.9550 | 0.9450 | 0.9900 | 0.9717 | 0.9950 | 0.9767 | 0.9783 |
| `bnci2015` | `2` | 0.9450 | 0.9800 | 0.9750 | 0.9750 | 0.9433 | 0.9733 | 0.9783 | 0.9750 | 0.9717 |
| `bnci2015` | `3` | 0.9200 | 0.9200 | 0.9200 | 0.9100 | 0.9333 | 0.9683 | 0.9550 | 0.9700 | 0.9700 |
| `bnci2015` | `4` | 0.8350 | 0.8450 | 0.8450 | 0.9350 | 0.8283 | 0.9167 | 0.8500 | 0.8883 | 0.9117 |
| `bnci2015` | `5` | 0.5000 | 0.5000 | 0.5000 | 0.5500 | 0.6267 | 0.6967 | 0.5667 | 0.5933 | 0.6367 |
| `bnci2015` | `6` | 0.5650 | 0.5250 | 0.5650 | 0.5200 | 0.6917 | 0.6567 | 0.6017 | 0.6217 | 0.6350 |
| `bnci2015` | `7` | 0.8550 | 0.8550 | 0.8500 | 0.9300 | 0.8767 | 0.9300 | 0.8933 | 0.8967 | 0.9100 |
| `bnci2015` | `12` | 0.4850 | 0.5000 | 0.4950 | 0.5400 | 0.6350 | 0.6600 | 0.6033 | 0.6333 | 0.6717 |
| `zhou2016` | `2` | 0.8267 | 0.8733 | 0.8800 | 0.7133 | 0.8711 | 0.8844 | 0.8978 | 0.8889 | 0.8956 |
| `zhou2016` | `3` | 0.8733 | 0.8133 | 0.8133 | 0.8333 | 0.7244 | 0.8311 | 0.8200 | 0.8333 | 0.8356 |
| `zhou2016` | `4` | 0.7400 | 0.6933 | 0.7733 | 0.7200 | 0.7911 | 0.8378 | 0.8133 | 0.7867 | 0.8222 |
