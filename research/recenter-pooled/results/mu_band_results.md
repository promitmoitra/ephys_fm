# Protocol 02 Results: Per-Person $\mu$-Peak Band Centering for `RiemannExperts` (Handoff §2c)

Generated automatically by `research/recenter-pooled/src/evaluate_mu_band.py`.
Evaluation uses the 17 cross-day dev people (`bnci2014` 6, `bnci2015` 8, `zhou2016` 3) and
14 fresh `dreyer_sim2` people. The 8 locked holdout people were never loaded.

## 1. Estimated Calibration $\mu$-Peak Frequencies $f_{\mu,k}$ (Hz)

| Dataset | People & Calibration $f_{\mu,k}$ (Hz) | Shifted Passbands $[\text{lo}_k, \text{hi}_k]$ (Hz) |
|---|---|---|
| `bnci2014` | `1`: 11.77, `4`: 11.55, `5`: 7.62, `6`: 9.59, `7`: 10.51, `8`: 10.19 | `1`: [9.8, 31.8], `4`: [9.6, 31.6], `5`: [5.6, 27.6], `6`: [7.6, 29.6], `7`: [8.5, 30.5], `8`: [8.2, 30.2] |
| `bnci2015` | `1`: 11.21, `2`: 12.14, `3`: 13.22, `4`: 11.95, `5`: 9.54, `6`: 11.42, `7`: 10.03, `12`: 9.30 | `1`: [9.2, 31.2], `2`: [10.1, 32.1], `3`: [11.0, 33.2], `4`: [9.9, 31.9], `5`: [7.5, 29.5], `6`: [9.4, 31.4], `7`: [8.0, 30.0], `12`: [7.3, 29.3] |
| `zhou2016` | `2`: 12.22, `3`: 11.95, `4`: 11.79 | `2`: [10.2, 32.2], `3`: [9.9, 31.9], `4`: [9.8, 31.8] |
| `dreyer_sim2` | `3`: 9.90, `8`: 10.62, `10`: 11.87, `14`: 7.00, `19`: 10.97, `20`: 11.93, `40`: 14.00, `45`: 7.72, `46`: 10.00, `47`: 11.18, `50`: 10.00, `55`: 9.86, `58`: 9.72, `82`: 8.74 | `3`: [7.9, 29.9], `8`: [8.6, 30.6], `10`: [9.9, 31.9], `14`: [5.5, 27.0], `19`: [9.0, 31.0], `20`: [9.9, 31.9], `40`: [11.0, 34.0], `45`: [5.7, 27.7], `46`: [8.0, 30.0], `47`: [9.2, 31.2], `50`: [8.0, 30.0], `55`: [7.9, 29.9], `58`: [7.7, 29.7], `82`: [6.7, 28.7] |

## 2. Level 1 — Standalone Per-Person `RiemannExperts` Task Decoding (Test Balanced Accuracy)

| Arm | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | `dreyer_sim2` (14) |
|---|---:|---:|---:|---:|---:|
| `fixed_8_30_static` | 0.6672 | 0.7881 | 0.7556 | **0.7397** | 0.6762 |
| `mu_shift_wide_static` | 0.6603 | 0.7906 | 0.7311 | **0.7341** | 0.6708 |
| `mu_lower_edge_static` | 0.6580 | 0.7925 | 0.7356 | **0.7350** | 0.6714 |
| `fixed_2band_static` | 0.6962 | 0.8094 | 0.7467 | **0.7584** | 0.6976 |
| `mu_beta_2band_static` | 0.6944 | 0.7994 | 0.8133 | **0.7648** | 0.6869 |
| `fixed_8_30_ema_shuf` | 0.6904 | 0.8431 | 0.8556 | **0.7914** | 0.6845 |
| `mu_shift_wide_ema_shuf` | 0.6892 | 0.8487 | 0.8622 | **0.7948** | 0.6804 |
| `mu_lower_edge_ema_shuf` | 0.6916 | 0.8481 | 0.8578 | **0.7946** | 0.6786 |
| `fixed_2band_ema_shuf` | 0.7135 | 0.8606 | 0.8044 | **0.7988** | 0.7125 |
| `mu_beta_2band_ema_shuf` | 0.7135 | 0.8625 | 0.8756 | **0.8122** | 0.7065 |
| `mu_shift_wide_sess_spd` | 0.7031 | 0.8594 | 0.8667 | **0.8055** | 0.6857 |

### Level 1 Paired Comparisons vs Fixed-Band Counterparts (17 Cross-Day Dev People)

| Comparison | Paired $\Delta$ BA | 95% Bootstrap CI | Better / Worse / Tied |
|---|---:|---:|---:|
| `mu_shift_wide_static − fixed_8_30_static` (1-band shift) | -0.0056 | [-0.0208, +0.0067] | 7 / 5 / 5 |
| `mu_lower_edge_static − fixed_8_30_static` (1-band lower edge) | -0.0047 | [-0.0193, +0.0072] | 7 / 5 / 5 |
| `fixed_2band_static − fixed_8_30_static` (fixed `[8,13]+[13,30]` vs `[8,30]`) | +0.0186 | [-0.0028, +0.0409] | 11 / 5 / 1 |
| `mu_beta_2band_static − fixed_2band_static` (**personal $\mu$ vs fixed 2-band**) | +0.0064 | [-0.0153, +0.0293] | 9 / 6 / 2 |
| `mu_shift_wide_ema_shuf − fixed_8_30_ema_shuf` (1-band shift + EMA) | +0.0034 | [-0.0039, +0.0102] | 10 / 4 / 3 |
| `mu_lower_edge_ema_shuf − fixed_8_30_ema_shuf` (1-band lower + EMA) | +0.0032 | [-0.0022, +0.0086] | 8 / 4 / 5 |
| `fixed_2band_ema_shuf − fixed_8_30_ema_shuf` (fixed `[8,13]+[13,30]` + EMA) | +0.0074 | [-0.0109, +0.0240] | 10 / 4 / 3 |
| `mu_beta_2band_ema_shuf − fixed_2band_ema_shuf` (**personal $\mu$ vs fixed 2-band + EMA**) | +0.0134 | [+0.0005, +0.0305] | 10 / 5 / 2 |

## 3. Level 2 — End-to-End Track 2 Integration (`C3` Mixture, Seeds 0–2 Mean)

| Arm | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | $\Delta$ vs Fixed-Band Ref (95% CI) | $\Delta$ vs `row 2` (95% CI) | `dreyer_sim2` (14) | Adopt $\mu$-Band? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `fixed_8_30_static` | 0.7870 | 0.8467 | 0.8511 | **0.8264** | ref | — | 0.9357 | ref |
| `mu_shift_wide_static` | 0.7843 | 0.8494 | 0.8459 | **0.8258** | -0.0006 [-0.0026, +0.0014] | -0.0006 [-0.0026, +0.0014] | 0.9351 | no |
| `mu_lower_edge_static` | 0.7859 | 0.8500 | 0.8504 | **0.8274** | +0.0010 [-0.0007, +0.0028] | +0.0010 [-0.0007, +0.0028] | 0.9351 | no |
| `fixed_2band_static` | 0.8054 | 0.8558 | 0.8348 | **0.8343** | +0.0079 [-0.0029, +0.0164] | +0.0079 [-0.0029, +0.0164] | 0.9351 | no |
| `mu_beta_2band_static` | 0.8100 | 0.8538 | 0.8637 | **0.8401** | +0.0058 [-0.0019, +0.0161] | +0.0137 [+0.0086, +0.0190] | 0.9327 | no |
| `fixed_8_30_ema_shuf` | 0.7882 | 0.8642 | 0.8681 | **0.8381** | ref | — | 0.9369 | ref |
| `mu_shift_wide_ema_shuf` | 0.7907 | 0.8640 | 0.8681 | **0.8388** | +0.0008 [-0.0019, +0.0035] | +0.0124 [+0.0041, +0.0227] | 0.9369 | no |
| `mu_lower_edge_ema_shuf` | 0.7905 | 0.8642 | 0.8696 | **0.8391** | +0.0011 [-0.0010, +0.0034] | +0.0127 [+0.0044, +0.0229] | 0.9363 | no |
| `fixed_2band_ema_shuf` | 0.8059 | 0.8723 | 0.8467 | **0.8444** | +0.0063 [-0.0042, +0.0138] | +0.0179 [+0.0072, +0.0310] | 0.9357 | no |
| `mu_beta_2band_ema_shuf` | 0.8113 | 0.8702 | 0.8859 | **0.8522** | +0.0079 [-0.0003, +0.0190] | +0.0258 [+0.0169, +0.0356] | 0.9369 | no |
| `mu_shift_wide_sess_spd` | 0.7919 | 0.8679 | 0.8733 | **0.8420** | +0.0156 [+0.0065, +0.0271] | +0.0156 [+0.0065, +0.0271] | 0.9381 | **YES** |
