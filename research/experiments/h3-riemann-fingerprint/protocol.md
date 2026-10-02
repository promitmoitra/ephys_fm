# H3: Riemannian fingerprint (covariance → tangent space → logistic regression)

**Status:** CONFIRMATORY

## Mechanism → prediction

Spatial covariance reflects head geometry, electrode placement and cap fit, all fixed within a
recording session. Dreyer has **one session of six runs per person**, so R1–R2 → R3 (and R1–R3 →
R4–R6) is a within-session, cross-run problem. On BNCI, a tangent-space fingerprint reached 0.999
within a session (leave-one-run-out) and 0.872 across days (EEGNet 0.909 across days). The kit's
per-recording robust scaling removes per-channel scale per run, but not the correlation
structure. **Prediction:** the Riemannian fingerprint reaches ≥ 0.90 balanced accuracy on R3
(vs H0's EEGNet), with lower NLL because multinomial logistic regression on a few hundred
features is less over-confident than a 150-epoch CNN.

## What (all deterministic; no seeds)

Train R1–R2 (1,680 windows), test R3 (840), the 21 evaluation people.

| Config | Features | Classifier |
|---|---|---|
| `ts_broad_C1` | OAS covariance of the full 0–4 s kit window (27 × 27) → tangent space at the Riemannian mean of the training windows (378 features) | StandardScaler → multinomial LogisticRegression, C = 1 |
| `ts_broad_C0.1` | same | C = 0.1 |
| `ts_fb_C1` | 6 bands (1–4, 4–8, 8–13, 13–20, 20–30, 30–45 Hz; 4th-order Butterworth, zero-phase) → per-band OAS covariance → tangent space → concatenated (2,268 features) | C = 1 |
| `ts_fb_C0.1` | same | C = 0.1 |

Zero-phase filtering within a 4 s window is legitimate at inference (the whole window is given).

## Metrics and decision

The locked metrics of H0 (balanced accuracy and NLL on R3; secondary p_true, ECE, top-3, worst
person). The best config by R3 NLL is carried forward to ensembling (H6) and confirmation.
Supported if the best config beats H0's 3-seed mean on both balanced accuracy and NLL.
