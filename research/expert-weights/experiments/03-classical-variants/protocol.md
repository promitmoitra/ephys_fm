# Protocol 03: a stronger classical expert (H9) and per-person reliability weights (H8)

Locked 2026-09-30 before any variant was scored. Uses the cross-fitted bank of protocol 02
(2,520 windows; each calibration run held out in turn). Code: `research/expert-weights/src/classical.py`.

## H9: classical expert variants

Mechanism: the ts expert (8–30 Hz, 0.5–4 s) is at chance for ~9 of 21 people, and it is
over-confident. Loop A found that per-band covariances carry more person-specific structure
than one broad band. A filter bank with a regularised classifier should raise the classical
expert's accuracy, and a stronger, still-different expert should raise the stacked
combination. Including the cue period (0–4 s) or low frequencies should make it stronger but
more EEGNet-like (the cue-locked response), so its combination gain may shrink.

| Variant | Bands (Hz) | Window (s) | LR C |
|---|---|---|---|
| ts | 8–30 | 0.5–4 | 1 (= bank) |
| ts_C0.1 | 8–30 | 0.5–4 | 0.1 |
| ts_w0 | 8–30 | 0–4 | 1 |
| ts_broad | 1–40 | 0–4 | 1 |
| ts_fb5 | 4–8, 8–13, 13–20, 20–30, 30–40 | 0.5–4 | 0.1, standardised |
| ts_fb5_w0 | same | 0–4 | 0.1, standardised |
| ts_mu_beta | 8–13, 13–30 | 0.5–4 | 0.1, standardised |
| ts_low | 1–4 | 0–4 | 1 |

Metrics per variant on the cross-fitted bank (true ID): the expert alone (balanced accuracy,
NLL), and **combined with EEGNet by H3** (log-linear + bias, LOPO; primary) and by R1 (equal
average). **Decision:** the variant with the best H3-combined balanced accuracy becomes the
classical expert (ties within 0.003 → lower NLL, then fewer bands).

Predictions: ts_fb5 alone > ts alone; ts_broad / ts_low alone strongest (they see the cue
response and eye movements) but add least on top of EEGNet; ts_C0.1 is better calibrated
than ts at equal accuracy.

## H8: per-person reliability weights

Mechanism: the classical expert's value is person-specific. A reliability score estimated on
calibration data alone can steer its weight. For a held-out run r, rel_k = mean accuracy of
person k's classical expert trained on one of the two other calibration runs and tested on
the other (both directions, 40 windows each). At test time, the same score comes from the 6
ordered run pairs of R1–R3. Combiner H8: log-linear + bias, with the classical weight
w_c(k) = w0 + w1 · (rel_k − 0.5) (3 global parameters + bias; LOPO). Compared with H3 on the
same expert. Prediction: H8 > H3 by ≥ 0.005 with more people better than worse; the learned
w1 > 0.
