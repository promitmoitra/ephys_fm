# H3.2: which frequency bands carry the stable identity signal?

**Status:** CONFIRMATORY (mechanism). Parent: H3.1.

## Why

The 6-band filter-bank tangent-space fingerprint (`ts_fb_C1`) reached 0.974 across days on
BNCI, against 0.872 for the broadband covariance. The broadband covariance is dominated by the
highest-power (lowest-frequency) activity. Splitting by band gives the classifier each band's
spatial structure separately. Which bands matter?

## Mechanism → prediction

The individual alpha peak frequency and the alpha/mu spatial topography are known stable
individual traits. Low frequencies (1–4 Hz) are dominated by eye movements and drifts, which
depend on the task and the day. **Predictions (D2, BNCI cross-day):**
1. The best single band is 8–13 Hz, and it beats the broadband covariance (0.872).
2. 1–4 Hz alone is the weakest band cross-day, even if it is strong within a Dreyer session.
3. Leave-one-band-out loses the most when 8–13 Hz is removed.
4. A finer bank (2 Hz steps from 4 to 40 Hz, 18 bands) beats the 6-band bank on D2 NLL by
   resolving individual peak frequencies.

## What

Tangent space + StandardScaler + LR (C = 1), OAS, zero-phase Butterworth (order 4), as in H3.
Benchmarks: R1–R2 → R3 (inner loop), D1_R1toR3, D2_bnci_s1tos2.

| Arm | Bands |
|---|---|
| `single_<band>` | each of the 6 bands alone |
| `drop_<band>` | the 6-band bank minus one band |
| `fine18` | 4–6, 6–8, …, 38–40 Hz |
| `fb_plus_broad` | the 6 bands + broadband |

## Decision

Mechanism only (no candidate selection unless `fine18` wins on D2 NLL and does not lose
elsewhere; then it joins the confirmation candidates).
