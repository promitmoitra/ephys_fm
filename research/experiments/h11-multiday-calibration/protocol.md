# H11: does calibrating on more days make the fingerprint more robust across days?

**Status:** CONFIRMATORY.

## Why

The sealed phase probably gives several labeled calibration sessions per evaluation person
(sessions 1–3), recorded on different days, and hides later sessions. Every cross-day
benchmark so far trained on one day. With two or more training days, the classifier sees
within-person, between-day variation and can learn to ignore it.

## Mechanism → prediction

Logistic regression on tangent-space features weights the directions that separate people
relative to within-person spread. If the training data contain only one day, day-specific
directions (electrode impedance, cap position, arousal) look like identity signal. With two
days, their within-person spread grows and they get down-weighted.
**Prediction:** training on days 1 + 2 lowers the NLL on day 3 compared with day 1 alone,
**and** compared with day 2 alone (the recency control: day 2 is closer in time to day 3).

## What

Tangent space + StandardScaler + LR (C = 1), banks `fb6` and `fine18` (both fixed before H3.3's
held-out results).

| Dataset | People | Test | Arms (training data) |
|---|---|---|---|
| Zhou2016 | 4 (4-way) | session 3 | S1; S2; S1 + S2 |
| BNCI2015_001 | 12 (12-way) | session 3 of the 4 people who have one (800 windows) | S1 (all 12); S2 (all 12); S1 + S2 (all 12) |

S1 + S2 has twice the data. The recency control (S2 alone) and the S1 arm bracket that. A
data-matched S1 + S2 arm (half of each day, same total as S1) isolates the effect of day
diversity from the effect of data amount.

## Decision

Supported if S1 + S2 beats both S1 and S2 on NLL for both banks on both datasets, and the
data-matched arm also beats S1 and S2. Then the recommendation for the sealed phase is to fit
the fingerprint on all calibration days pooled. (This is the default anyway; the result sets the
expectation for how much robustness to expect.)
