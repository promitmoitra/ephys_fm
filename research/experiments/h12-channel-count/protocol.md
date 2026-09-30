# H12: does channel count explain cross-day fingerprint accuracy?

**Status:** CONFIRMATORY (mechanism / extrapolation to the sealed phase).

## Why

Cross-day `fb6` accuracy: BNCI2014_001 0.974 (22 channels, 9 people), BNCI2015_001 0.923
(13 channels, 12 people), Zhou2016 0.807 (14 channels, 4 people). The sealed phase has 43 EEG
channels. If channel count drives cross-day identifiability, the sealed phase should be at
least as easy as BNCI2014, and the weak held-out numbers are not a warning for it.

## Mechanism → prediction

A covariance fingerprint has C(C + 1)/2 dimensions per band. Fewer channels give fewer, coarser
spatial features, so day-to-day shifts matter relatively more.
**Prediction:** on BNCI2014_001 day 1 → day 2, `fb6` accuracy falls monotonically as channels
are removed, reaching ≈ 0.92 or below at 13 channels (the BNCI2015_001 level), and NLL rises.

## What

BNCI2014_001 (22 channels), day 1 → day 2, `fb6`, LR C = 1. Random channel subsets of size
k ∈ {6, 9, 13, 17, 22}; 10 subsets per k (numpy seed 0; k = 22 is the full set, once).
Report the mean ± SD over subsets of balanced accuracy and NLL.

Same on Dreyer R1 → R3 (27 channels; k ∈ {6, 9, 13, 20, 27}) as a within-session contrast.
Prediction: within a session, the curve is much flatter.

## Decision

Mechanism only. It sets the expectation for sealed-phase fingerprint accuracy and says whether
the held-out results (H3.3) are a risk for 43 channels.
