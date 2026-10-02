# H0: EEGNet fingerprint baseline under the locked protocol

**Status:** CONFIRMATORY (baseline re-measurement)

## Why

The seed-0 fingerprint in `outputs/track2_dreyer_sim/submission/mixture.pt` picked its epoch on
R3 (best of 150), so its R3 score (0.857) is optimistically biased and cannot be a baseline.
Every later hypothesis is compared against this re-run.

## What

- Data: the 21 evaluation people (Dreyer subjects 61–81), kit windows (27 ch × 480 samples).
- Train: runs R1–R2 (1,680 windows, 80 per person). Test: R3 (840 windows, 40 per person).
- Model: braindecode `EEGNet` defaults, 21 outputs. AdamW lr 1e-3, weight decay 1e-3,
  batch 64, **150 epochs, last epoch kept** (the shipped recipe's budget, no selection).
- Seeds 0, 1, 2; `torch.set_num_threads(4)`.
- Logged every 25 epochs on R3 for the learning curve only (diagnostic; no selection).

## Metrics (locked for the whole loop)

Primary, on R3: **21-way balanced accuracy** and **NLL** of the true person.
Secondary: mean p(true person) (the mass soft routing puts on the right expert), ECE (15 bins),
top-3 accuracy, worst-person recall.

## Prediction

Balanced accuracy below the biased 0.857, around 0.80–0.85; NLL well above what the accuracy
implies if the last-epoch network is over-confident (EEGNet trained 150 epochs on 80 windows per
class typically is), i.e. ECE > 0.05.

## Decision rule

A later hypothesis "beats the baseline" if its 3-seed mean improves NLL **and** does not lower
balanced accuracy by more than one seed-SD of H0 (or, for deterministic models, improves both).
