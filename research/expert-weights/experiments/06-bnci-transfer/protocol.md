# Protocol 06: does the combination rule transfer zero-shot to BNCI 2014-001?

Locked 2026-09-30, before any BNCI classical expert or combination is computed. External
check; no decisions about the Dreyer combiner are taken from it.

**Why.** The sealed phase has different data (cross-day, 3 classes). A shipped combiner
either reuses coefficients fitted elsewhere or needs an expensive cross-fitted bank on the
new calibration data. This tests the cheap option on a dataset with 4 classes, 9 people and
a day gap between calibration and test.

**Setting** (as `experiments/fingerprint_tangermann/finetune_routing_eegnet.py`): train on
session 1 runs 0–4, validation run 5 (pooled epoch choice), test on session 2. Pooled EEGNet
150 epochs, per-person whole-network fine-tune (lr 1e-4, 50 epochs, last epoch). 5 seeds
(0–4), 2 threads. Per-person ts and ts_C0.1 experts (8–30 Hz, 0.5–4 s, multinomial LR) on
the same training runs. Reliability = mean balanced accuracy of a one-run ts_C0.1 model on
another training run (20 ordered pairs of runs 0–4), mapped to the 2-class scale:
rel' = 0.5 + 0.5·(rel − 1/4)/(1 − 1/4), so chance → 0.5 and perfect → 1.

**Combiners, coefficients transferred from Dreyer unchanged** (fitted on the cross-fitted
dev bank; class bias dropped, it was ≈ 0):
- R0: EEGNet expert only. R1: plain average with ts.
- C1: z = 0.826·log p_eeg + 0.420·log p_ts.
- C3: z = 0.810·log p_eeg + (0.960 + 1.701·(rel' − 0.5))·log p_ts_C0.1.

All under oracle ID (loop A shows routing ≈ oracle).

**Metric:** session-2 balanced accuracy (4 classes, chance 0.25) and NLL; Δ vs R0 per seed,
paired over seeds (mean ± SD, seeds better).

**Predictions.** On BNCI the classical expert is relatively stronger (its ts is a standard
decoder for this dataset), so: C1 and C3 > R0 on ≥ 4/5 seeds; NLL lower; R1 more volatile
(it can go either way). A failure (C1/C3 ≤ R0 on ≥ 3 seeds) means the Dreyer coefficients
don't transfer and a shipped combiner must be refitted per dataset.

*Exploratory, clearly labelled:* H4 (log-linear) fitted leave-one-person-out on BNCI's own
session 2 (8 people's test windows → 9th), to show how far the transferred weights are from
BNCI-optimal ones.
