# Protocol 02: cross-fitted dev bank, and protocol 01's sweep re-run on it

Locked 2026-09-30, before the new folds exist. Motivation: checkpoint 1 showed dev (R3, 840
windows) and test disagree on the sign of the ts effect; the dev bank is below the resolution
needed (paired SE ≈ 0.01).

## Bank

For each held-out calibration run r ∈ {R1, R2, R3}: pooled EEGNet on the training pool + the
other two calibration runs (100 epochs, epoch on val, 4 threads, seed 0), control and
per-person fine-tunes on those two runs, ts / csp on those two runs, all predicting run r.
Fold R3 is the existing dev bank. Concatenated: `bank_xfit_seed0.npz`, 2,520 windows, 120 per
person, with a `fold` key. (Training on later runs to predict earlier ones is fine here: the
bank measures how experts trained on two runs generalize to a third.)

## Evaluation (locked)

Same combiners as protocol 01 (`01-combiners/code/run.py` RUNS). Global combiners:
leave-one-person-out on all 2,520 windows. Per-person combiners: leave-one-run-out within
person (fit on 80 windows of two folds, predict the third). Primary metric: balanced
accuracy; secondary NLL, paired per-person comparison vs R0 (EEGNet only). Also per fold,
to test whether R3 is special.

## Predictions

1. Per-fold, R1 − R0 (equal average minus EEGNet) is negative on the R3 fold (known,
   −0.018) and around zero or positive on R1 and R2 folds; pooled over folds, R1 − R0 lies
   between the dev (−0.018) and test (+0.014) values. If all three folds are negative, the
   test-set gain came from the third training run (explanation ii in checkpoint 1) or chance.
2. H3 (log-linear + bias) ≥ R0 in accuracy and has the lowest NLL of the global combiners,
   as on both dev and test.
3. With 80 fitting windows per person, per-person weights (H2) are no longer worse than
   global ones.

## Decision rule for checkpoint 2

Candidates: the best global combiner by xfit balanced accuracy, and by xfit NLL (fewer
parameters on ties within 0.002), plus R0/R1/H3 as references (already confirmed). A new
combiner is adopted over H3 only if it beats H3 on the xfit bank by ≥ 0.005 balanced accuracy
with more people better than worse, or matches its accuracy (±0.002) with lower NLL.
