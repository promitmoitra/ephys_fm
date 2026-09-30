# Protocol 04: is the classical-expert gain robust to the EEGNet training seed?

Locked 2026-09-30, before any seed-1/2 model is trained. Every EEGNet result so far is seed 0.
The combination gain (+0.012 to +0.014 on R4–R6) could depend on how good that seed's
EEGNet experts happen to be (dev folds showed the gain is larger when EEGNet is weaker).

**Setup.** For seeds 1 and 2: pooled EEGNet on the training pool + R1–R3 (100 epochs, epoch on
val, 2 threads), per-person fine-tunes on R1–R3 (lr 1e-4, 50 epochs, last epoch), predicting
R4–R6 (`bank.py testseed`). Classical experts are deterministic and unchanged. The
combiners are the checkpoint-2 fits (on the seed-0 cross-fitted dev bank), applied
**unchanged**. This is a confirmation-only experiment: nothing is selected on it.

**Reported per seed:** oracle-ID balanced accuracy and NLL for R0 (EEGNet only), R1, C1, C3;
Δ vs R0 with the person-bootstrap CI; and the mean over seeds 0–2.

**Prediction:** C1 and C3 beat R0 on both new seeds, by +0.005 to +0.015; NLL improves on
both. If the gain vanishes or flips on a seed, the adoption of C3 is downgraded to "seed-
dependent" and the loop returns to the dev bank with multi-seed EEGNet experts.
