# Protocol 00: prediction banks and the locked evaluation (loop B)

Locked 2026-09-30, before any bank exists.

## Question for the whole loop

Given, per evaluation person, a fine-tuned EEGNet expert and per-person classical experts
(Riemannian tangent space `ts`, CSP + LDA `csp`), what combination rule, learned from
calibration data only, gives the best soft-routed mixture on the hidden runs? Reference on
R4–R6 (seed 0): EEGNet-only soft mixture **0.901**, EEGNet + ts equal average **0.909**.

## Banks (`research/src/bank.py`, artifacts in `outputs/t2-expert-weights/`)

- **Dev bank** (`bank_dev_seed0.npz`): pooled EEGNet on the training pool + calibration
  R1–R2 (100 epochs, epoch picked on the kit's val split, 4 threads, seed 0); control and 21
  per-person fine-tunes on R1–R2 (lr 1e-4, batch 32, 50 epochs, last epoch); `ts` and `csp`
  per person on R1–R2 (8–30 Hz, 0.5–4 s). All predict **R3** (840 windows, 40 per person).
  Also `fp_biased`: the packaged fingerprint's p(person) on R3 (optimistic: it picked its
  epoch on R3; exploratory only).
- **Test bank** (`bank_test_packaged.npz`): EEGNet experts and fingerprint = the packaged
  seed-0 mixture (trained on R1–R3, 8 threads), so its EEGNet-only soft mixture reproduces
  0.901; `ts` and `csp` fitted on R1–R3. All predict **R4–R6** (2,520 windows, 120 per person).
  *Deviation from the handoff:* the test bank reuses the packaged EEGNet experts instead of
  retraining them at 4 threads, so the confirmation baseline is exactly the shipped one.
  Known mismatch: dev experts see 2 calibration runs, test experts 3, so the EEGNet expert is
  relatively stronger at test time than on dev.

Sanity checks before use: every per-expert oracle accuracy on R3 above chance; EEGNet expert
oracle on R3 in the neighbourhood of its R4–R6 value (0.908); the test bank reproduces
0.908 oracle / 0.901 soft / ts 0.688 oracle / eegnet+ts soft 0.909.

## Inner-loop metric (decisions)

On the dev bank, under the **true ID** (identity is known during calibration):

- **Global combiners** (a few parameters shared by all people): leave-one-person-out CV,
  fit on 20 people's R3 predictions, score the held-out person.
- **Per-person combiners**: within-person 2-fold CV on R3 (20 / 20 windows, stratified,
  fixed split seed), averaged over 5 repetitions of the split.

Primary metric: **balanced accuracy** of the pooled out-of-fold predictions (840 windows).
Secondary: mean log-loss (NLL), and per-person paired comparison against the equal-average
EEGNet+ts rule (people better / worse). With 840 windows, one window = 0.0012; differences
under ~0.01 are within noise. Report the dev-bank equal-average and EEGNet-only numbers as
the in-loop baselines.

## Confirmation (outer-loop checkpoints only, pre-registered candidates)

Fit the chosen combiner on the full dev bank, apply to the test bank soft-routed with the
packaged fingerprint (baseline 0.901, equal average 0.909); also report oracle-ID. Then with
loop A's best fingerprint when available. **R4–R6 are never used to choose among
candidates**; each checkpoint lists its candidates in a committed protocol before running.
