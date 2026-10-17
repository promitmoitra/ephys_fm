# Handoff: what NeuralPrint's results mean for ephys_fm

Written 2026-10-17 from NeuralPrint at commit `a87b47f` (`/home/promit/Documents/neuralprint`).
NeuralPrint is the sister project on longitudinal EEG identity (biometrics). Its loop A fingerprint
is already in ephys_fm (Track 2 `FBFingerprint`). This note lists what it has found since then,
and which of its failures bear on ephys_fm's plans.

**Before acting on any item:**

- Follow this repository's `CLAUDE.md`. Work in a worktree, not the main checkout, and give each
  item its own `exp/<topic>` branch. This file is untracked in the main checkout. Move it to
  `docs/handoff/` on a docs branch if it should be kept.
- Treat NeuralPrint as read-only: read its code and copy what you need.
- NeuralPrint's verdicts come from `research/check_verdicts.py` and only check arithmetic against
  pre-registered criteria. Read the cited `analysis.md` before relying on a number.
- Its findings summary is `openwiki/research/findings.md`, and its methodological rules (FM-1 to
  FM-11) are in `openwiki/protocols/failure-modes.md`. Both apply here: test "X beats Y" on the
  paired per-person difference, give the baseline the same calibration data, never select on the
  test set, and commit pass bars before reading related results.
- NeuralPrint numbers are N-way identification at one month (60- to 159-way). The Track 2 sealed
  phase is 10-way with several calibration days, so effect sizes will not carry over directly.

Paths below without a prefix are relative to the NeuralPrint root.

## 1. Port to the Track 2 fingerprint (strongest evidence)

The sealed phase is across days, and the fingerprint is weakest there (0.79–0.94 with one
calibration day; loop A). Evaluate both items on the identity-integration cross-day harness
(`docs/superpowers/specs/2026-10-05-identity-integration-design.md`: 17 dev people; never load
the 8 holdout people).

### 1a. Complex Hermitian template

- **What:** build each band's covariance from the Hilbert analytic signal. The Hermitian matrix's
  imaginary part keeps lagged co-variation, which the real covariance drops. Map it to the
  tangent space with a complex Fréchet mean and vectorize the real and imaginary parts.
- **Evidence:** at one month, against the shipped real template, PainLab 159-way 74.8% → 83.9%
  and ds004148 60-way 46.7% → 59.8%. Both parts of the gain replicate on both datasets: the
  Hermitian estimator gives +4.1 / +5.6 points and the imaginary part +5.4 / +8.1. On ds004148
  open-set EER goes 15.4% → 12.0%. Removing frontal and temporal channels and keeping only
  4–20 Hz still leaves +8.8 points. The gain sits in α, with none in δ. This rules out the easy
  eye and muscle explanations but does not show the signal is neural.
- **Code:** `experiments/h-complex-hermitian-tangent/code/run_experiment.py`
  (`compute_complex_frechet_mean`, `oas_covariance`, `vectorize_complex_tangent`), with tests in
  `test_complex_math.py`. The replication is in `experiments/h-complex-replication/`. Analyses:
  `analysis.md` in each folder.
- **Export:** it should stay pure torch. A Hilbert transform on a fixed-length window is linear, so
  the band filter and the Hilbert transform fold into one complex T × T matrix per band, and
  `torch.linalg.eigh` accepts complex Hermitian input. Check float32 against the reference, as
  loop A did (it matched sklearn to 4e-6). The fingerprint's size doubles, and README already
  warns about ≈ 100 MB at 500 Hz.
- **Test:** current fingerprint vs complex fingerprint, test-session ID accuracy and NLL, then
  end-to-end Track 2 balanced accuracy with soft routing. The routing gain is bounded by the
  fingerprint's errors, so it only matters where ID is below about 0.95.

### 1b. Drop the delta band (5 bands, 4–45 Hz)

- **What:** the shipped fingerprint uses 6 bands including 1–4 Hz. Loop A itself found delta is
  day-specific (0.989 within a session, 0.710 across days).
- **Evidence:** a pre-registered held-out check (PainLab, 16-way, about 30 days apart):
  93.75% → 95.50%, open-set AUC 0.984 → 0.989 (ledger `H_DELTA_HELDOUT` supported).
  **Counter-result:** on Zhou2016 trained on a single day it fell (0.790 → 0.735). With
  multi-day calibration, which is the sealed-phase regime, this has not been tested.
- **Code:** `experiments/h-delta-band-heldout-check/`. The band banks are `model.FP_BANDS`
  (5-band) and `model.FB6_BANDS` in NeuralPrint.
- **Test:** cheap; run it together with 1a as a 2 × 2 (real/complex × 5/6 bands).

### Not worth porting

On top of the complex template, the following add at most a point or two, and usually nothing:

- a 0.5-Hz spectrum;
- twelve half-bands (+1.6, PainLab only, below its bar);
- α peak features;
- spectral parameters and aperiodic dynamics;
- wavelet envelope covariance;
- phase-amplitude coupling;
- delay-embedding dynamics;
- per-channel gain normalisation (+2.9, p = 0.087).

Sources: NeuralPrint findings §5, plus `experiments/h-gain-invariant/`.

## 2. Identity results read in reverse, for task decoding

For identity, features that are stable within a person are the signal and features that change
with state or day are the nuisance. For task decoding the roles swap. Everything in this section
is a lead with no task-decoding test behind it.

### 2a. Fingerprint-routed re-centering (Track 2)

- **Idea:** `research/literature/survey.md` (ephys_fm) dismissed Riemannian re-centering because
  `predict` gets no IDs. The fingerprint can supply that missing ID. Route the *reference
  covariance* rather than the expert: re-centre each window at the calibration mean of its
  predicted person (or a p(person | x)-weighted mean), then use one pooled task model on the
  re-centred tangent space. This is a cheaper relative of Option B (per-person adapters).
- **Evidence:** the evidence is weak. NeuralPrint's data-atlas variance partition
  (`experiments/data-atlas-profiling/results/variance_partition_report.md`, **uncommitted work of
  another session, unreviewed**) shows a 0% subject share after re-centering. That holds by
  construction once each subject's mean is removed, so it is not evidence that re-centering
  helps. The session share (6–26%) stays. The support for the idea is the transfer-learning
  literature (Zanini et al. 2018; Rodrigues et al. 2019 RPA).
- **Against:** in `H_REENROLL` (identity, M3CV), plain retraining on the calibration data beat
  re-centering and Procrustes at every duration. Routing errors also re-centre a window at the
  wrong mean.
- **Test:** against the current design *and* against a pooled model re-centred with oracle
  identity (the ceiling), on the same harness. FM-2 applies: every arm sees the same calibration
  data.

### 2b. State features for Track 3 (sleep onset)

- **Idea:** some features failed as identity traits because they change with state, which makes
  them candidates for sleep onset. Spectral-parameter ICC(2,1) on ds004148, 90 minutes vs one
  month:
  - aperiodic offset: 0.93 within a visit, 0.50 across a month;
  - aperiodic exponent: 0.71, then 0.32.
  
  Delta is day-specific. The aperiodic exponent and α slowing are also standard drowsiness and
  sleep-depth markers in the literature.
- **Code:** `experiments/h-specparam/code/run_experiment.py`. It has `fit_params` / `fit_many`
  (specparam fits), `time_resolved` (1-minute blocks) and `icc_with_ci`.
- **Caveat:** NeuralPrint's low ICC across visits may come from electrode contact (cap
  re-fitting), not from the person's state. Within one night's recording that confound is
  mostly absent.

### 2c. Per-person μ band for the Riemannian expert (speculative)

- The α peak frequency is the most stable spectral trait (ICC 0.87 across ds004148's cap change).
  The Riemannian expert uses a fixed 8–30 Hz band. Centring the μ band on each person's peak,
  estimated from calibration data, could sharpen it.
- Lowest priority here. Loop B found filter-bank Riemannian experts did not help.

## 3. NeuralPrint failures that bear on ephys_fm plans

| ephys_fm plan or habit | NeuralPrint result | Source |
|---|---|---|
| Identity-integration **Option C** (identity as a network input) | Static subject tokens fed into attention cut cross-session decoding from 58.6% to 37.7% (committed JSON). The model memorised session noise (FM-5). The held-out-subject figures (bottleneck 50.55%, no gain) are in `analysis.md` only, with no committed runner or JSON. Bottleneck or drop out the identity input from the start, and check cross-session transfer early. | `experiments/h-track4a-fm-routing/`, FM-5 |
| Mixture or routing designs | Foundation-model soft-MoE scored 21.8 points below one monolithic decoder. Cluster-MoE gained −2.4 to +0.8 points. Both trained experts on partitions of the data. This agrees with ephys_fm's own result (heads from scratch −0.030, fine-tunes of a pooled model +0.045). Specialise from a pooled model; never partition. | findings §8, `H_TRACK_4A.2`, `H_TOPOLOGY_*` |
| Atlas dynamics features (RQA, entropy vs IAAFT surrogates) | Delay-embedding recurrence and entropy scored *below* their Gaussian surrogates (−3.2 points), and envelope correlation −1.5. Their identity was the spectrum in disguise. Expect the same; keep the surrogate test. | `experiments/h-dynamics-surrogate-check/code/run_check.py` |
| Any trick that uses window order | The kit's test loaders are not shuffled (batches of 64, in order). Smoothing over a batch, sticky priors, or re-centring per batch will look good on block-ordered warm-up data. NeuralPrint's Viterbi gain came entirely from the block design (−0.2 points once trials were interleaved). The sealed phase's ordering is unknown (FM-10). | `experiments/h-block-viterbi-task-control/` |
| Option B adapters, alignment, any calibration-data method | It must beat plain retraining on the same calibration data. NeuralPrint's drift projection lost to that baseline by 2.3 points (FM-2). | `H_INVARIANT_*`, `H_REENROLL_2` |
| Soft routing, which depends on NLL as well as accuracy | Unsupervised session centering improved NLL (0.631 → 0.481) but lowered ID accuracy (83.2% → 78.2%) on Zhou2016. Report both. | findings §8, `H_LOOP_C` |

## 4. Out of scope

- **Tracks 1 (EEG-to-image) and 4 (EMG-to-pose):** NeuralPrint has nothing directly
  transferable. It never worked on stimulus decoding or EMG.
- **Multi-day enrollment** (`H11`: pooling calibration days beats one day, and day diversity
  matters more than data amount) is already in ephys_fm loop A, which fits the fingerprint on all
  calibration sessions pooled.

## Suggested order

1. 1a + 1b as one 2 × 2 on the cross-day harness. This is cheap and has the strongest evidence.
2. 2a, as a pooled-model alternative to Options B and C. Pre-register it against the current
   design and an oracle-identity ceiling.
3. 2b, when Track 3 work starts.
4. Fold §3 into the identity-integration spec before Option C's loop begins.
