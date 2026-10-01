# Track 2 (BCI decoding): fingerprint-routed mixture

`predict(X)` gets no subject IDs, but every hidden sealed-phase window comes
from one of the 10 evaluation participants, whose sessions 1–3 are released
as labeled calibration. The submission therefore ships:

- **experts**: one copy of a pooled EEGNet per evaluation participant,
  fine-tuned (all weights, lr 1e-4, 50 epochs) on that participant's
  calibration data;
- **fingerprint**: an EEGNet that recognises the participant from the window;
- `predict(X)` mixes the experts by the fingerprint's posterior:
  `p(class | x) = Σ_s p(s | x) · p_s(class | x)`.

On BNCI2014_001 this beat an epoch-matched pooled control by +0.034 ± 0.007
(5/5 seeds) without IDs; see `experiments/fingerprint_tangermann/`.

## Current best model

1. **Input**: the kit's windows as-is (0–4 s from the cue, per-recording
   robust scaling). No extra preprocessing: baseline correction and
   asymmetry features did not help (`experiments/dreyer_eog/`).
2. **Pooled EEGNet** on all labeled data (training people + evaluation
   people's calibration sessions); AdamW lr 1e-3, epoch picked on held-out
   people.
3. **Per-person experts**: whole-network fine-tune of the pooled model per
   evaluation person on their calibration data; lr 1e-4, 50 epochs, **last
   epoch** (no epoch selection).
4. **Fingerprint**: a filter-bank covariance model recognises the person
   from one window: 6 bands (1–4, 4–8, 8–13, 13–20, 20–30, 30–45 Hz) → OAS
   covariance per band → Riemannian tangent space → multinomial logistic
   regression (C = 1), fitted on **all calibration sessions pooled**. Pure-torch
   inference module: `research/src/torch_fp.py` (loop A, `research/findings.md`).
   **Not yet in `submission.py`**, which still ships the EEGNet
   fingerprint (0.744 on R4–R6).
5. **Soft routing** in `predict(X)`.
6. **Per-person Riemannian expert, combined in log space** (loop B,
   `research/expert-weights/findings.md`): OAS covariance → tangent space → logistic
   regression (8–30 Hz, 0.5–4 s, **C = 0.1**) fitted on each person's
   calibration data. Per person k, before routing:
   log p ∝ 0.81·log p_EEGNet + (0.96 + 1.70·(rel_k − 0.5))·log p_Riemann,
   where rel_k is the person's run-to-run Riemannian accuracy on calibration
   data (weight ≈ 0.1 for people whose imagery it can't decode, ≈ 1 at 0.9).
   Not yet in `submission.py`; the torch export is verified
   (`research/expert-weights/src/torch_experts.py`, |Δp| 2e-8, float64).

| Evidence | Result |
|---|---|
| BNCI, 5 seeds, cross-session | mixture +0.034 ± 0.007 over the epoch-matched control, 5/5 seeds (oracle ID +0.045) |
| Dreyer simulation, seed 0 | pooled 0.873 → control 0.890 → **mixture 0.901** (oracle 0.908) |
| Dreyer, + Riemannian expert (step 6), 3 seeds, oracle ID | EEGNet experts 0.898 → **0.917** (+0.019; +0.014 / +0.020 / +0.024, every person-bootstrap CI > 0); NLL 0.253 → 0.215. Soft-routed with loop A's filter-bank fingerprint, seed 0: 0.921 |
| BNCI, 5 seeds, Dreyer's step-6 coefficients unchanged | +0.052 ± 0.003 over the EEGNet experts (0.745 → 0.797), 5/5 seeds |
| Contract | passes the kit's benchopt run, read-only; 5,040 windows in 57 s on CPU; ZIP 0.22 MB |
| Fingerprint, Dreyer R4–R6 (loop A, pre-registered) | filter-bank Riemann **0.996** (EEGNet 0.744); soft mixture 0.907 = oracle 0.9075 |
| Fingerprint, cross-day (loop A) | BNCI2014 day 1 → 2: 0.974; trained on 2 days → day 3: BNCI2015_001 1.000, Zhou2016 0.975 (1 day: 0.935, 0.790) |

**Tried and dropped**: EEGNet fingerprints (0.745 ± 0.10 on R3 at a fixed
budget, seed-sensitive; the shipped 0.857 picked its epoch on R3), EEGNet +
Riemann fingerprint ensembles (below Riemann alone), finer filter banks (won
on BNCI2014, lost on two held-out datasets), broadband covariance fingerprints
(delta is day-specific: 0.872 across days); per-person heads trained from scratch (−0.030 vs
pooled even with oracle ID, 0/5 seeds); BatchNorm-statistics-only adaptation
(±0.000); epoch selection on small validation sets; hard routing (≤ soft);
CSP experts or equal averaging of all expert types in the mixture (−0.004,
−0.068); a *plain* average with the Riemannian expert (its sign flips
between calibration folds, −0.018 to +0.010: the unregularised Riemannian
expert is over-confident); confidence gating, filter-bank or broadband
Riemannian experts; explicit hemispheric-asymmetry features (≈ per-channel band power);
per-trial pre-cue baseline correction (−0.03 within-subject).

**What it uses on Dreyer** (`experiments/dreyer_confound/`,
`experiments/dreyer_eog/`): mostly an early cue-locked brain response
(~250 ms), plus sustained eye position and some wrist EMG. Genuine motor
imagery is present but transfers poorly across people, which is where
per-person calibration helps.

**Next, in priority order**: integrate loop A's fingerprint and step 6 into
`submission.py` (all weights as tensors) and re-run the kit contract check;
then re-validate on the 2026 Graz + BrainHero data (47 channels incl.
EOG/EMG, 3 classes) once released. There, refit the step-6 coefficients on a
leave-one-calibration-session-out bank: across a day gap the EEGNet expert
becomes over-confident and the best Riemannian weight rises (BNCI prefers
≈ 0.67 / 0.83 over Dreyer's 0.81 / 0.42).

## Files

| File | Role |
|---|---|
| `submission.py` | the uploaded code: `Solver(CompetSolver)` + `FingerprintMixture`; maps evaluation channels to training channels **by name** and fails loudly on missing channels, `n_times` or `n_classes` mismatches |
| `train_mixture.py` | trains pooled, fingerprint and experts on Dreyer 2023, packages `outputs/track2_dreyer_sim/track2_dreyer_sim.zip` (`submission.py`, `mixture.pt`, `config.json` at the ZIP root), and scores the shipped code path |
| `classical_experts.py` | per-person CSP + LDA and Riemannian experts, alone and averaged with the packaged EEGNet experts, oracle / soft / hard routing (evaluation only; not yet in `submission.py`) |
| `results/` | per-seed results |

```bash
python track2/train_mixture.py --seed 0      # ~55 min on 8 CPUs
```

## Dreyer 2023 as a sealed-phase simulation

Dreyer has one session of six runs per person. The kit's test people
(subjects 61–81) play the evaluation participants: runs R1–R3 are their
calibration data, R4–R6 the hidden test. The training pool is the kit's
train split; the kit's val split selects the pooled model's epoch.

> ⚠️ **Do not upload this build to the warm-up leaderboard.** Its training
> uses labeled runs of the warm-up test subjects, so its warm-up score is
> inflated and would contaminate the leaderboard. It validates the design and
> the submission contract. A warm-up-legitimate build can only use the train
> split, where the test people are unseen, so there is nothing to calibrate
> on and the mixture reduces to a similarity-weighted ensemble of other
> people's experts.

## Contract check

The packaged ZIP, unzipped read-only, runs through the kit exactly as
Codabench ingestion does:

```bash
cd external/2026-competition
COMPET_SUBMISSION_DIR=$UNZIPPED benchopt run tracks/bci_decoding \
    -d "BCI[study=dreyer2023]" -s $UNZIPPED/submission.py
```

It passes: 5,040 warm-up test windows in ~1 min on CPU with 22 EEGNets
(budget: 60 min on one A100).

## Before the sealed phase

- Retrain on the 2026 Graz + BrainHero data: 47 channels (43 EEG, 2 EMG,
  2 EOG) at 500 Hz, 3 classes, 10 training + 10 evaluation participants.
  The channel mapping is by name, so the evaluation `meta["ch_names"]` must
  contain every training channel.
- The fingerprint bounds the gain (on BNCI, the weakest fingerprint seed kept
  only 62% of the oracle gain), so it is the first thing to improve.
- Declare the external data and a compute estimate (Rule 04).
