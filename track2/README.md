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
   regression (C = 1), fitted on **all calibration sessions pooled** (loop A,
   `research/findings.md`). Shipped as `submission.FBFingerprint` (float32).
5. **Soft routing** in `predict(X)`.
6. **Per-person Riemannian expert, combined in log space** (loop B,
   `research/expert-weights/findings.md`): OAS covariance → tangent space → logistic
   regression (8–30 Hz, 0.5–4 s, **C = 0.1**) fitted on each person's
   calibration data. Per person k, before routing:
   log p ∝ 0.81·log p_EEGNet + (0.96 + 1.70·(rel_k − 0.5))·log p_Riemann,
   where rel_k is the person's run-to-run Riemannian accuracy on calibration
   data (weight ≈ 0.1 for people whose imagery it can't decode, ≈ 1 at 0.9).
   Shipped as `submission.RiemannExperts` + `LogLinearCombiner` (float64).
   The four coefficients were fitted on Dreyer; refit them for a new dataset.

| Evidence | Result |
|---|---|
| BNCI, 5 seeds, cross-session | mixture +0.034 ± 0.007 over the epoch-matched control, 5/5 seeds (oracle ID +0.045) |
| **Shipped package, Dreyer simulation, seed 0** (`results/dreyer_sim_seed0_fb_riemann_c3_reused.md`) | `predict` on R4–R6: **0.921** (previous package 0.901); fingerprint 0.996 (was 0.744); EEGNet experts soft-routed 0.907, + Riemannian 0.921 = oracle 0.921 |
| Dreyer simulation, seed 0, previous package | pooled 0.873 → control 0.890 → mixture 0.901 (oracle 0.908) |
| Dreyer, + Riemannian expert (step 6), 3 seeds, oracle ID | EEGNet experts 0.898 → **0.917** (+0.019; +0.014 / +0.020 / +0.024, every person-bootstrap CI > 0); NLL 0.253 → 0.215. Soft-routed with loop A's filter-bank fingerprint, seed 0: 0.921 |
| BNCI, 5 seeds, Dreyer's step-6 coefficients unchanged | +0.052 ± 0.003 over the EEGNet experts (0.745 → 0.797), 5/5 seeds |
| Contract | the shipped package passes the kit's benchopt run, read-only: 5,040 warm-up test windows, 73 s end to end on CPU (4 threads); ZIP 4.0 MB. Its score there (0.947) is leaky: those windows include the calibration runs it trained on |
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

**Next, in priority order**: re-validate on the 2026 Graz + BrainHero data
(47 channels incl. EOG/EMG, 3 classes) once released; there, refit the
step-6 coefficients on a leave-one-calibration-session-out bank (across a day
gap the EEGNet expert becomes over-confident and the best Riemannian weight
rises: BNCI prefers ≈ 0.67 / 0.83 over Dreyer's 0.81 / 0.42). More seeds of
the full pipeline (only the step-6 gain has 3 seeds).

## Files

| File | Role |
|---|---|
| `submission.py` | the uploaded code: `Solver(CompetSolver)` + `FingerprintMixture`; maps evaluation channels to training channels **by name** and fails loudly on missing channels, `n_times` or `n_classes` mismatches |
| `train_mixture.py` | trains pooled EEGNet, experts, fingerprint, Riemannian experts and combiner on Dreyer 2023, packages `<out>/track2_dreyer_sim.zip` (`submission.py`, `mixture.pt`, `config.json` at the ZIP root), and scores the shipped code path with ablations. `--reuse-eegnet` loads the EEGNet experts of an existing `mixture.pt`; `--fingerprint eegnet` / `--no-riemann-experts` build the older designs |
| `riemann_parts.py` | training side of the covariance models: fits loop A's fingerprint and loop B's per-person Riemannian experts with scipy / pyriemann / sklearn and exports them as tensors for `submission.py`; holds the C3 coefficients |
| `classical_experts.py` | the original per-person CSP / Riemannian expert evaluation (plain averaging; superseded by step 6) |
| `results/` | per-seed results |

```bash
python track2/train_mixture.py --seed 0      # everything, ~55 min on 8 CPUs
# new fingerprint + Riemannian experts on the seed-0 EEGNet experts (~2 min)
python track2/train_mixture.py --threads 4 --out outputs/t2-integration/fb_c3 \
    --reuse-eegnet outputs/track2_dreyer_sim/submission/mixture.pt
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

The current package (`outputs/t2-integration/fb_c3/`) passes: 5,040 warm-up
test windows in 73 s on CPU (21 EEGNets, the filter-bank fingerprint and 21
Riemannian experts). The previous package passed in ~1 min with 22 EEGNets
(budget: 60 min on one A100).

## Before the sealed phase

- Retrain on the 2026 Graz + BrainHero data: 47 channels (43 EEG, 2 EMG,
  2 EOG) at 500 Hz, 3 classes, 10 training + 10 evaluation participants.
  The channel mapping is by name, so the evaluation `meta["ch_names"]` must
  contain every training channel.
- The fingerprint no longer bounds the gain on Dreyer (0.996; soft = oracle).
  Across days it relies on calibration from several days (loop A: one day →
  0.79–0.94 on held-out days, two days → 0.975–1.000).
- The covariance models bake the window length into their filter matrices
  (T × T per fingerprint band). At the sealed data's sampling rate, check
  their size and speed (at 500 Hz, 4 s: ≈ 100 MB for the fingerprint in
  float32), or resample before filtering.
- Declare the external data and a compute estimate (Rule 04).
