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

## Files

| File | Role |
|---|---|
| `submission.py` | the uploaded code: `Solver(CompetSolver)` + `FingerprintMixture`; maps evaluation channels to training channels **by name** and fails loudly on missing channels, `n_times` or `n_classes` mismatches |
| `train_mixture.py` | trains pooled, fingerprint and experts on Dreyer 2023, packages `outputs/track2_dreyer_sim/track2_dreyer_sim.zip` (`submission.py`, `mixture.pt`, `config.json` at the ZIP root), and scores the shipped code path |
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
