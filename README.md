# ephys_fm

Our entry for the [EEG/EMG Foundation Challenge 2026](https://neural-interfaces26.github.io/)
(NeurIPS 2026 Brain & Body Workshop), all four tracks.

| # | Track | `predict(X)` returns | Metric | Codabench |
|---|---|---|---|---|
| 1 | EEG-to-Image (`image_decoding`) | DINOv2 embeddings `(B, D)` | top-5 retrieval acc ↑ | [17974](https://www.codabench.org/competitions/17974/) |
| 2 | BCI decoding (`bci_decoding`) | class index `(B,)` | balanced acc ↑ | [17982](https://www.codabench.org/competitions/17982/) |
| 3 | Sleep onset (`sleep_onset`) | seconds to first N2 `(B,)` | weighted binned MAE ↓ | [17983](https://www.codabench.org/competitions/17983/) |
| 4 | EMG-to-Pose (`emg_pose`) | joint angles `(B, n_joints, T)`, **radians** | angular MAE (°) ↓ | [17984](https://www.codabench.org/competitions/17984/) |

**Key dates:** registration closes Oct 24 · warm-up until Oct 25 (5 subs/day) ·
sealed phase Oct 28 – Nov 21 (1 sub/day, the only phase that ranks).
**Limits:** inference ≤ 60 min on one A100 · 15 GB submission storage per profile
(shared across tracks) · declare all external pretraining data + compute estimate.

## Setup

Requires Python ≥ 3.12 (neuralset) and [uv](https://docs.astral.sh/uv/).

```bash
# upstream starter kit (gitignored; the platform evaluates with this exact code)
git clone https://github.com/neural-interfaces26/2026-competition external/2026-competition

uv venv --python 3.12 .venv
source .venv/bin/activate
# CPU torch; on a CUDA machine use e.g. https://download.pytorch.org/whl/cu126 (the platform's channel)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install -r external/2026-competition/requirements.txt
```

Run benchopt from the kit root (`cd external/2026-competition`). Keep data in the
gitignored `data/` at our repo root (shared by all tracks, survives re-cloning the
kit), with quieter logs as on the platform:

```bash
export BENCHOPT_DATA_HOME="$(realpath ../../data)"
export PYTHONWARNINGS=ignore::FutureWarning MNE_LOGGING_LEVEL=ERROR TQDM_DISABLE=1
```

## Usage

```bash
# zero-download smoke test (inference-only floor on simulated data)
benchopt run tracks/<track> --config tracks/<track>/starter.yml

# train a solver on simulated data -> packaged submission in tracks/<track>/outputs/<SolverName>/
benchopt run tracks/<track> -d Simulated -s Torch-Linear -o "<Objective>[training=True]"

# real warm-up data: download once, then train
benchopt prepare tracks/<track> --config tracks/<track>/training.yml
benchopt run     tracks/<track> --config tracks/<track>/training.yml -s MySolver

# check a submission folder against the contract
COMPET_SUBMISSION_DIR="$PWD/my_submission" benchopt run tracks/<track> -d Simulated -s MySolver
```

`<Objective>` is `Image-decoding`, `BCI-decoding`, `Sleep-onset` or `EMG-pose`.
A submission ZIP holds `submission.py` (with `class Solver(CompetSolver)`) and
its weights **at the ZIP root**; see the kit's `codabench/pages/participate.md`.

## Data

Prepared with `benchopt prepare tracks/<track> --config tracks/<track>/training.yml`
into `$BENCHOPT_DATA_HOME/neural_compet/`.

| Track | Warm-up data | Status | Raw + cache on disk | Floor score (warm-up split) |
|---|---|---|---|---|
| 1 | THINGS-EEG2 (59 GB) | — | | |
| 2 | Dreyer 2023 | ✅ prepared | 8.1 + ~2 GB | Constant: balanced acc 0.500 (2 classes) |
| 3 | Sleep-EDF | ✅ prepared | 7.1 + ~12 GB | Median: W-bMAE 205.4 s, MAE 448.8 s |
| 4 | EMG2Pose | — | | |

The extraction cache (`neural_compet/cache/`) is shared by all tracks; benchmark
runs read only the cache, the raw files matter only for re-extraction.

**Downloaders keep redundant copies.** They are deduplicated in place, without
triggering re-downloads (each step verified by reloading the raw data):

- *Sleep-EDF* (~30 min S3 seed, ~10 min extraction on 8 CPUs / 16 GB RAM): the
  kit also syncs a wget-layout copy under `physionet-sleep-data/physionet.org/…`.
  Only the flat EDFs are read; the kit just checks that the mirror folder exists
  (its "seeded" marker), so its files were deleted and the empty folder kept.
- *Dreyer 2023* (~2 h download, ~8 min extraction): MOABB ≥ 1.7 fetches every
  subject twice, from NEMAR (`NEMAR/nm000250/`) and from OSF
  (`MNE-dreyer2023-data/`, unzipped, zip kept). Only the OSF EDFs are read, but a
  missing NEMAR file or zip triggers a re-download. The NEMAR EDFs (byte-identical)
  are hard links to the OSF ones, and the zips are empty placeholders (MOABB checks
  existence only). 21 GB → 8.1 GB.
- Events sit on the cue for Dreyer (codes 769/770), so the kit's 0–4 s windows are
  cue + imagery. (MOABB's BNCI2014_001 events sit on the *trial start*, 2 s before
  the cue: check each new study's event semantics.)

## Experiments

| Experiment | Question | Result |
|---|---|---|
| [`fingerprint_tangermann`](experiments/fingerprint_tangermann/results/results.md) | Can one 4-s EEG window identify the subject on a different day, and does routing to per-subject decoders help? | Subject ID day 1 → day 2: 0.909 balanced acc (EEGNet; chance 0.111). Against a *pooled Riemannian* baseline, soft routing looked useful in 4-class MI (0.490 → 0.553, oracle 0.572), but the fair EEGNet test below reverses this |
| [`routing_eegnet`](experiments/fingerprint_tangermann/results/routing_eegnet.md) | Same trunk, same budget: does routing to per-subject EEGNet heads beat a pooled EEGNet? (5 seeds) | **No.** Pooled 0.692 ± 0.007; per-subject heads with oracle ID 0.661 (−0.030, 0/5 seeds), soft-routed 0.645, uniform-average control 0.595. Fingerprint still 0.886 ± 0.046. Per-subject heads trained from scratch lose to pooling |
| [`finetune_eegnet`](experiments/fingerprint_tangermann/results/finetune_eegnet.md) | Most favourable specialization: fine-tune a copy of the pooled EEGNet per subject (oracle ID) | Yes: whole-network fine-tune 0.744 vs pooled 0.692 (+0.053, 5/5 seeds); head-only +0.036. Last-epoch weights beat epoch selection on 48 val windows |
| [`finetune_routing_eegnet`](experiments/fingerprint_tangermann/results/finetune_routing_eegnet.md) | Is that gain just extra training, does it survive without IDs, and is it BatchNorm? | **Real and routable.** Beats an epoch-matched pooled control by +0.045 ± 0.004 (5/5); soft routing by the fingerprint keeps +0.034 ± 0.007 (5/5, ~75%). BN-statistics-only adaptation: ±0.000. **Track 2 design:** pooled model → per-participant fine-tunes on calibration sessions → mix by p(subject \| window) |
| [`dreyer_confound`](experiments/dreyer_confound/results/seed0.md) | Is the Dreyer warm-up (pooled EEGNet 0.80 cross-subject) decoding motor imagery, or the screen? | **Mostly a cue-locked signal.** The arrow period alone (0–1.25 s) scores 0.780 vs 0.799 for the full 0–4 s window; the feedback period alone scores 0.710. Not frontal: dropping Fz/F3/F4 costs 0.005 (horizontal saccades, F7/F8, not recorded, remain possible). Motor strip only 0.732. Shuffled 0.503. One seed |
| [`dreyer_eog`](experiments/dreyer_eog/README.md) | What is that cue-locked signal: eye movements, muscle, or EEG? Does motor imagery survive, and do asymmetry or baseline-relative features help? | **Mostly an early (~250 ms) cue-locked brain response**, plus sustained eye position (EOG montage inverted between Dreyer parts A and B/C: 58/60 vs 0/21) and some wrist EMG (0.59–0.63). Genuine MI lateralization in 70–80% of people but weak cross-subject (~0.56); within-person CSP best (0.628). Asymmetry ≈ band power; per-trial baseline correction hurts (−0.03). 3 seeds; the EOG baseline check is inconclusive (non-causal filter) |
| [`track2/classical_experts`](track2/results/classical_experts_seed0.md) | Do per-person CSP / Riemannian experts help the Track 2 mixture? | Alone far weaker (CSP 0.671, Riemannian 0.688 oracle); averaged with the EEGNet experts, Riemannian +0.008 soft-routed (11 / 9 people), CSP −0.004. One seed |
| [`research/` (loop B: expert weights)](research/findings.md) | Which rule, learned on calibration data only, should combine each person's EEGNet and Riemannian experts? | **Log-space pooling, with the Riemannian expert kept from being over-confident.** Reliability-weighted log-linear pooling (C3) on Dreyer R4–R6: 0.898 → **0.917** (+0.019, 3 seeds, every CI > 0), NLL 0.253 → 0.215. The plain average was unreliable (its sign flipped between dev folds). Coefficients transfer zero-shot to BNCI 2014-001 (+0.052, 5/5 seeds). Torch export verified. Decided on a cross-fitted dev bank; R4–R6 used only at pre-registered checkpoints |

The current best Track 2 model, the evidence for it and what was dropped are
summarized in [`track2/README.md`](track2/README.md#current-best-model).

## Submissions

`submissions/floor/floor_<track>.zip` are the constant baselines. Upload one per
track first to confirm ingestion and scoring work end to end.
