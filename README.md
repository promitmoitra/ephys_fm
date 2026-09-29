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

| Track | Warm-up data | Status | On disk | Floor score (warm-up split) |
|---|---|---|---|---|
| 1 | THINGS-EEG2 (59 GB) | — | | |
| 2 | Dreyer 2023 (19 GB) | — | | |
| 3 | Sleep-EDF | ✅ prepared | 26 GB | Median: W-bMAE 205.4 s, MAE 448.8 s |
| 4 | EMG2Pose | — | | |

Sleep-EDF notes: the S3 seed takes ~30 min and extraction ~10 min on 8 CPUs /
16 GB RAM. The 26 GB is 7 GB of flat EDFs (what the loader reads), a redundant
7 GB wget-layout copy under `physionet-sleep-data/physionet.org/…` (the kit only
checks that this folder exists, as its "seeded" marker), and a 12 GB extraction
cache.

## Submissions

`submissions/floor/floor_<track>.zip` are the constant baselines. Upload one per
track first to confirm ingestion and scoring work end to end.
