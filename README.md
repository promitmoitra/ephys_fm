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

Run benchopt from the kit root (`cd external/2026-competition`). Quieter logs, as on the platform:

```bash
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

`submissions/floor/floor_<track>.zip` are the constant baselines. Upload one per
track first to confirm ingestion and scoring work end to end.
