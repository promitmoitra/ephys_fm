# Track 2 Loop C: Expert Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen-REVE probe stream and a ShallowFBCSPNet expert stream next to the shipped
EEGNet and Riemannian experts. Weight all streams per dataset with an N-stream log-linear
combiner. Ship the streams that earn weight, and render a leakage-free benchopt comparison page.

**Architecture:** Every stream maps a window to per-person class log-probabilities `(B, K, C)`, so
routing (fingerprint) and combination stay architecture-agnostic.
- Training-side helpers (`track2/models.py`, `track2/reve_parts.py`) fit streams and export
  tensors.
- `track2/submission.py` holds the torch-only inference modules behind config flags.
- Research code under `research/expert-portfolio/` builds prediction banks and runs the locked
  evaluation, reusing loop B's banks and combiner code in `research/expert-weights/src/`.

**Tech Stack:**
- Python 3.12 venv at `.venv` (shared, symlinked; never install into it). torch, braindecode
  1.8.1 (`REVE`, `ShallowFBCSPNet`, `EEGNet`), scipy, sklearn, pyriemann, huggingface_hub.
- Tests: stdlib `unittest`, since pytest is not installed in the shared venv.
- benchopt from the kit at `external/2026-competition`.

**Spec:** `docs/superpowers/specs/2026-10-02-expert-portfolio-design.md`

## Global Constraints

- `submission.py` imports only torch and braindecode (plus stdlib). No sklearn, pyriemann or
  scipy at inference; weights as tensors.
- Inference ≤ 60 min on one A100 for all windows, streams and the fingerprint. Submission storage
  is 15 GB per profile, shared across four tracks.
- Pretrained models must load offline. Only `brain-bzh/reve-base` and `facebook/dinov2-giant`
  are pre-staged on the worker. Declare external data and compute (Rule 04).
- Local compute: ≤ 4 CPU threads for this loop (`--threads 4`, `OMP_NUM_THREADS=4`). Long jobs:
  `setsid nohup .venv/bin/python … > log 2>&1 < /dev/null &`, then find the Python PID with
  `ps -C python -o pid,args`.
- R4–R6 (Dreyer hidden runs) are report-only upper bounds: loops A and B already reused them,
  so they never gate a decision here except the pre-registered gross-failure veto. Ship decisions
  use the dev bank and BNCI; honest Dreyer estimates come from sim2 (Task 9). Protocols are committed
  before their results (separate commits).
- Branch `exp/expert-portfolio` only: run `git branch --show-current` before every commit.
  Never bare `git stash`. Commits end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run artifacts go to `outputs/t2-portfolio/`. Only read `data/` and `outputs/t2-expert-weights/`.
- REVE input: 200 Hz; per-channel z-score, clipped at ±15 SD (braindecode REVE docstring).
- Old packages (configs without the new keys) must keep loading and scoring as before (0.901 for
  `outputs/track2_dreyer_sim/submission`, 0.921 for `outputs/t2-integration/fb_c3/submission`).
- Never upload any Dreyer-simulation build to the warm-up leaderboard.

## Review Focus

1. **The sealed data's channel names missing from the REVE position bank.** Expected: packaging
   fails loudly, listing the missing names, instead of silently dropping channels (braindecode
   only logs a warning and returns fewer rows). Test in Task 4.
2. **A 500 Hz or non-120 Hz input.** Expected: the resampling matrix and the ShallowFBCSPNet time
   constants are derived from the data's `sfreq`, and an `sfreq` that doesn't match the packaged
   value raises in `build_model`. Tests in Tasks 3 and 4.
3. **The worker without network.** Expected: `load_model` builds REVE with
   `HF_HUB_OFFLINE=1` and the shipped positions file, with no download attempt. Test in Task 4
   (stub encoder + env check); real check in Task 10's contract run under `HF_HUB_OFFLINE=1`.
4. **A constant (flat) channel in a window.** Expected: the z-score doesn't produce NaN (std is
   clamped). Test in Task 4.
5. **An old `config.json` without the new keys.** Expected: it loads as before, with no REVE
   download attempted. Test in Task 5.

---

## File Structure

| Path | Responsibility |
|---|---|
| `track2/models.py` (create) | `make_model(arch, …)`: EEGNet or ShallowFBCSPNet, with ShallowFBCSPNet time constants scaled to `sfreq` |
| `track2/reve_parts.py` (create) | Training side of the REVE stream: resampling matrix, position lookup, offline encoder loading, embedding, pooled and per-person head fitting, export to `submission.ReveProbe` state |
| `track2/submission.py` (modify) | Adds `standardize_clip`, `ReveProbe`, `PortfolioCombiner`, a ShallowFBCSPNet expert list, and config flags `shallow_experts`, `reve_probe`, `combiner: "portfolio"` |
| `track2/train_mixture.py` (modify) | Uses `make_model`; packages the new streams (`--shallow-experts`, `--reve-probe`, `--combiner portfolio`) |
| `track2/tests/` (create) | `unittest` tests: `test_models.py`, `test_reve_parts.py`, `test_portfolio.py` |
| `track2/bench/` (create) | Page solvers (`shallow_solver.py`, `reve_solver.py`, `riemann_solver.py`) and `train_bench.py` (train-split-only weights) |
| `research/expert-portfolio/` (create) | Loop C workspace: state, log, findings, protocols, `src/stream_bank.py` (banks for new streams), `src/portfolio.py` (N-stream combiner + evaluation), reports |

---

### Task 1: Loop C workspace and the locked evaluation protocol

**Files:**
- Create: `research/expert-portfolio/research-state.yaml`
- Create: `research/expert-portfolio/research-log.md`
- Create: `research/expert-portfolio/findings.md`
- Create: `research/expert-portfolio/experiments/00-protocol/protocol.md`

**Interfaces:**
- Consumes: the spec's decision rule.
- Produces: the locked evaluation that Tasks 7–8 cite.

- [ ] **Step 1: Create the workspace files**

`research/expert-portfolio/research-state.yaml`:
```yaml
project:
  title: "Track 2 loop C: expert portfolio behind a learned combiner"
  question: >
    Do a frozen-REVE probe and ShallowFBCSPNet experts earn weight next to the shipped EEGNet and
    Riemannian experts, on Dreyer (dev) and across days (BNCI 2014-001)?
  status: active
  started: "2026-10-02"
  worktree: .claude/worktrees/t2-portfolio
  branch: exp/expert-portfolio
  spec: docs/superpowers/specs/2026-10-02-expert-portfolio-design.md
  plan: docs/superpowers/plans/2026-10-02-expert-portfolio.md
hypotheses:
  - {id: S1, statement: "REVE frozen probe earns weight (drop-one ≥ 0.005 acc or NLL) and does not hurt on BNCI", status: pending}
  - {id: S2, statement: "ShallowFBCSPNet experts earn weight and do not hurt on BNCI", status: pending}
jobs: []
checkpoints: []
```

`research/expert-portfolio/research-log.md`:
```markdown
# Research log: loop C (expert portfolio)

## 2026-10-02

- Workspace created; protocol 00 locked (evaluation and ship rule from the spec).
```

`research/expert-portfolio/findings.md`:
```markdown
# Findings: loop C (expert portfolio)

## Current understanding

(none yet)

## Lessons and constraints

- R4–R6 are report-only upper bounds (reused by loops A and B); honest Dreyer numbers come
  from sim2, whose 14 people are fixed in protocol 00 and never used for any choice.
```

- [ ] **Step 2: Write the locked protocol**

`research/expert-portfolio/experiments/00-protocol/protocol.md`:
```markdown
# Protocol 00: loop C's locked evaluation and ship rule

Locked 2026-10-02, before any new stream exists. Source: the spec's "Evaluation protocol".

## Banks
1. **Dreyer cross-fitted dev bank:** each held-out calibration run r ∈ {R1, R2, R3}; streams
   trained on the training pool + the other two runs (pooled parts) and on those two runs
   (per-person parts); 2,520 windows. Window order and masks identical to loop B's
   `outputs/t2-expert-weights/bank_xfit_seed0.npz` (fold r = 0, 1, 2; within a fold, dataset
   order of `split == "test" & run == r`).
2. **Dreyer test bank:** streams trained on R1–R3, predicting R4–R6. Report-only (upper bound).
3. **BNCI 2014-001:** session 1 runs 0–4 train, run 5 val, session 2 test; seeds 0–4.

## Metrics (true person ID)
- Each stream alone: balanced accuracy, NLL.
- Full N-stream combination (leave-one-person-out on the dev bank) and drop-one ablations.
- Per-person paired differences with person-bootstrap 95% CIs (5,000 resamples, rng seed 0).
- Shortcut diagnostic per stream: early-only input (samples ≥ 1.25 s zeroed) and late-only
  input (samples < 1.25 s zeroed); models trained on full windows.

## Ship rule (from the spec)
A new stream ships if
(a) dropping it from the full combination on the dev bank costs ≥ 0.005 balanced accuracy, or
≥ 0.005 NLL with the bootstrap CI of the NLL difference excluding zero; and
(b) on BNCI, adding it to the shipped two-stream combination does not lower balanced accuracy
(mean over 5 seeds ≥ −0.005) with Dreyer-fitted weights.
**Only (a) and (b) decide.** R4–R6 have already gated decisions in loops A and B; their scores
are optimistic upper bounds and are report-only here.

## Honest estimates
- **sim2:**
  - The 14 people in `sim2_people.json` (drawn below, before any training) are never used for
    any choice.
  - Their R1–R3 are calibration and their R4–R6 hidden.
  - The training pool is the other 38 training-pool people plus all six runs of the 21 original
    evaluation people; the pooled epoch is chosen on the kit's val people.
  - Full pipeline, seeds 0–2.
- **R4–R6:** candidates scored for seeds 0–2, report-only.
- **Veto (gross failure only):** withdraw a candidate if the full combination is more than 0.02
  below the two-stream combination on sim2 (mean over seeds 0–2) or on R4–R6 (mean over seeds
  0–2).
```

- [ ] **Step 2b: Draw and lock the sim2 people**

Run:
```bash
.venv/bin/python - <<'EOF'
import json, numpy as np
d = np.load("data/experiments/dreyer_windows.npz", allow_pickle=True)
pool = sorted(np.unique(d["subject"][d["split"] == "train"]), key=int)
assert len(pool) == 52, len(pool)
people = sorted(np.random.default_rng(20261002).choice(pool, 14, replace=False).tolist(), key=int)
json.dump({"seed": 20261002, "from": "kit train split (52 people)", "people": people},
          open("research/expert-portfolio/experiments/00-protocol/sim2_people.json", "w"), indent=2)
print(people)
EOF
```
Expected: 14 subject IDs printed and the JSON written. Every person must have 6 runs × 40
windows; check with:
`np.unique(d["run"][d["subject"] == s], return_counts=True)`.

- [ ] **Step 3: Commit**

```bash
git branch --show-current   # expect exp/expert-portfolio
git add research/expert-portfolio
git commit -m "research(protocol): loop C workspace, locked evaluation and sim2 people (protocol 00)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Probe P0, REVE offline feasibility (spike; go / no-go for S1)

**Files:**
- Create: `research/expert-portfolio/experiments/p0-reve-offline/probe.py`
- Create: `research/expert-portfolio/experiments/p0-reve-offline/results.md`

**Interfaces:**
- Consumes: Hugging Face network access (local only), `data/experiments/dreyer_windows.npz`,
  `data/experiments/tangermann_windows.npz`.
- Produces:
  - the go / no-go decision, recorded in `results.md`;
  - `outputs/t2-portfolio/reve_positions/reve_positions.json` (the position bank, cached once);
  - the confirmed constructor call and feature shape used by Task 4.

- [ ] **Step 1: Write the probe**

```python
"""P0: can REVE run offline with shipped positions, and how fast does it embed on CPU?"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "outputs" / "t2-portfolio"
POS_DIR = OUT / "reve_positions"


def main():
    torch.set_num_threads(4)
    POS_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. Prefetch (online, once): weights into the HF cache, positions into POS_DIR.
    os.environ["REVE_POSITIONS_PATH"] = str(POS_DIR)
    from braindecode.models import REVE
    t0 = time.time()
    # The plain call may fail if REVE's constructor needs the input geometry; then retry with it.
    kwargs = {}
    try:
        enc = REVE.from_pretrained("brain-bzh/reve-base")
    except Exception as e:                                  # record and retry
        report["plain_call_error"] = f"{type(e).__name__}: {e}"[:300]
        kwargs = dict(n_outputs=2, n_chans=27, n_times=800, sfreq=200)
        enc = REVE.from_pretrained("brain-bzh/reve-base", **kwargs)
    report["constructor_kwargs"] = kwargs
    report["download_s"] = round(time.time() - t0, 1)
    report["n_params"] = int(sum(p.numel() for p in enc.parameters()))
    report["positions_file"] = (POS_DIR / "reve_positions.json").exists()

    # 2. Offline reload: no network, positions from POS_DIR only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    enc = REVE.from_pretrained("brain-bzh/reve-base", **kwargs).eval()
    report["offline_load"] = "ok"

    # 3. Channel coverage for both datasets.
    bank = json.loads((POS_DIR / "reve_positions.json").read_text())
    for name, path in [("dreyer", "dreyer_windows.npz"), ("bnci", "tangermann_windows.npz")]:
        d = np.load(REPO / "data" / "experiments" / path, allow_pickle=True)
        chs = [str(c) for c in d["ch_names"]]
        report[f"{name}_missing_channels"] = [c for c in chs if c not in bank]

    # 4. Feature shape and timing on 1,000 Dreyer windows (zeros-safe standardised input).
    d = np.load(REPO / "data" / "experiments" / "dreyer_windows.npz", allow_pickle=True)
    chs = [str(c) for c in d["ch_names"]]
    pos = torch.tensor([bank[c] for c in chs], dtype=torch.float32)
    from scipy.signal import resample_poly
    X = resample_poly(d["X"][:1000], 5, 3, axis=-1).astype(np.float32)        # 120 -> 200 Hz
    X = (X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-6)
    X = np.clip(X, -15, 15)
    with torch.inference_mode():
        f = enc(torch.from_numpy(X[:2]), pos=pos.expand(2, -1, -1), return_features=True)
        report["features_shape"] = list(f["features"].shape)
        t0 = time.time()
        for i in range(0, 1000, 50):
            xb = torch.from_numpy(X[i:i + 50])
            enc(xb, pos=pos.expand(len(xb), -1, -1), return_features=True)
        report["embed_s_per_1000"] = round(time.time() - t0, 1)
    report["est_dreyer_all_min"] = round(report["embed_s_per_1000"] * 20.792 / 60, 1)
    # The constructor call travels with the positions file (read by load_reve_encoder).
    (POS_DIR / "reve_kwargs.json").write_text(json.dumps(kwargs))
    print(json.dumps(report, indent=2))
    (Path(__file__).parent / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

Run: `OMP_NUM_THREADS=4 .venv/bin/python -W ignore research/expert-portfolio/experiments/p0-reve-offline/probe.py`

Expected:
- `offline_load: ok`;
- `features_shape: [2, 27, 4, 512]` (27 channels × 4 patches × 512);
- empty `dreyer_missing_channels`;
- `embed_s_per_1000` printed.

If `from_pretrained` raises (e.g. gated access), record the exact error.

- [ ] **Step 3: Record the decision**

Write `results.md` with the JSON numbers and one of:
- **GO:** offline load ok, no missing Dreyer channels, estimated full Dreyer embedding ≤ 120 min.
  Continue with Task 4.
- **NO-GO:** state the blocker. Skip Tasks 4 and the REVE parts of Tasks 6–10, and record S1 as
  "not feasible" in `research-state.yaml`.

For BNCI channels missing from the bank, list them. Task 6 drops those channels for BNCI (logged)
only if fewer than 3 are missing; otherwise BNCI is REVE-skipped.

- [ ] **Step 4: Commit**

```bash
git branch --show-current
git add research/expert-portfolio/experiments/p0-reve-offline
git commit -m "research(results): P0 REVE offline probe — <GO|NO-GO>, <s>/1000 windows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `make_model` with ShallowFBCSPNet scaled to the sampling rate

**Files:**
- Create: `track2/models.py`
- Create: `track2/tests/__init__.py` (empty)
- Create: `track2/tests/test_models.py`
- Modify: `track2/train_mixture.py` (replace `make_eegnet` body by a call to `make_model`)

**Interfaces:**
- Produces:
  - `make_model(arch: str, n_chans: int, n_outputs: int, n_times: int, sfreq: float) -> nn.Module`,
    with `arch ∈ {"eegnet", "shallow"}`;
  - `shallow_kwargs(sfreq: float) -> dict`.
  - `train_mixture.make_eegnet(n_chans, n_outputs, n_times)` keeps its signature (returns
    `make_model("eegnet", …, sfreq=120.0)`).

- [ ] **Step 1: Write the failing tests**

`track2/tests/test_models.py`:
```python
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import make_model, shallow_kwargs  # noqa: E402


class TestModels(unittest.TestCase):
    def test_shallow_constants_scale_with_sfreq(self):
        self.assertEqual(shallow_kwargs(250.0), dict(filter_time_length=25,
                         pool_time_length=75, pool_time_stride=15))
        self.assertEqual(shallow_kwargs(120.0), dict(filter_time_length=12,
                         pool_time_length=36, pool_time_stride=7))
        self.assertEqual(shallow_kwargs(500.0), dict(filter_time_length=50,
                         pool_time_length=150, pool_time_stride=30))

    def test_output_shapes(self):
        for arch in ("eegnet", "shallow"):
            m = make_model(arch, n_chans=27, n_outputs=3, n_times=480, sfreq=120.0).eval()
            self.assertEqual(tuple(m(torch.zeros(4, 27, 480)).shape), (4, 3), arch)

    def test_unknown_arch_raises(self):
        with self.assertRaises(ValueError):
            make_model("resnet", 27, 2, 480, 120.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -W ignore -m unittest track2/tests/test_models.py -v`
Expected: `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement `track2/models.py`**

```python
"""Neural expert architectures for Track 2 (shared by training and submission tooling)."""

SHALLOW_REF_SFREQ = 250.0          # braindecode ShallowFBCSPNet defaults assume 250 Hz
_SHALLOW_REF = dict(filter_time_length=25, pool_time_length=75, pool_time_stride=15)


def shallow_kwargs(sfreq):
    """ShallowFBCSPNet time constants rescaled from 250 Hz to `sfreq` (same durations)."""
    s = sfreq / SHALLOW_REF_SFREQ
    return {k: max(1, int(round(v * s))) for k, v in _SHALLOW_REF.items()}


def make_model(arch, n_chans, n_outputs, n_times, sfreq):
    if arch == "eegnet":
        from braindecode.models import EEGNet
        return EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    if arch == "shallow":
        from braindecode.models import ShallowFBCSPNet
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
                               final_conv_length="auto", **shallow_kwargs(sfreq))
    raise ValueError(f"unknown arch {arch!r}")
```

Note: `round(7.2) = 7`, `round(36.0) = 36`, `round(12.0) = 12`; the expected test values follow.

- [ ] **Step 4: Route `train_mixture.make_eegnet` through it**

In `track2/train_mixture.py`, replace the body of `make_eegnet`:
```python
def make_eegnet(n_chans, n_outputs, n_times):
    from models import make_model
    return make_model("eegnet", n_chans, n_outputs, n_times, sfreq=120.0)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -W ignore -m unittest track2/tests/test_models.py -v`
Expected: 3 tests OK.

- [ ] **Step 6: Commit**

```bash
git branch --show-current
git add track2/models.py track2/tests track2/train_mixture.py
git commit -m "Add make_model: EEGNet or ShallowFBCSPNet with time constants scaled to sfreq

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: REVE probe stream (training side + `submission.ReveProbe`)

Skip if P0 was NO-GO.

**Files:**
- Create: `track2/reve_parts.py`
- Modify: `track2/submission.py` (add `standardize_clip`, `ReveProbe`, `load_reve_encoder`)
- Create: `track2/tests/test_reve_parts.py`

**Interfaces:**
- Consumes:
  - from P0: the constructor `REVE.from_pretrained("brain-bzh/reve-base")`, and
    `forward(x, pos=(B, C, 3), return_features=True)["features"]` with shape `(B, C, P, 512)`;
  - the cached `outputs/t2-portfolio/reve_positions/reve_positions.json`.
- Produces, in `reve_parts.py`:
  - `resample_matrix(n_times: int, sfreq_in: float, sfreq_out: float = 200.0) -> np.ndarray`,
    shape `(n_times, n_out)`;
  - `positions(ch_names: list[str], bank_file: Path) -> np.ndarray`, shape `(C, 3)`; raises
    `ValueError` listing missing names;
  - `embed(encoder, X: np.ndarray, R: np.ndarray, pos: np.ndarray, batch: int = 50) -> np.ndarray`,
    shape `(n, 512)`, float32;
  - `fit_head(Z, y, n_classes, lam, W0=None, b0=None) -> (W (C, D), b (C,))`, float64;
  - `fit_person_heads(Z, y, person, n_people, n_classes, W0, b0, lam) -> (W (K, C, D), b (K, C))`;
  - `export(R, pos, mu, sd, W, b) -> dict`: the state dict for `submission.ReveProbe`.
- Produces, in `submission.py`:
  - `standardize_clip(X, clip=15.0)`;
  - `ReveProbe(encoder, n_people, n_classes, n_chans, n_times, n_out, emb_dim=512)` with
    `forward(X) -> (B, K, C)` log-probabilities;
  - `load_reve_encoder(positions_dir: Path) -> nn.Module`.

- [ ] **Step 1: Write the failing tests**

`track2/tests/test_reve_parts.py`:
```python
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent / "external" / "2026-competition"), str(HERE)]
import reve_parts  # noqa: E402
from submission import ReveProbe, load_reve_encoder, standardize_clip  # noqa: E402


class StubEncoder(torch.nn.Module):
    """Mimics REVE's return_features output: (B, C, P, 512) from (B, C, T)."""

    def forward(self, x, pos=None, return_features=False):
        B, C, T = x.shape
        P = 4
        f = x[:, :, : P * 128].reshape(B, C, P, 128).repeat(1, 1, 1, 4)
        return {"features": f, "cls_token": None}


class TestReveParts(unittest.TestCase):
    def test_resample_matrix_matches_scipy(self):
        from scipy.signal import resample_poly
        R = reve_parts.resample_matrix(480, 120.0, 200.0)
        self.assertEqual(R.shape, (480, 800))
        x = np.random.default_rng(0).standard_normal((3, 27, 480))
        np.testing.assert_allclose(x @ R, resample_poly(x, 5, 3, axis=-1), atol=1e-10)

    def test_resample_matrix_500hz(self):
        self.assertEqual(reve_parts.resample_matrix(2000, 500.0, 200.0).shape, (2000, 800))

    def test_positions_missing_channel_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "reve_positions.json"
            f.write_text(json.dumps({"C3": [0.1, 0.2, 0.3], "C4": [0.4, 0.5, 0.6]}))
            np.testing.assert_allclose(reve_parts.positions(["C4", "C3"], f),
                                       [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]])
            with self.assertRaisesRegex(ValueError, "EOG1"):
                reve_parts.positions(["C3", "EOG1"], f)

    def test_standardize_clip_flat_channel_no_nan(self):
        X = torch.randn(2, 3, 100) * 50
        X[:, 1] = 7.0                                              # flat channel
        Z = standardize_clip(X)
        self.assertFalse(torch.isnan(Z).any())
        self.assertLessEqual(Z.abs().max().item(), 15.0)
        self.assertTrue(torch.allclose(Z[:, 1], torch.zeros_like(Z[:, 1])))

    def test_fit_head_separable(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((200, 16))
        y = (Z[:, 0] > 0).astype(int)
        W, b = reve_parts.fit_head(Z, y, n_classes=2, lam=1e-3)
        self.assertGreater(((Z @ W.T + b).argmax(1) == y).mean(), 0.95)

    def test_person_heads_shrink_to_pooled(self):
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((60, 8)); y = rng.integers(0, 2, 60); p = np.repeat([0, 1], 30)
        W0, b0 = reve_parts.fit_head(Z, y, 2, lam=1e-3)
        W, b = reve_parts.fit_person_heads(Z, y, p, 2, 2, W0, b0, lam=1e6)
        np.testing.assert_allclose(W[0], W0, atol=1e-3)

    def test_probe_forward_shapes_and_export_roundtrip(self):
        rng = np.random.default_rng(2)
        R = reve_parts.resample_matrix(480, 120.0, 200.0)
        pos = rng.standard_normal((27, 3))
        mu, sd = np.zeros(512), np.ones(512)
        W, b = rng.standard_normal((3, 2, 512)), rng.standard_normal((3, 2))
        state = reve_parts.export(R, pos, mu, sd, W, b)
        probe = ReveProbe(StubEncoder(), n_people=3, n_classes=2, n_chans=27, n_times=480,
                          n_out=800)
        probe.load_state_dict(state)
        lp = probe(torch.randn(5, 27, 480))
        self.assertEqual(tuple(lp.shape), (5, 3, 2))
        self.assertTrue(torch.allclose(lp.exp().sum(-1), torch.ones(5, 3, dtype=lp.dtype)))

    def test_load_encoder_sets_offline_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                load_reve_encoder(Path(tmp), _constructor=lambda: torch.nn.Identity())
            finally:
                self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
                self.assertEqual(os.environ.get("REVE_POSITIONS_PATH"), tmp)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -W ignore -m unittest track2/tests/test_reve_parts.py -v`
Expected: `ModuleNotFoundError: No module named 'reve_parts'` (or an ImportError for `ReveProbe`).

- [ ] **Step 3: Implement `track2/reve_parts.py`**

```python
"""Training side of the frozen-REVE probe stream (loop C).

The encoder is frozen and loaded from the Hugging Face cache; only linear heads are trained:
a pooled multinomial head on all labelled windows, then one head per evaluation person,
fitted on their calibration windows with an L2 pull toward the pooled head.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

REVE_SFREQ = 200.0


def resample_matrix(n_times, sfreq_in, sfreq_out=REVE_SFREQ):
    """(n_times, n_out) matrix M with resample_poly(x, up, down) == x @ M."""
    from scipy.signal import resample_poly
    fr = Fraction(sfreq_out / sfreq_in).limit_denominator(1000)
    return np.ascontiguousarray(
        resample_poly(np.eye(n_times), fr.numerator, fr.denominator, axis=-1))


def positions(ch_names, bank_file):
    bank = json.loads(Path(bank_file).read_text())
    missing = [c for c in ch_names if c not in bank]
    if missing:
        raise ValueError(f"channels missing from the REVE position bank: {missing}")
    return np.asarray([bank[c] for c in ch_names], dtype=np.float64)


def embed(encoder, X, R, pos, batch=50):
    """Mean-pooled REVE features (n, 512) for windows X (n, C, T) at the data's rate."""
    from submission import standardize_clip
    Rt = torch.as_tensor(R, dtype=torch.float32)
    P = torch.as_tensor(pos, dtype=torch.float32)
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            x = standardize_clip(torch.as_tensor(X[i:i + batch], dtype=torch.float32) @ Rt)
            f = encoder(x, pos=P.expand(len(x), -1, -1), return_features=True)["features"]
            out.append(f.mean(dim=(1, 2)).float().numpy())
    return np.concatenate(out)


def _fit(Z, y, n_classes, lam, W0, b0):
    Z = torch.as_tensor(Z, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.long)
    D = Z.shape[1]
    W0 = torch.zeros(n_classes, D, dtype=torch.float64) if W0 is None else torch.as_tensor(W0)
    b0 = torch.zeros(n_classes, dtype=torch.float64) if b0 is None else torch.as_tensor(b0)
    W = W0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z @ W.T + b, y) \
            + 0.5 * lam * (((W - W0) ** 2).sum() + ((b - b0) ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def fit_head(Z, y, n_classes, lam, W0=None, b0=None):
    """Multinomial logistic head minimising mean NLL + lam/2 · ||θ − θ0||²."""
    return _fit(Z, y, n_classes, lam, W0, b0)


def fit_person_heads(Z, y, person, n_people, n_classes, W0, b0, lam):
    W, b = [], []
    for k in range(n_people):
        m = person == k
        Wk, bk = _fit(Z[m], y[m], n_classes, lam, W0, b0)
        W.append(Wk)
        b.append(bk)
    return np.stack(W), np.stack(b)


def export(R, pos, mu, sd, W, b):
    """State dict for submission.ReveProbe. Heads act on (z − mu) / sd embeddings."""
    f32, f64 = torch.float32, torch.float64
    return {"resample": torch.as_tensor(R, dtype=f32),
            "pos": torch.as_tensor(pos, dtype=f32),
            "mu": torch.as_tensor(mu, dtype=f64), "sd": torch.as_tensor(sd, dtype=f64),
            "weight": torch.as_tensor(W, dtype=f64), "bias": torch.as_tensor(b, dtype=f64)}
```

- [ ] **Step 4: Add the inference modules to `track2/submission.py`**

Add `import os` and `from pathlib import Path` to the imports. Then add after `LogLinearCombiner`:
```python
def standardize_clip(X, clip=15.0):
    """Per-window, per-channel z-score clipped at ±clip SD (REVE's pretraining input)."""
    mu = X.mean(-1, keepdim=True)
    sd = X.std(-1, keepdim=True).clamp_min(1e-6)
    return ((X - mu) / sd).clamp(-clip, clip)


def load_reve_encoder(positions_dir, _constructor=None):
    """Frozen REVE from the pre-staged Hugging Face cache, fully offline.

    braindecode's REVE reads its position bank from $REVE_POSITIONS_PATH/reve_positions.json
    when it is constructed; the submission ships that file, plus reve_kwargs.json (the
    constructor arguments probe P0 found necessary, possibly {})."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["REVE_POSITIONS_PATH"] = str(positions_dir)
    if _constructor is None:
        from braindecode.models import REVE
        kw_file = Path(positions_dir) / "reve_kwargs.json"
        kwargs = json.loads(kw_file.read_text()) if kw_file.exists() else {}

        def _constructor():
            return REVE.from_pretrained("brain-bzh/reve-base", **kwargs)
    enc = _constructor().eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


class ReveProbe(nn.Module):
    """Frozen REVE encoder + per-participant linear heads (loop C); forward -> (B, K, C)."""

    def __init__(self, encoder, n_people, n_classes, n_chans, n_times, n_out, emb_dim=512):
        super().__init__()
        self.encoder = encoder           # not in this module's state dict (frozen, cached)
        self.register_buffer("resample", torch.zeros(n_times, n_out))
        self.register_buffer("pos", torch.zeros(n_chans, 3))
        d = torch.float64
        self.register_buffer("mu", torch.zeros(emb_dim, dtype=d))
        self.register_buffer("sd", torch.ones(emb_dim, dtype=d))
        self.register_buffer("weight", torch.zeros(n_people, n_classes, emb_dim, dtype=d))
        self.register_buffer("bias", torch.zeros(n_people, n_classes, dtype=d))

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("encoder.")}

    def load_state_dict(self, state, strict=True):
        return super().load_state_dict(state, strict=False)

    def forward(self, X):
        x = standardize_clip(X.to(self.resample.dtype) @ self.resample)
        f = self.encoder(x, pos=self.pos.expand(len(x), -1, -1), return_features=True)
        z = (f["features"].mean(dim=(1, 2)).to(self.mu.dtype) - self.mu) / self.sd
        logits = torch.einsum("bd,kcd->bkc", z, self.weight) + self.bias
        return torch.log_softmax(logits, -1)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -W ignore -m unittest track2/tests/test_reve_parts.py -v`
Expected: 8 tests OK.

- [ ] **Step 6: Commit**

```bash
git branch --show-current
git add track2/reve_parts.py track2/submission.py track2/tests/test_reve_parts.py
git commit -m "Add the frozen-REVE probe stream: resampling, offline encoder, heads, ReveProbe

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: N-stream combiner (research + inference) and the submission's config flags

**Files:**
- Create: `research/expert-portfolio/src/portfolio.py` (combiner part; evaluation added in Task 7)
- Modify: `track2/submission.py` (`PortfolioCombiner`, `FingerprintMixture` streams, `build_model` flags)
- Create: `track2/tests/test_portfolio.py`

**Interfaces:**
- Consumes:
  - loop B's `research/expert-weights/src/combine.py`: `Combiner`, `RelLogLinear`, `own`,
    `load_bank`, `eval_global`;
  - Task 4's `ReveProbe` and `load_reve_encoder`;
  - Task 3's `make_model`.
- Produces:
  - `portfolio.PortfolioLogLinear(experts: list[str])` with
    `experts = [*neural_streams, riemann_key, rel_key]`, a loop B `Combiner` subclass;
    `theta = {"w": (S,), "c": (2,), "b": (C-1,)}`;
  - `submission.PortfolioCombiner(n_streams, n_people, n_classes)`: buffers `w (S,)`,
    `c (2,)`, `class_bias (C,)`, `rel (K,)`;
    `forward(neural: list[(B,K,C)], logp_riemann) -> (B,K,C)`;
  - `FingerprintMixture(..., shallow=False, reve=None, combiner="C3")`;
  - config keys: `"shallow_experts": bool`, `"reve_probe": bool`,
    `"combiner": "C3" | "portfolio"`, `"stream_order": [...]`.

- [ ] **Step 1: Write the failing tests**

`track2/tests/test_portfolio.py`:
```python
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(HERE),
                str(REPO / "research" / "expert-weights" / "src"),
                str(REPO / "research" / "expert-portfolio" / "src")]
from submission import LogLinearCombiner, PortfolioCombiner, build_model  # noqa: E402
from portfolio import PortfolioLogLinear  # noqa: E402
from combine import RelLogLinear  # noqa: E402


def logp(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(*shape, generator=g, dtype=torch.float64), -1)


class TestPortfolio(unittest.TestCase):
    def test_inference_one_stream_equals_c3(self):
        K, C = 4, 2
        rel = torch.rand(K, dtype=torch.float64)
        c3 = LogLinearCombiner(K, C)
        c3.coef.copy_(torch.tensor([0.81, 0.96, 1.70]))
        c3.class_bias.copy_(torch.tensor([0.0, -0.016])); c3.rel.copy_(rel)
        pc = PortfolioCombiner(1, K, C)
        pc.w.copy_(torch.tensor([0.81])); pc.c.copy_(torch.tensor([0.96, 1.70]))
        pc.class_bias.copy_(torch.tensor([0.0, -0.016])); pc.rel.copy_(rel)
        e, r = logp(5, K, C, seed=1), logp(5, K, C, seed=2)
        self.assertTrue(torch.allclose(c3(e, r), pc([e], r)))

    def test_research_one_stream_equals_rel_loglinear(self):
        rng = np.random.default_rng(0)
        n, C = 300, 2
        F = {"eegnet": torch.log_softmax(torch.tensor(rng.standard_normal((n, C))), -1).numpy(),
             "ts": torch.log_softmax(torch.tensor(rng.standard_normal((n, C))), -1).numpy(),
             "rel": rng.uniform(0.4, 1.0, (n, 1))}
        y = rng.integers(0, 2, n)
        a = RelLogLinear(["eegnet", "ts", "rel"]).fit(F, y).predict(F)
        b = PortfolioLogLinear(["eegnet", "ts", "rel"]).fit(F, y).predict(F)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_old_config_loads_without_new_keys(self):
        sub = REPO / "outputs" / "t2-integration" / "fb_c3" / "submission"
        if not sub.exists():
            self.skipTest("integration package not present")
        config = json.loads((sub / "config.json").read_text())
        state = torch.load(sub / "mixture.pt", map_location="cpu", weights_only=True)
        meta = {"ch_names": config["ch_names"], "n_times": config["n_times"],
                "n_classes": config["n_classes"], "device": "cpu",
                "sfreq": config["sfreq"]}
        m = build_model(meta, config, state)
        self.assertIsNone(m.reve)
        self.assertEqual(len(m.shallow_experts), 0)

    def test_sfreq_mismatch_raises(self):
        sub = REPO / "outputs" / "t2-integration" / "fb_c3" / "submission"
        if not sub.exists():
            self.skipTest("integration package not present")
        config = json.loads((sub / "config.json").read_text())
        meta = {"ch_names": config["ch_names"], "n_times": config["n_times"],
                "n_classes": config["n_classes"], "device": "cpu", "sfreq": 500.0}
        with self.assertRaisesRegex(ValueError, "sfreq"):
            build_model(meta, config, None)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -W ignore -m unittest track2/tests/test_portfolio.py -v`
Expected: ImportError for `PortfolioCombiner` / `portfolio`.

- [ ] **Step 3: Implement the research combiner**

`research/expert-portfolio/src/portfolio.py`:
```python
"""Loop C: N-stream reliability-weighted log-linear combiner, and its locked evaluation."""

import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[2]
sys.path.insert(0, str(REPO / "research" / "expert-weights" / "src"))
from combine import Combiner  # noqa: E402

OUT = REPO / "outputs" / "t2-portfolio"


class PortfolioLogLinear(Combiner):
    """z = Σ_s w_s·log p_s + (c0 + c1·(rel − 0.5))·log p_riemann + [0, b];
    experts = [*neural_streams, riemann_key, rel_key]. With one neural stream this is
    loop B's RelLogLinear (same initialisation and penalty)."""

    def init(self):
        S = len(self.experts) - 2
        th = {"w": torch.full((S,), 0.8, dtype=torch.float64),
              "c": torch.tensor([0.4, 0.0], dtype=torch.float64),
              "b": torch.zeros(1, dtype=torch.float64)}
        return th

    def penalty(self, theta, theta0):
        return sum(((theta[k] - theta0[k]) ** 2).sum() for k in theta)

    def combine(self, F, theta):
        *neural, cls, rel = self.experts
        z = sum(w * F[e] for w, e in zip(theta["w"], neural))
        z = z + (theta["c"][0] + theta["c"][1] * (F[rel] - 0.5)) * F[cls]
        z = z + torch.cat([torch.zeros_like(theta["b"]), theta["b"]])
        return torch.log_softmax(z, -1)
```

Loop B's `RelLogLinear` init is `w = [0.8, 0.4, 0.0]`, `b = 0`, and it uses the same summed
squared penalty. The one-stream equivalence test holds because `PortfolioLogLinear`'s init maps
onto it (`w = [0.8]`, `c = [0.4, 0.0]`) and the penalty is the identical sum over the same
numbers.

- [ ] **Step 4: Implement the inference combiner and flags in `submission.py`**

Add after `LogLinearCombiner`:
```python
class PortfolioCombiner(nn.Module):
    """N-stream reliability-weighted log-linear pooling (loop C):
    z = Σ_s w_s·log p_s + (c0 + c1·(rel − 0.5))·log p_riemann + class_bias."""

    def __init__(self, n_streams, n_people, n_classes):
        super().__init__()
        d = torch.float64
        self.register_buffer("w", torch.zeros(n_streams, dtype=d))
        self.register_buffer("c", torch.zeros(2, dtype=d))
        self.register_buffer("class_bias", torch.zeros(n_classes, dtype=d))
        self.register_buffer("rel", torch.zeros(n_people, dtype=d))

    def forward(self, neural, logp_riemann):
        z = sum(w * lp.to(self.w.dtype) for w, lp in zip(self.w, neural))
        wc = (self.c[0] + self.c[1] * (self.rel - 0.5))[None, :, None]
        return torch.log_softmax(z + wc * logp_riemann + self.class_bias, -1)
```

Change `FingerprintMixture.__init__`'s signature to add
`shallow=False, reve_encoder=None, n_reve_out=None, combiner="C3", sfreq=120.0`.
`submission.py` must stay self-contained, so it does not import `track2/models.py`; add this
module-level helper instead (same numbers as `models.shallow_kwargs`, pinned by a test below):
```python
_SHALLOW_REF = dict(filter_time_length=25, pool_time_length=75, pool_time_stride=15)


def _shallow(n_chans, n_classes, n_times, sfreq):
    from braindecode.models import ShallowFBCSPNet
    s = sfreq / 250.0
    kw = {k: max(1, int(round(v * s))) for k, v in _SHALLOW_REF.items()}
    return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times,
                           final_conv_length="auto", **kw)
```
(`track2/tests/test_models.py` already pins these numbers for `models.py`; Step 1 of this task
adds a test that the two agree.)

In `__init__`, after the EEGNet experts:
```python
        self.shallow_experts = nn.ModuleList(
            _shallow(n_chans, n_classes, n_times, sfreq) for _ in range(n_experts)
        ) if shallow else nn.ModuleList()
        self.reve = (ReveProbe(reve_encoder, n_experts, n_classes, n_chans, n_times, n_reve_out)
                     if reve_encoder is not None else None)
        self.combiner_kind = combiner
        if combiner == "portfolio":
            n_streams = 1 + int(shallow) + int(reve_encoder is not None)
            self.combiner = PortfolioCombiner(n_streams, n_experts, n_classes)
```
(the existing `riemann_n_times_out` branch keeps creating `RiemannExperts`, and creates
`LogLinearCombiner` only when `combiner == "C3"`).

Replace `expert_logp`:
```python
    def expert_logp(self, X):
        neural = [torch.stack([torch.log_softmax(e(X), 1) for e in self.experts], 1)]
        if len(self.shallow_experts):
            neural.append(torch.stack([torch.log_softmax(e(X), 1)
                                       for e in self.shallow_experts], 1))
        if self.reve is not None:
            neural.append(self.reve(X))
        if self.riemann is None:
            return neural[0]
        if self.combiner_kind == "C3":
            return self.combiner(neural[0], self.riemann(X))
        return self.combiner(neural, self.riemann(X))
```

In `build_model`:
- After the `n_classes` check, add:
  ```python
  if "sfreq" in config and "sfreq" in meta and abs(meta["sfreq"] - config["sfreq"]) > 1e-6:
      raise ValueError(f"sfreq {meta['sfreq']} != trained {config['sfreq']}")
  ```
- Build the encoder only when `config.get("reve_probe")`, from
  `load_reve_encoder(meta["submission_dir"])`.
- Pass `shallow=config.get("shallow_experts", False)`,
  `combiner=config.get("combiner", "C3")`, `n_reve_out=config.get("reve_n_out")` and
  `sfreq=config.get("sfreq", 120.0)`.
- Load the new state keys `"shallow_experts"` (list of state dicts), `"reve"` and
  `"combiner"` when present.

Add to `track2/tests/test_portfolio.py`:
```python
    def test_inline_shallow_matches_models(self):
        import submission
        from models import make_model
        for sf, n_times in ((120.0, 480), (250.0, 1000), (500.0, 2000)):
            a = submission._shallow(27, 2, n_times, sf)
            b = make_model("shallow", 27, 2, n_times, sf)
            self.assertEqual([p.shape for p in a.parameters()],
                             [p.shape for p in b.parameters()], sf)
```

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -W ignore -m unittest discover -s track2/tests -v`
Expected: all OK (the package-dependent tests run when `outputs/t2-integration/fb_c3/submission`
exists, which it does).

- [ ] **Step 6: Check the old packages still score as before**

Run:
```bash
OMP_NUM_THREADS=4 .venv/bin/python -W ignore - <<'EOF'
import sys, json, numpy as np, torch
sys.path[:0]=['external/2026-competition','track2']
from submission import build_model
from sklearn.metrics import balanced_accuracy_score as bal
d=np.load('data/experiments/dreyer_windows.npz',allow_pickle=True); h=(d['split']=='test')&(d['run']>=3)
X=torch.from_numpy(d['X'][h])
for sub in ['outputs/track2_dreyer_sim/submission/','outputs/t2-integration/fb_c3/submission/']:
    c=json.load(open(sub+'config.json')); s=torch.load(sub+'mixture.pt',weights_only=True)
    m=build_model({"ch_names":c["ch_names"],"n_times":480,"n_classes":2,"device":"cpu","sfreq":120.0},c,s)
    p=torch.cat([m.predict(X[i:i+256]) for i in range(0,len(X),256)]).numpy()
    print(sub, round(bal(d['y'][h],p),4))
EOF
```
Expected: `0.9008` and `0.9214`.

- [ ] **Step 7: Commit**

```bash
git branch --show-current
git add research/expert-portfolio/src/portfolio.py track2/submission.py track2/tests/test_portfolio.py
git commit -m "Add the N-stream portfolio combiner and config flags for ShallowFBCSPNet / REVE streams

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Prediction banks for the new streams (Dreyer xfit / test, BNCI)

**Files:**
- Create: `research/expert-portfolio/src/stream_bank.py`

**Interfaces:**
- Consumes:
  - `train_mixture.load_windows`, `train_mixture.fit` (signature
    `fit(model, X, y, *, lr, epochs, seed, bs, X_val=None, y_val=None, name="", log_every=10)`)
    and `logits`;
  - `models.make_model`;
  - `reve_parts.{resample_matrix, positions, embed, fit_head, fit_person_heads}`;
  - `submission.load_reve_encoder`;
  - BNCI loading via `experiments/fingerprint_tangermann/run.py:load_windows`.
- Produces files in `outputs/t2-portfolio/`:
  - `reve_emb_dreyer.npz` (`full (20792, 512)`, plus `early`, `late` for `split == "test"`
    windows only, with their indices `eval_idx`);
  - `reve_emb_bnci.npz` (`full`, `early`, `late` for all 5,184 windows);
  - `bank_xfit_{stream}_seed0.npz` with keys `{stream}`, `{stream}_early`, `{stream}_late`
    `(2520, 21, C)`, `y`, `subject`, `fold`;
  - `bank_test_{stream}_seed{s}.npz` with the same keys `(2520, 21, C)`;
  - `bnci_{stream}_seed{s}.npz` with keys `logp (n_te, C)`, `early`, `late`
    (own-person, like loop B's BNCI files);
  - here `stream ∈ {"reve", "shallow"}`.

- [ ] **Step 1: Write `stream_bank.py`**

```python
"""Loop C prediction banks for the new streams (REVE probe, ShallowFBCSPNet experts).

Window order and masks match loop B's banks exactly (asserted against
outputs/t2-expert-weights/bank_xfit_seed0.npz and bank_test_packaged.npz).

    python research/expert-portfolio/src/stream_bank.py embed --dataset dreyer
    python research/expert-portfolio/src/stream_bank.py dreyer --stream reve --bank xfit
    python research/expert-portfolio/src/stream_bank.py dreyer --stream shallow --bank xfit --seed 0
    python research/expert-portfolio/src/stream_bank.py dreyer --stream shallow --bank test --seed 1
    python research/expert-portfolio/src/stream_bank.py bnci --stream shallow --seed 3
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[2]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2"),
                str(REPO / "experiments" / "fingerprint_tangermann")]
OUT = REPO / "outputs" / "t2-portfolio"
LOOPB = REPO / "outputs" / "t2-expert-weights"
POS_FILE = OUT / "reve_positions" / "reve_positions.json"
EARLY_S = 1.25
LAM_POOL = 1e-3
LAM_PERSON_GRID = (0.01, 0.1, 1.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def masked(X, sfreq, part):
    """part='early': zero samples ≥ 1.25 s; 'late': zero samples < 1.25 s."""
    cut = int(round(EARLY_S * sfreq))
    Y = X.copy()
    if part == "early":
        Y[..., cut:] = 0
    else:
        Y[..., :cut] = 0
    return Y


def dreyer():
    from train_mixture import load_windows
    d = load_windows()
    people = sorted(np.unique(d["subject"][d["split"] == "test"]), key=int)
    lab = np.array([people.index(s) if s in people else -1 for s in d["subject"]])
    return d, people, lab


def folds(d, bank):
    """[(calib_mask, target_mask)] in loop B's order."""
    ev = d["split"] == "test"
    if bank == "xfit":
        return [(ev & (d["run"] < 3) & (d["run"] != r), ev & (d["run"] == r)) for r in range(3)]
    return [(ev & (d["run"] < 3), ev & (d["run"] >= 3))]


def check_alignment(y, subject, bank):
    ref = np.load(LOOPB / ("bank_xfit_seed0.npz" if bank == "xfit" else "bank_test_packaged.npz"),
                  allow_pickle=True)
    assert (ref["y"] == y).all() and (ref["subject"] == subject).all(), "window order differs"


# ------------------------------------------------------------------ REVE

def cmd_embed(args):
    from reve_parts import embed, positions, resample_matrix
    from submission import load_reve_encoder
    enc = load_reve_encoder(POS_FILE.parent)
    if args.dataset == "dreyer":
        d, _, _ = dreyer()
        X, chs, sf = d["X"], [str(c) for c in d["ch_names"]], float(d["sfreq"])
        sub = np.where(d["split"] == "test")[0]
    else:
        from run import load_windows
        d = load_windows()
        X, chs, sf = d["X"], [str(c) for c in d["ch_names"]], 120.0
        sub = np.arange(len(X))
    R, pos = resample_matrix(X.shape[-1], sf), positions(chs, POS_FILE)
    out = {"eval_idx": sub}
    log(f"embedding {len(X)} windows")
    out["full"] = embed(enc, X, R, pos)
    for part in ("early", "late"):
        out[part] = embed(enc, masked(X[sub], sf, part), R, pos)
    np.savez(OUT / f"reve_emb_{args.dataset}.npz", **out)
    log("saved")


def reve_predict(Z_tr, y_tr, Z_cal, y_cal, p_cal, K, C, lam_person, Z_tgt_list):
    from reve_parts import fit_head, fit_person_heads
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-6
    W0, b0 = fit_head((Z_tr - mu) / sd, y_tr, C, LAM_POOL)
    W, b = fit_person_heads((Z_cal - mu) / sd, y_cal, p_cal, K, C, W0, b0, lam_person)
    outs = []
    for Z in Z_tgt_list:
        logits = np.einsum("nd,kcd->nkc", (Z - mu) / sd, W) + b
        outs.append(torch.log_softmax(torch.as_tensor(logits), -1).numpy())
    return outs, (mu, sd, W0, b0, W, b)


def cmd_dreyer_reve(args):
    d, people, lab = dreyer()
    e = np.load(OUT / "reve_emb_dreyer.npz")
    pos_in_eval = {g: i for i, g in enumerate(e["eval_idx"])}
    K, C = len(people), int(d["y"].max()) + 1
    for lam in LAM_PERSON_GRID:
        parts = {k: [] for k in ("reve", "reve_early", "reve_late")}
        ys, ss, fs = [], [], []
        for f, (cal, tgt) in enumerate(folds(d, args.bank)):
            pool = (d["split"] == "train") | cal
            ti = np.where(tgt)[0]
            ei = [pos_in_eval[g] for g in ti]
            (lp, lpe, lpl), _ = reve_predict(
                e["full"][pool], d["y"][pool], e["full"][cal], d["y"][cal], lab[cal], K, C, lam,
                [e["full"][ti], e["early"][ei], e["late"][ei]])
            parts["reve"].append(lp); parts["reve_early"].append(lpe); parts["reve_late"].append(lpl)
            ys.append(d["y"][tgt]); ss.append(d["subject"][tgt]); fs.append(np.full(tgt.sum(), f))
        bank = {k: np.concatenate(v) for k, v in parts.items()}
        bank |= {"y": np.concatenate(ys), "subject": np.concatenate(ss), "fold": np.concatenate(fs)}
        check_alignment(bank["y"], bank["subject"], args.bank)
        np.savez(OUT / f"bank_{args.bank}_reve_lam{lam}.npz", **bank)
        log(f"saved REVE {args.bank} bank, lam_person={lam}")


# ------------------------------------------------------------------ ShallowFBCSPNet

def train_stream(X, y, pool, val, cal_sets, sfreq, seed, threads, epochs=100, ft_epochs=50):
    from models import make_model
    from train_mixture import fit
    torch.set_num_threads(threads)
    n_chans, n_times = X.shape[1:]
    C = int(y.max()) + 1
    pooled, info = fit(make_model("shallow", n_chans, C, n_times, sfreq), X[pool], y[pool],
                       lr=1e-3, epochs=epochs, seed=seed, bs=64, X_val=X[val], y_val=y[val],
                       name="shallow pooled", log_every=5)
    experts = [fit(copy.deepcopy(pooled), X[m], y[m], lr=1e-4, epochs=ft_epochs, seed=seed,
                   bs=32)[0] for m in cal_sets]
    return pooled, experts, info


def predict_all(experts, X):
    from train_mixture import logits
    return np.stack([torch.log_softmax(logits(e, X), 1).numpy() for e in experts], 1)


def cmd_dreyer_shallow(args):
    d, people, lab = dreyer()
    X, y, sf = d["X"], d["y"], float(d["sfreq"])
    parts = {k: [] for k in ("shallow", "shallow_early", "shallow_late")}
    ys, ss, fs = [], [], []
    for f, (cal, tgt) in enumerate(folds(d, args.bank)):
        pool = (d["split"] == "train") | cal
        _, experts, info = train_stream(X, y, pool, d["split"] == "val",
                                        [cal & (d["subject"] == s) for s in people], sf,
                                        args.seed, args.threads, args.epochs)
        log(f"fold {f}: {info}")
        Xt = X[tgt]
        parts["shallow"].append(predict_all(experts, Xt))
        parts["shallow_early"].append(predict_all(experts, masked(Xt, sf, "early")))
        parts["shallow_late"].append(predict_all(experts, masked(Xt, sf, "late")))
        ys.append(y[tgt]); ss.append(d["subject"][tgt]); fs.append(np.full(tgt.sum(), f))
    bank = {k: np.concatenate(v) for k, v in parts.items()}
    bank |= {"y": np.concatenate(ys), "subject": np.concatenate(ss), "fold": np.concatenate(fs)}
    check_alignment(bank["y"], bank["subject"], args.bank)
    np.savez(OUT / f"bank_{args.bank}_shallow_seed{args.seed}.npz", **bank)
    log("saved")


# ------------------------------------------------------------------ EEGNet shortcut diagnostic

def cmd_eegnet_shortcut(args):
    """Early/late-masked predictions of loop B's saved dev EEGNet experts (no retraining)."""
    from train_mixture import make_eegnet
    d, people, _ = dreyer()
    X, sf = d["X"], float(d["sfreq"])
    parts = {"eegnet_early": [], "eegnet_late": []}
    ys, ss = [], []
    for r, (_, tgt) in enumerate(folds(d, "xfit")):
        tag = "" if r == 2 else f"_hold{r}"
        sds = torch.load(LOOPB / f"experts_dev{tag}_seed0.pt", weights_only=True)["experts"]
        experts = []
        for sd in sds:
            e = make_eegnet(X.shape[1], int(d["y"].max()) + 1, X.shape[2])
            e.load_state_dict(sd)
            experts.append(e)
        for part in ("early", "late"):
            parts[f"eegnet_{part}"].append(predict_all(experts, masked(X[tgt], sf, part)))
        ys.append(d["y"][tgt]); ss.append(d["subject"][tgt])
    bank = {k: np.concatenate(v) for k, v in parts.items()}
    bank |= {"y": np.concatenate(ys), "subject": np.concatenate(ss)}
    check_alignment(bank["y"], bank["subject"], "xfit")
    np.savez(OUT / "bank_xfit_eegnet_shortcut.npz", **bank)
    log("saved")


# ------------------------------------------------------------------ BNCI

def cmd_bnci(args):
    from run import load_windows
    d = load_windows()
    subjects = np.unique(d["subject"])
    y = np.searchsorted(np.unique(d["task"]), d["task"])
    tr = (d["session"] == 0) & (d["run"] != 5)
    va = (d["session"] == 0) & (d["run"] == 5)
    te = d["session"] == 1
    C, ste = len(np.unique(y)), d["subject"][te]
    out = {}
    if args.stream == "shallow":
        _, experts, _ = train_stream(d["X"], y, tr, va, [tr & (d["subject"] == s) for s in subjects],
                                     120.0, args.seed, args.threads, epochs=150)
        idx = np.searchsorted(subjects, ste)
        n = np.arange(te.sum())
        for k, X in [("logp", d["X"][te]), ("early", masked(d["X"][te], 120.0, "early")),
                     ("late", masked(d["X"][te], 120.0, "late"))]:
            out[k] = predict_all(experts, X)[n, idx]
    else:
        e = np.load(OUT / "reve_emb_bnci.npz")
        K, p_tr = len(subjects), np.searchsorted(subjects, d["subject"][tr])
        lps, _ = reve_predict(e["full"][tr], y[tr], e["full"][tr], y[tr], p_tr, K, C,
                              args.lam_person, [e["full"][te], e["early"][te], e["late"][te]])
        idx, n = np.searchsorted(subjects, ste), np.arange(te.sum())
        out = {k: lp[n, idx] for k, lp in zip(("logp", "early", "late"), lps)}
    np.savez(OUT / f"bnci_{args.stream}_seed{args.seed}.npz", **out)
    log("saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "dreyer", "bnci", "eegnet-shortcut"])
    ap.add_argument("--dataset", choices=["dreyer", "bnci"], default="dreyer")
    ap.add_argument("--stream", choices=["reve", "shallow"], default="reve")
    ap.add_argument("--bank", choices=["xfit", "test"], default="xfit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lam-person", type=float, default=0.1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "embed":
        cmd_embed(args)
    elif args.cmd == "eegnet-shortcut":
        cmd_eegnet_shortcut(args)
    elif args.cmd == "dreyer":
        (cmd_dreyer_reve if args.stream == "reve" else cmd_dreyer_shallow)(args)
    else:
        cmd_bnci(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the ShallowFBCSPNet path (1 epoch) without overwriting real banks**

Run:
```bash
OMP_NUM_THREADS=4 .venv/bin/python -W ignore research/expert-portfolio/src/stream_bank.py \
    dreyer --stream shallow --bank xfit --seed 99 --epochs 1
```
Expected: three "fold" log lines, then "saved". `outputs/t2-portfolio/bank_xfit_shallow_seed99.npz`
has `shallow` of shape `(2520, 21, 2)` and alignment passes. Delete that smoke file afterwards
(`rm outputs/t2-portfolio/bank_xfit_shallow_seed99.npz`).

- [ ] **Step 3: Launch the long jobs** (REVE only if P0 was GO)

```bash
mkdir -p outputs/t2-portfolio/logs
OMP_NUM_THREADS=4 setsid nohup .venv/bin/python -W ignore research/expert-portfolio/src/stream_bank.py \
    embed --dataset dreyer > outputs/t2-portfolio/logs/embed_dreyer.log 2>&1 < /dev/null &
```
When it has finished, run in sequence (each in the background the same way):
- `eegnet-shortcut` (minutes; no training);
- `embed --dataset bnci`;
- `dreyer --stream reve --bank xfit`;
- `dreyer --stream shallow --bank xfit --seed 0` (about 3 × 1.5 h);
- `bnci --stream shallow --seed {0..4}` (5 runs).

Check with `ps -C python -o pid,args` and `tail` the logs.

- [ ] **Step 4: Commit the code (results come with Task 7)**

```bash
git branch --show-current
git add research/expert-portfolio/src/stream_bank.py
git commit -m "Add loop C prediction banks for the REVE probe and ShallowFBCSPNet streams

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Protocol 01 and the dev / BNCI evaluation (the ship decision)

**Files:**
- Create: `research/expert-portfolio/experiments/01-dev-eval/protocol.md`
- Modify: `research/expert-portfolio/src/portfolio.py` (add `evaluate()` and the CLI)
- Create: `research/expert-portfolio/experiments/01-dev-eval/results/` (written by the script)
- Create: `research/expert-portfolio/experiments/01-dev-eval/analysis.md`

**Interfaces:**
- Consumes:
  - Task 6 banks;
  - loop B: `bank_xfit_seed0.npz` (`eegnet`), `classical_xfit_ts_C0.1.npz` (`logp`, `rel_dev`),
    `bnci/eegnet_seed{s}.npz` (`eegnet`), `bnci/classical.npz` (`ts_C0.1`, `rel_ts_C0.1`);
  - `combine.eval_global(bank, make, experts) -> {"bal_acc", "nll", "per_person", "oof"}`.
- Produces: `results/dev.json` + `dev.md` (stream-alone, full combo, drop-one, CIs, shortcut),
  `results/bnci.json` + `bnci.md` (condition (b)), and the candidate list in `analysis.md`.

- [ ] **Step 1: Write and commit the protocol before running anything**

`research/expert-portfolio/experiments/01-dev-eval/protocol.md`:
```markdown
# Protocol 01: dev-bank and BNCI evaluation of the new streams

Locked before any new-stream bank is scored.

- **REVE per-person regularisation:** λ_person ∈ {0.01, 0.1, 1.0}, chosen on the dev bank by REVE
  stream-alone true-ID balanced accuracy (ties within 0.002 → larger λ). λ_pool = 1e-3 fixed.
  The chosen λ is used everywhere after (BNCI, test).
- **Streams:** E = EEGNet experts (loop B bank), T = Riemannian ts_C0.1 + reliability,
  R = REVE probe, S = ShallowFBCSPNet experts.
- **Combinations** (PortfolioLogLinear, leave-one-person-out): E+T (shipped), E+T+R, E+T+S,
  E+T+R+S. Drop-one ablations from E+T+R+S.
- **Ship rule:** protocol 00 (a) on the dev bank, (b) on BNCI with weights fitted on the full
  Dreyer dev bank and applied unchanged.
- **Shortcut diagnostic:** early-only and late-only balanced accuracy for E (loop B's saved dev
  experts, via `stream_bank.py eegnet-shortcut`), S and R; reported, not used for the ship
  decision.
```

```bash
git branch --show-current
git add research/expert-portfolio/experiments/01-dev-eval/protocol.md
git commit -m "research(protocol): loop C protocol 01 — dev-bank and BNCI evaluation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Add the evaluation to `portfolio.py`**

Append:
```python
import json  # noqa: E402

from combine import bal, eval_global, nll, own  # noqa: E402

LOOPB = REPO / "outputs" / "t2-expert-weights"
EXP = REPO / "research" / "expert-portfolio" / "experiments" / "01-dev-eval" / "results"


def boot_ci(d, seed=0, n=5000):
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def dev_bank(lam):
    b = dict(np.load(LOOPB / "bank_xfit_seed0.npz", allow_pickle=True))
    c = np.load(LOOPB / "classical_xfit_ts_C0.1.npz")
    b["ts"], b["rel"] = c["logp"], c["rel_dev"][b["fold"]][:, :, None]
    for name, f in [("reve", OUT / f"bank_xfit_reve_lam{lam}.npz"),
                    ("shallow", OUT / "bank_xfit_shallow_seed0.npz"),
                    ("eegnet", OUT / "bank_xfit_eegnet_shortcut.npz")]:   # eegnet_early/_late only
        if f.exists():
            s = np.load(f)
            assert (s["y"] == b["y"]).all()
            for k in s.files:
                if k.startswith(name):
                    b[k] = s[k]
    return b


def person_nll(y, t, lp, K):
    return np.array([nll(lp[t == k], y[t == k]) for k in range(K)])


def evaluate_dev(lam):
    b = dev_bank(lam)
    y, t, K = b["y"], b["true_idx"], len(b["people"])
    streams = [s for s in ("eegnet", "reve", "shallow") if s in b]
    res = {"alone": {}, "combo": {}, "drop": {}, "shortcut": {}}
    for s in streams + ["ts"]:
        F = own(b, [s])[s]
        res["alone"][s] = {"bal_acc": bal(y, F), "nll": nll(F, y)}
        for part in ("early", "late"):
            if f"{s}_{part}" in b:
                res["shortcut"].setdefault(s, {})[part] = bal(y, own(b, [f"{s}_{part}"])[f"{s}_{part}"])
    combos = {"E+T": ["eegnet"], "E+T+R": ["eegnet", "reve"], "E+T+S": ["eegnet", "shallow"],
              "E+T+R+S": ["eegnet", "reve", "shallow"]}
    oofs = {}
    for name, neural in combos.items():
        if not all(s in b for s in neural):
            continue
        r = eval_global(b, PortfolioLogLinear, [*neural, "ts", "rel"])
        oofs[name] = r
        res["combo"][name] = {"bal_acc": r["bal_acc"], "nll": r["nll"]}
    full = "E+T+R+S" if "E+T+R+S" in oofs else max(oofs, key=lambda k: len(k))
    for drop, name in [("reve", full.replace("+R", "")), ("shallow", full.replace("+S", ""))]:
        if name == full or name not in oofs:
            continue
        a, z = oofs[full], oofs[name]
        d_acc = a["per_person"] - z["per_person"]
        d_nll = person_nll(y, t, z["oof"], K) - person_nll(y, t, a["oof"], K)
        res["drop"][drop] = {"acc_cost": a["bal_acc"] - z["bal_acc"], "acc_ci": boot_ci(d_acc),
                             "nll_cost": z["nll"] - a["nll"], "nll_ci": boot_ci(d_nll)}
        crit_a = res["drop"][drop]["acc_cost"] >= 0.005
        crit_n = res["drop"][drop]["nll_cost"] >= 0.005 and res["drop"][drop]["nll_ci"][1] > 0
        res["drop"][drop]["passes_a"] = bool(crit_a or crit_n)
    return res, b


def evaluate_bnci(lam_unused, dev):
    """Condition (b): Dreyer-fitted weights, applied to BNCI's own-person streams."""
    cl = np.load(LOOPB / "bnci" / "classical.npz")
    from run import load_windows  # experiments/fingerprint_tangermann
    d = load_windows()
    y = np.searchsorted(np.unique(d["task"]), d["task"])[d["session"] == 1]
    subj = d["subject"][d["session"] == 1]
    C = len(np.unique(y))
    rel = cl["rel_ts_C0.1"]
    rel_n = 0.5 + 0.5 * (rel - 1 / C) / (1 - 1 / C)
    rel_w = rel_n[np.searchsorted(np.unique(subj), subj)][:, None]
    out = {}
    for seed in range(5):
        F = {"eegnet": np.load(LOOPB / "bnci" / f"eegnet_seed{seed}.npz")["eegnet"],
             "ts": cl["ts_C0.1"], "rel": rel_w}
        for s in ("reve", "shallow"):
            f = OUT / f"bnci_{s}_seed{seed if s == 'shallow' else 0}.npz"
            if f.exists():
                F[s] = np.load(f)["logp"]
        row = {}
        for name, neural in {"E+T": ["eegnet"], "E+T+R": ["eegnet", "reve"],
                             "E+T+S": ["eegnet", "shallow"]}.items():
            if not all(s in F for s in neural):
                continue
            comb = PortfolioLogLinear([*neural, "ts", "rel"]).fit(own(dev, [*neural, "ts", "rel"]),
                                                                  dev["y"])
            if C != 2:
                comb.theta["b"] = torch.zeros(C - 1, dtype=torch.float64)
            row[name] = bal(y, comb.predict({k: F[k] for k in [*neural, "ts", "rel"]}))
        out[seed] = row
    return out


def main():
    import argparse
    sys.path.insert(0, str(REPO / "experiments" / "fingerprint_tangermann"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    EXP.mkdir(parents=True, exist_ok=True)
    res, dev = evaluate_dev(a.lam)
    (EXP / "dev.json").write_text(json.dumps(res, indent=2))
    bn = evaluate_bnci(a.lam, dev)
    (EXP / "bnci.json").write_text(json.dumps({str(k): v for k, v in bn.items()}, indent=2))
    print(json.dumps(res, indent=2)); print(json.dumps(bn, indent=2))


if __name__ == "__main__":
    main()
```

Note on BNCI with C = 4: Dreyer's fit has a 2-class bias, so it is zeroed for BNCI (it was
≈ 0 in loop B, −0.016).

- [ ] **Step 3: Choose λ_person (protocol 01 rule)**

Run:
```bash
.venv/bin/python -W ignore - <<'EOF'
import sys; sys.path.insert(0,'research/expert-portfolio/src')
from portfolio import dev_bank, own, bal
for lam in (0.01, 0.1, 1.0):
    b = dev_bank(lam); print(lam, round(bal(b["y"], own(b,["reve"])["reve"]), 4))
EOF
```
Pick by the rule; record the choice in `research-log.md`.

- [ ] **Step 4: Run the evaluation**

Prerequisite: the BNCI REVE file `bnci_reve_seed0.npz` exists. If not, generate it with
`stream_bank.py bnci --stream reve --lam-person <chosen>`.

Run:
`OMP_NUM_THREADS=2 .venv/bin/python -W ignore research/expert-portfolio/src/portfolio.py --lam <chosen>`

Expected: `dev.json` with `alone`, `combo`, `drop.{reve,shallow}.passes_a`, and `shortcut`;
`bnci.json` with per-seed E+T, E+T+R and E+T+S balanced accuracies.

- [ ] **Step 5: Write `analysis.md`**

Write it from the JSONs:
- the predictions vs outcomes;
- condition (a) per stream (`passes_a`);
- condition (b) per stream: mean over seeds of (E+T+X − E+T) ≥ −0.005;
- the shortcut readout;
- **the candidate list** (streams passing both).

If none pass, the finding is "neither new stream earns weight"; Task 8 is still built (sim2
support), Task 9 runs its E+T part only, and Task 10 is skipped.

Write the shipped combination's coefficients, in stream order [EEGNet, Shallow, REVE] with only
the shipped streams, fitted on the full dev bank:
```bash
.venv/bin/python -W ignore - <<'EOF'
import json, sys; sys.path.insert(0, 'research/expert-portfolio/src')
from portfolio import PortfolioLogLinear, dev_bank, own, OUT
LAM = 0.1                                   # the λ chosen in Step 3
NEURAL = ["eegnet", "shallow", "reve"]      # keep only the shipped streams, in this order
dev = dev_bank(LAM); ex = [*NEURAL, "ts", "rel"]
coef = PortfolioLogLinear(ex).fit(own(dev, ex), dev["y"]).describe()
(OUT / "portfolio_coef.json").write_text(json.dumps(coef)); print(coef)
EOF
```

- [ ] **Step 6: Commit results**

```bash
git branch --show-current
git add research/expert-portfolio
git commit -m "research(results): loop C protocol 01 — <candidates or none>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Packaging flags for the new streams, and sim2 support in `train_mixture.py`

**Files:**
- Modify: `track2/train_mixture.py`
- Modify: `track2/tests/test_portfolio.py` (mask helper test)

**Interfaces:**
- Consumes:
  - `stream_bank.train_stream(X, y, pool, val, cal_sets, sfreq, seed, threads, epochs=100, ft_epochs=50) -> (pooled, experts, info)`;
  - `reve_parts.*` and `submission.load_reve_encoder(positions_dir)`;
  - the combiner coefficients JSON from Task 7: `{"w": [...], "c": [c0, c1], "b": [b1]}`, stream
    order [EEGNet, Shallow, REVE] (only the streams present).
- Produces:
  - `train_mixture.masks(d, eval_people=None) -> (train_mask, calib, hidden, people)`;
  - flags `--eval-people FILE`, `--shallow-experts`, `--reve-probe`, `--reve-lam`,
    `--reve-emb-cache FILE`, `--combiner {C3,portfolio}`, `--portfolio-coef FILE`.
  - Results tags gain `_sim2`, `_shallow`, `_reve` and `_portfolio` suffixes.
  - In portfolio mode, the score table also reports the C3 two-stream rows, computed from the
    same experts, so every run compares like with like.

- [ ] **Step 1: Write the failing mask test** (append to `track2/tests/test_portfolio.py`)

```python
    def test_masks_default_and_sim2(self):
        import numpy as np
        from train_mixture import masks
        d = {"subject": np.array(["1", "1", "61", "61", "2", "2"]),
             "run": np.array([0, 4, 0, 4, 1, 5]),
             "split": np.array(["train", "train", "test", "test", "train", "train"])}
        tr, cal, hid, people = masks(d)
        self.assertEqual(people, ["61"])
        self.assertEqual(tr.tolist(), [True, True, False, False, True, True])
        self.assertEqual(cal.tolist(), [False, False, True, False, False, False])
        tr, cal, hid, people = masks(d, eval_people=["2"])
        self.assertEqual(people, ["2"])
        self.assertEqual(tr.tolist(), [True, True, True, True, False, False])  # 61 becomes training
        self.assertEqual(cal.tolist(), [False, False, False, False, True, False])
        self.assertEqual(hid.tolist(), [False, False, False, False, False, True])
```

Run: `.venv/bin/python -W ignore -m unittest track2/tests/test_portfolio.py -v`
Expected: FAIL with `ImportError: cannot import name 'masks'`.

- [ ] **Step 2: Implement `masks` and use it in `main()`**

Add to `track2/train_mixture.py` (above `main`):
```python
def masks(d, eval_people=None):
    """Training / calibration / hidden masks.

    Default (the original simulation): the kit's test people are the evaluation people.
    sim2 (`eval_people` given): those people are the evaluation people, and every other
    train- or test-split person is an ordinary training person (all six runs)."""
    subj, run, split = d["subject"], d["run"], d["split"]
    if eval_people is None:
        is_eval = split == "test"
        train = split == "train"
    else:
        is_eval = np.isin(subj, list(eval_people))
        train = np.isin(split, ["train", "test"]) & ~is_eval
    calib = is_eval & (run < N_CALIB_RUNS)
    hidden = is_eval & (run >= N_CALIB_RUNS)
    people = sorted(np.unique(subj[is_eval]), key=int)
    return train, calib, hidden, people
```

In `main()`, replace the lines that build `is_eval`, `calib`, `hidden` and `people` with:
```python
    eval_people = (json.loads(Path(args.eval_people).read_text())["people"]
                   if args.eval_people else None)
    if eval_people is not None and args.reuse_eegnet:
        raise SystemExit("--reuse-eegnet experts belong to the original evaluation people")
    train_mask, calib, hidden, people = masks(d, eval_people)
```
Then replace every `(split == "train") | calib` with `train_mask | calib` (the pooled EEGNet's
data), and log `train_mask.sum()` instead of `(split == 'train').sum()`.

- [ ] **Step 3: Add the stream blocks** (after the Riemannian block)

```python
    extra_files = []
    if args.shallow_experts:
        log("ShallowFBCSPNet pooled + per-person fine-tunes")
        sys.path.insert(0, str(REPO / "research" / "expert-portfolio" / "src"))
        from stream_bank import train_stream
        _, sh, info_sh = train_stream(X, y, train_mask | calib, split == "val",
                                      [calib & (subj == s) for s in people], sfreq, args.seed,
                                      args.threads)
        state["shallow_experts"] = [e.state_dict() for e in sh]
        config_extra["shallow_experts"] = True
    if args.reve_probe:
        log("REVE probe heads")
        import reve_parts
        from submission import load_reve_encoder
        pos_dir = REPO / "outputs" / "t2-portfolio" / "reve_positions"
        R = reve_parts.resample_matrix(n_times, sfreq)
        pos = reve_parts.positions(ch_names, pos_dir / "reve_positions.json")
        if args.reve_emb_cache:
            Z = np.load(args.reve_emb_cache)["full"]
            assert len(Z) == len(X), "embedding cache does not match the windows"
        else:
            Z = reve_parts.embed(load_reve_encoder(pos_dir), X, R, pos)
        pool = train_mask | calib
        mu, sd = Z[pool].mean(0), Z[pool].std(0) + 1e-6
        W0, b0 = reve_parts.fit_head((Z[pool] - mu) / sd, y[pool], n_classes, 1e-3)
        W, b = reve_parts.fit_person_heads((Z[calib] - mu) / sd, y[calib], lab[calib], K,
                                           n_classes, W0, b0, args.reve_lam)
        state["reve"] = reve_parts.export(R, pos, mu, sd, W, b)
        config_extra |= {"reve_probe": True, "reve_n_out": int(R.shape[1])}
        extra_files += [pos_dir / "reve_positions.json", pos_dir / "reve_kwargs.json"]
    if args.combiner == "portfolio":
        coef = json.loads(Path(args.portfolio_coef).read_text())
        n_streams = 1 + int(args.shallow_experts) + int(args.reve_probe)
        assert len(coef["w"]) == n_streams, "coefficients don't match the enabled streams"
        b_full = [0.0, *coef["b"]] if n_classes == 2 else [0.0] * n_classes
        state["combiner"] = {"w": torch.tensor(coef["w"], dtype=torch.float64),
                             "c": torch.tensor(coef["c"], dtype=torch.float64),
                             "class_bias": torch.tensor(b_full, dtype=torch.float64),
                             "rel": state["combiner"]["rel"]}
        config_extra |= {"combiner": "portfolio", "combiner_coef": coef}
```
After `sub_dir` is created, copy the extra files:
`for f in extra_files: shutil.copyfile(f, sub_dir / f.name)`.

Add the argparse flags:
```python
    ap.add_argument("--eval-people", default=None, help="JSON with a 'people' list (sim2)")
    ap.add_argument("--shallow-experts", action="store_true")
    ap.add_argument("--reve-probe", action="store_true")
    ap.add_argument("--reve-lam", type=float, default=0.1)
    ap.add_argument("--reve-emb-cache", default=None, help="outputs/t2-portfolio/reve_emb_dreyer.npz")
    ap.add_argument("--combiner", choices=["C3", "portfolio"], default="C3")
    ap.add_argument("--portfolio-coef", default=None)
```

- [ ] **Step 4: Report the two-stream C3 rows next to the portfolio rows**

In the scoring section, after `lp_all` is computed, add:
```python
    if args.combiner == "portfolio":
        from submission import LogLinearCombiner
        c3 = LogLinearCombiner(K, n_classes)
        c3.coef.copy_(torch.tensor([riemann_parts.C3["a"], riemann_parts.C3["c0"],
                                    riemann_parts.C3["c1"]], dtype=torch.float64))
        if n_classes == 2:
            c3.class_bias[1] = riemann_parts.C3["b"]
        c3.rel.copy_(mixture.combiner.rel)
        with torch.inference_mode():
            lp_c3 = c3(lp_eeg, mixture.riemann(Xt)).double()
        preds |= {"E+T (C3), oracle": lp_c3[n, true_idx].argmax(1).numpy(),
                  "E+T (C3), soft": soft(lp_c3)}
```
Rename the existing `"+ Riemannian (C3)"` keys to `"full combination"` when
`args.combiner == "portfolio"`. Extend the results tag:
```python
    tag += ("_sim2" if args.eval_people else "") + ("_shallow" if args.shallow_experts else "") \
        + ("_reve" if args.reve_probe else "") + ("_portfolio" if args.combiner == "portfolio" else "")
```
and the output directory default to `args.out / ("sim2" if args.eval_people else "")`.

- [ ] **Step 5: Tests and a smoke run**

Run: `.venv/bin/python -W ignore -m unittest discover -s track2/tests -v`
Expected: all OK.

Smoke the sim2 path with tiny budgets:
```bash
OMP_NUM_THREADS=4 .venv/bin/python -W ignore track2/train_mixture.py --threads 4 --seed 99 \
    --epochs 1 --ft-epochs 1 \
    --eval-people research/expert-portfolio/experiments/00-protocol/sim2_people.json \
    --out outputs/t2-portfolio/smoke
```
Expected:
- the log shows 14 evaluation people;
- calibration is 14 × 120 = 1,680 windows and hidden is 1,680;
- the score table prints (numbers are meaningless at 1 epoch).

Delete `outputs/t2-portfolio/smoke` and the smoke results files in `track2/results/`.

- [ ] **Step 6: Commit**

```bash
git branch --show-current
git add track2/train_mixture.py track2/tests/test_portfolio.py
git commit -m "train_mixture: package ShallowFBCSPNet / REVE streams with the portfolio combiner; sim2 masks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Honest confirmation, sim2 (decides nothing) and the R4–R6 report (upper bounds)

Run if Task 7 named at least one shipping stream. Also run the E+T part regardless, because it
gives the shipped design its first honest Dreyer estimate.

**Files:**
- Create: `research/expert-portfolio/experiments/02-confirm/protocol.md`
- Create: `research/expert-portfolio/experiments/02-confirm/r4r6.py`
- Create: `research/expert-portfolio/experiments/02-confirm/analysis.md`

**Interfaces:**
- Consumes:
  - Task 8's `train_mixture.py` flags;
  - Task 7's coefficients `outputs/t2-portfolio/portfolio_coef.json` (from
    `PortfolioLogLinear([...]).fit(own(dev, ex), dev["y"]).describe()`, written by Task 7);
  - loop B's test banks.
- Produces:
  - `track2/results/dreyer_sim_seed{0,1,2}_fb_riemann_c3_sim2*_portfolio.{md,json}`;
  - `02-confirm/results_r4r6.json`.

- [ ] **Step 1: Write and commit the protocol before any sim2 or R4–R6 scoring**

`research/expert-portfolio/experiments/02-confirm/protocol.md`:
```markdown
# Protocol 02: honest confirmation (sim2) and the R4–R6 report

Locked before any sim2 model is trained.

- **Streams shipped by Task 7:** <list>. Combiner coefficients: `outputs/t2-portfolio/portfolio_coef.json`
  (fitted on the dev bank), transferred unchanged.
- **sim2:**
  - The people in `00-protocol/sim2_people.json`; seeds 0, 1, 2.
  - The full pipeline is trained from scratch (`train_mixture.py --eval-people …
    --shallow-experts/--reve-probe … --combiner portfolio`).
  - Reported rows: EEGNet experts, E+T (C3), and the full combination, under oracle and
    soft routing.
- **R4–R6:** seeds 0–2 (loop B's test banks plus this loop's test banks); E+T vs the full
  combination, oracle. **Upper bounds.**
- **Veto (the only decision here):** withdraw a stream if the full combination is more than 0.02
  below E+T on sim2 (mean over seeds) or on R4–R6 (mean over seeds).
- Everything else is reported, not acted on.
```
Commit it as `research(protocol): loop C protocol 02 — sim2 confirmation and R4–R6 report`.

- [ ] **Step 2: Run sim2, seeds 0–2, in the background**

Run one seed at a time, about 3 h each. With the REVE cache present, embedding is skipped.
```bash
for s in 0 1 2; do
  OMP_NUM_THREADS=4 .venv/bin/python -W ignore track2/train_mixture.py --threads 4 --seed $s \
      --eval-people research/expert-portfolio/experiments/00-protocol/sim2_people.json \
      <--shallow-experts> <--reve-probe --reve-lam λ --reve-emb-cache outputs/t2-portfolio/reve_emb_dreyer.npz> \
      --combiner portfolio --portfolio-coef outputs/t2-portfolio/portfolio_coef.json \
      --out outputs/t2-portfolio/sim2_seed$s
done > outputs/t2-portfolio/logs/sim2.log 2>&1
```
Launch the loop itself with `setsid nohup bash -c '…' < /dev/null &`.

- [ ] **Step 3: The R4–R6 report**

`research/expert-portfolio/experiments/02-confirm/r4r6.py`:
```python
"""Report-only: E+T vs the full combination on R4–R6, seeds 0–2 (upper bounds)."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research" / "expert-portfolio" / "src"))
from portfolio import LOOPB, OUT, PortfolioLogLinear, bal, dev_bank, nll, own  # noqa: E402

LAM = float(sys.argv[1]); STREAMS = sys.argv[2].split(",")       # e.g. 0.1 shallow,reve
dev = dev_bank(LAM)
c = np.load(LOOPB / "classical_xfit_ts_C0.1.npz")
res = {}
for seed, name in [(0, "bank_test_packaged.npz"), (1, "bank_test_seed1.npz"), (2, "bank_test_seed2.npz")]:
    t = dict(np.load(LOOPB / name, allow_pickle=True))
    t["ts"] = np.load(LOOPB / "classical_test_ts_C0.1.npy")
    t["rel"] = np.broadcast_to(c["rel_test"][None, :, None], t["eegnet"].shape[:2] + (1,)).copy()
    if "shallow" in STREAMS:
        t["shallow"] = np.load(OUT / f"bank_test_shallow_seed{seed}.npz")["shallow"]
    if "reve" in STREAMS:
        t["reve"] = np.load(OUT / f"bank_test_reve_lam{LAM}.npz")["reve"]
    row = {}
    for label, neural in {"E+T": ["eegnet"], "full": ["eegnet", *STREAMS]}.items():
        ex = [*neural, "ts", "rel"]
        lp = PortfolioLogLinear(ex).fit(own(dev, ex), dev["y"]).predict(own(t, ex))
        row[label] = {"oracle": bal(t["y"], lp), "nll": nll(lp, t["y"])}
    res[seed] = row
(Path(__file__).parent / "results_r4r6.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
```

Before running it, generate this loop's test banks:
- `stream_bank.py dreyer --stream shallow --bank test --seed {0,1,2}`;
- `stream_bank.py dreyer --stream reve --bank test`.

Then run: `.venv/bin/python -W ignore research/expert-portfolio/experiments/02-confirm/r4r6.py <λ> <streams>`

- [ ] **Step 4: Apply the veto and write `analysis.md`**

Collect from the sim2 result JSONs (`scores["E+T (C3), oracle"]`, `scores["full combination, oracle"]`)
and `results_r4r6.json`:
- per seed and the mean;
- the veto outcome;
- **the honest Dreyer estimates:** sim2 E+T and full, labelled as such. R4–R6 rows are labelled
  "upper bound".

Commit as `research(results): loop C protocol 02 — sim2 <numbers>; R4–R6 upper bounds; veto <none|stream>`.

---

### Task 10: Package the shipped streams and pass the contract check offline

Run if Task 9 left at least one stream un-vetoed.

**Files:**
- Modify: `track2/README.md`
- Create: `track2/results/dreyer_sim_seed0_*_portfolio.md` (written by `train_mixture.py`)

- [ ] **Step 1: Build the original-simulation package (seed 0, packaged EEGNet experts reused)**

```bash
OMP_NUM_THREADS=4 .venv/bin/python -W ignore track2/train_mixture.py --threads 4 \
    --reuse-eegnet outputs/track2_dreyer_sim/submission/mixture.pt \
    --out outputs/t2-portfolio/package <--shallow-experts> \
    <--reve-probe --reve-lam λ --reve-emb-cache outputs/t2-portfolio/reve_emb_dreyer.npz> \
    --combiner portfolio --portfolio-coef outputs/t2-portfolio/portfolio_coef.json
```
Expected: `shipped mixture (predict)` equals `full combination, soft` exactly, and is within 0.002
of `full combination, oracle` (fingerprint 0.996).

- [ ] **Step 2: Contract check, offline**

```bash
U=$PWD/outputs/t2-portfolio/package/unzipped; rm -rf $U; mkdir -p $U
(cd $U && unzip -q ../track2_dreyer_sim.zip && chmod -R a-w .)
cd external/2026-competition
export BENCHOPT_DATA_HOME="$(realpath ../../data)" PYTHONWARNINGS=ignore::FutureWarning \
       MNE_LOGGING_LEVEL=ERROR TQDM_DISABLE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
COMPET_SUBMISSION_DIR=$U ../../.venv/bin/benchopt run tracks/bci_decoding \
    -d "BCI[study=dreyer2023]" -s $U/submission.py
```
Expected: exit 0, `FingerprintMixture: done`. Record the wall time and ZIP size.

- [ ] **Step 3: Update `track2/README.md`**

- **Current best model:** a step 7 for the shipped stream(s), with their coefficients.
- **Evidence table:**
  - dev bank (drop-one), BNCI (condition (b)) and **sim2 (honest)** rows;
  - R4–R6 rows marked "upper bound (reused hidden runs)";
  - the existing loop A/B R4–R6 rows get the same marker.
- **Contract row:** time, ZIP size, `HF_HUB_OFFLINE=1`.
- **Files table:** `models.py`, `reve_parts.py`, `tests/`.
- **Before the sealed phase:**
  - Rule 04 declaration of REVE (pretraining data: 92 datasets, 60k hours; frozen; staged on
    the worker);
  - the sizes at 500 Hz, 4 s, computed: `reve_parts.resample_matrix(2000, 500.0).nbytes`
    (float32 in the package) and the ShallowFBCSPNet state size per person at `sfreq=500`.

- [ ] **Step 4: Tests and commit**

```bash
.venv/bin/python -W ignore -m unittest discover -s track2/tests -v
git branch --show-current
git add track2
git commit -m "Ship <streams> in the Track 2 submission; contract check passes offline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Leakage-free benchopt comparison page

Independent of Tasks 7–10; needs Tasks 3–6 (REVE embeddings from Task 6).

**Files:**
- Create: `track2/bench/shallow_solver.py`
- Create: `track2/bench/reve_solver.py` (skip if P0 NO-GO)
- Create: `track2/bench/riemann_solver.py`
- Create: `track2/bench/train_bench.py`
- Create: `track2/bench/README.md`

**Interfaces:**
- Consumes: the kit's solver contract. Without `COMPET_SUBMISSION_DIR`, each solver's
  `meta["submission_dir"]` is `tracks/bci_decoding/outputs/<Solver.name>/`
  (`benchmark_utils/base_solver.py:set_objective`).
- Produces: weights in those folders (training split only) and one HTML page from a single
  `benchopt run` with all solvers.

- [ ] **Step 1: Write the page solvers**

`track2/bench/shallow_solver.py`:
```python
"""Benchmark-page solver: pooled ShallowFBCSPNet trained on the warm-up train split only."""
import json
import torch
from braindecode.models import ShallowFBCSPNet
from benchmark_utils.base_solver import CompetSolver

_REF = dict(filter_time_length=25, pool_time_length=75, pool_time_stride=15)


class _Model(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self.net(X.float()).argmax(1)


class Solver(CompetSolver):
    name = "ShallowFBCSPNet-pooled"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        cfg = json.loads((meta["submission_dir"] / "config.json").read_text())
        s = meta["sfreq"] / 250.0
        net = ShallowFBCSPNet(n_chans=meta["n_chans"], n_outputs=meta["n_classes"],
                              n_times=meta["n_times"], final_conv_length="auto",
                              **{k: max(1, int(round(v * s))) for k, v in _REF.items()})
        net.load_state_dict(torch.load(meta["submission_dir"] / "model.pt",
                                       map_location=meta["device"], weights_only=True))
        assert cfg["ch_names"] == list(meta["ch_names"]), "channel order differs"
        return _Model(net).to(meta["device"]).eval()
```

`track2/bench/riemann_solver.py` follows the same pattern and wraps a pooled filter-bank tangent
space classifier:
- `submission.FBFingerprint` reused with `n_subjects = n_classes`;
- weights from `riemann_parts.fit_fingerprint(X_train, y_train, sfreq)`, i.e. classes instead of
  people;
- `name = "Riemann-FB-pooled"`.

```python
"""Benchmark-page solver: pooled filter-bank tangent-space classifier (train split only)."""
import json
import torch
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "Riemann-FB-pooled"
    requirements = []

    def load_model(self, meta):
        import importlib.util
        sub = meta["submission_dir"]
        spec = importlib.util.spec_from_file_location("track2_submission", sub / "submission_lib.py")
        lib = importlib.util.module_from_spec(spec); spec.loader.exec_module(lib)
        cfg = json.loads((sub / "config.json").read_text())
        m = lib.FBFingerprint(cfg["n_bands"], meta["n_chans"], meta["n_times"], meta["n_classes"])
        m.load_state_dict(torch.load(sub / "model.pt", map_location=meta["device"], weights_only=True))

        class _M(torch.nn.Module):
            def __init__(self, f):
                super().__init__(); self.f = f

            @torch.inference_mode()
            def predict(self, X):
                return self.f(X).argmax(1)
        return _M(m).to(meta["device"]).eval()
```

`track2/bench/reve_solver.py` has the same structure:
- `name = "REVE-probe-pooled"`;
- builds `lib.ReveProbe(lib.load_reve_encoder(sub), n_people=1, …)` and loads `model.pt`;
- `predict` returns `self.p(X)[:, 0].argmax(-1)`.

`train_bench.py` copies `track2/submission.py` into each solver folder as `submission_lib.py`
(it holds `FBFingerprint`, `ReveProbe` and `load_reve_encoder`).

- [ ] **Step 2: Write `train_bench.py`**

```python
"""Train the page solvers on the warm-up train split (val for epochs) and write their folders.

    python track2/bench/train_bench.py --threads 4
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2"),
                str(REPO / "research" / "expert-portfolio" / "src")]
KIT_OUT = REPO / "external" / "2026-competition" / "tracks" / "bci_decoding" / "outputs"


def folder(name, ch_names, extra=None):
    d = KIT_OUT / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "track2" / "submission.py", d / "submission_lib.py")
    (d / "config.json").write_text(json.dumps({"ch_names": ch_names, **(extra or {})}))
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-reve", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from train_mixture import fit, load_windows
    from models import make_model
    import riemann_parts
    d = load_windows()
    X, y, split = d["X"], d["y"], d["split"]
    chs, sf = [str(c) for c in d["ch_names"]], float(d["sfreq"])
    tr, va = split == "train", split == "val"            # no evaluation-people labels
    C, (n_chans, n_times) = int(y.max()) + 1, X.shape[1:]
    net, _ = fit(make_model("shallow", n_chans, C, n_times, sf), X[tr], y[tr], lr=1e-3,
                 epochs=100, seed=0, bs=64, X_val=X[va], y_val=y[va], name="shallow", log_every=10)
    torch.save(net.state_dict(), folder("ShallowFBCSPNet-pooled", chs) / "model.pt")
    st = riemann_parts.fit_fingerprint(X[tr], y[tr], sf)
    torch.save(st, folder("Riemann-FB-pooled", chs, {"n_bands": len(riemann_parts.FP_BANDS)}) / "model.pt")
    if not a.skip_reve:
        import reve_parts
        from submission import load_reve_encoder
        pos_dir = REPO / "outputs" / "t2-portfolio" / "reve_positions"
        e = np.load(REPO / "outputs" / "t2-portfolio" / "reve_emb_dreyer.npz")["full"]
        mu, sd = e[tr].mean(0), e[tr].std(0) + 1e-6
        W, b = reve_parts.fit_head((e[tr] - mu) / sd, y[tr], C, 1e-3)
        st = reve_parts.export(reve_parts.resample_matrix(n_times, sf),
                               reve_parts.positions(chs, pos_dir / "reve_positions.json"),
                               mu, sd, W[None], b[None])
        dd = folder("REVE-probe-pooled", chs, {"n_out": 800})
        torch.save(st, dd / "model.pt")
        for f in ("reve_positions.json", "reve_kwargs.json"):
            shutil.copyfile(pos_dir / f, dd / f)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Train the kit baselines and our page solvers**

```bash
OMP_NUM_THREADS=4 .venv/bin/python -W ignore track2/bench/train_bench.py --threads 4
cd external/2026-competition
export BENCHOPT_DATA_HOME="$(realpath ../../data)" PYTHONWARNINGS=ignore::FutureWarning MNE_LOGGING_LEVEL=ERROR TQDM_DISABLE=1
../../.venv/bin/benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
    -o "BCI-decoding[training=True]" -s MeanLogReg -s EEGNet
```
Expected: `outputs/MeanLogReg/` and `outputs/EEGNet/` written by the kit's own training (train
split only).

- [ ] **Step 4: Render the page (inference-only, all solvers)**

```bash
unset COMPET_SUBMISSION_DIR
../../.venv/bin/benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
    -s Constant -s MeanLogReg -s EEGNet \
    -s ../../track2/bench/shallow_solver.py -s ../../track2/bench/riemann_solver.py \
    -s ../../track2/bench/reve_solver.py
```
Expected:
- one HTML under `tracks/bci_decoding/outputs/` listing six solvers;
- every score is a warm-up test score with no calibration leakage, so these numbers are
  legitimate warm-up estimates.

If benchopt rejects multiple file-path `-s` values, copy the three solver files into
`tracks/bci_decoding/solvers/` (inside the git-ignored kit clone) and refer to them by name.
Record the variant used in `track2/bench/README.md`.

- [ ] **Step 5: Document and commit**

`track2/bench/README.md`:
- the commands above;
- the table of scores (balanced accuracy per solver, from the parquet);
- the note that this page is leakage-free, while the calibrated mixture's 0.947 is not.

```bash
git branch --show-current
git add track2/bench
git commit -m "Add a leakage-free benchopt comparison page for Track 2 (6 solvers)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Findings, report, PR

**Files:**
- Modify: `research/expert-portfolio/{findings.md, research-log.md, research-state.yaml}`
- Create: `research/expert-portfolio/to_human/report-001.html` (reuse the pattern of
  `research/expert-weights/src/report.py`: inline SVG, light/dark tokens, table views)
- Modify: `README.md` (experiments table row for loop C)

- [ ] **Step 1:** Update findings with each stream's alone, combination, drop-one, BNCI,
  shortcut and checkpoint results, and the ship decision with its reason.
- [ ] **Step 2:** Generate the report and render-check it with headless Chrome
  (`google-chrome --headless=new --screenshot=…`).
- [ ] **Step 3:** Add the README experiments row.
- [ ] **Step 4:** Commit, push (`git push -u origin exp/expert-portfolio`), and open the PR
  with `gh pr create --base main`, ending the body with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
