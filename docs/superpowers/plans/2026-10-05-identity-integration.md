# Identity Integration (A, B, C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared cross-day evaluation harness (Phase 0), then bootstrap three
autoresearch loops that each test one way of using the fingerprint's identity more robustly:
- **A:** a gated pooled fallback;
- **B:** one shared network with per-person adapters;
- **C:** identity as a network input.
- **D** (revision 2026-10-09): a batch-level identity prior from neuralprint's Transition Grammar
  Biometric Prior. It is evaluated like A, B and C, but **not shippable until the organisers confirm
  sealed-phase batch composition**.

**Architecture:**
- **Phase 0** (`research/integration-harness/`, branch `exp/integration-harness`) trains the
  shared parts once per dataset and seed. Those parts are the fingerprint, the pooled EEGNet, the
  per-person EEGNet experts and the Riemannian experts. Phase 0 caches per-window predictions
  ("banks") in `outputs/t2-int-harness/`, and computes reference rows 1–3 and the adoption rule.
- **Each option** lives in its own worktree, branch and `research/<loop>/` workspace. It reads the
  banks, so no loop retrains the shared parts.

**Tech Stack:**
- Python 3.12 shared venv `.venv` (never install into it); torch, braindecode 1.8.1, scipy,
  sklearn, pyriemann.
- Existing Track 2 code from `main` after loop C merges: `track2/riemann_parts.py`,
  `track2/submission.py`, `track2/models.py`, `track2/train_mixture.py`.
- Tests: stdlib `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-05-identity-integration-design.md`

## Global Constraints

- **Precondition:** loop C (`exp/expert-portfolio`) is merged into `main` before Task 1. Every
  branch below starts from that `main`.
- **Primary metric:** cross-day test-session balanced accuracy, end to end with the **real**
  fingerprint (no oracle), as per-person differences against **row 2** (current design: E+T, C3
  as shipped, soft-routed). Pooled over the **17 dev people**; person-bootstrap 95% CI, 5,000
  resamples, rng seed 0; person score = mean over seeds 0, 1, 2.
- **Holdout:** the 8 people in `docs/superpowers/specs/2026-10-05-identity-integration-holdout.json`
  are never loaded by any loop (`data.load(name)` defaults to `part="dev"`). They are used once, in
  Task 9, for the chosen candidate only.
- **Candidate (dev):** pooled gain ≥ 0.005 with CI lower bound > 0, **and** sim2 (Dreyer, seed 0)
  not worse than row 2 by more than 0.01.
- **Adoption (holdout, Task 9, once):** the single chosen candidate's holdout mean gain ≥ +0.005.
  At most one option ships.
- **No leakage:** learned integration parameters are fitted on calibration data or
  leave-one-dataset-out (LODO), never on the test sessions being scored. Protocols are committed
  before their results.
- **Not used:** Dreyer R4–R6 (reused hidden runs).
- **Compute:**
  - ≤ 2 CPU threads per loop (`OMP_NUM_THREADS=2`, `--threads 2`).
  - Long jobs: `setsid nohup .venv/bin/python … > log 2>&1 < /dev/null &`, then track the PID
    from `ps -C python -o pid,args`.
  - Never launch background chains with `A=… && B=… && cmd &`: the `&` backgrounds the whole
    list, and its assignments are lost.
- **Git:** check `git branch --show-current` before every commit; never bare `git stash`.
  Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. PR bodies end with
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Artifacts:**
  - the harness writes `outputs/t2-int-harness/`;
  - loop A writes `outputs/t2-int-gate/`, B `outputs/t2-int-adapters/`, C
    `outputs/t2-int-conditioned/`;
  - the dataset caches are read-only.

## Deviation from the spec (ruled here, before any code)

BNCI 2015-001 has **one** calibration run per person (200 windows). Wherever the spec uses
"calibration runs", each person's single run is split into two contiguous halves (window order),
which act as runs 0 and 1. This applies to:
- "the last calibration run is validation";
- Riemannian reliability (run to run);
- C's leave-one-calibration-run-out posteriors.

BNCI 2014-001 (6 runs) and Zhou 2016 (4 session×run units) use their real runs.

## Review Focus

1. **A person with one calibration run** (BNCI 2015-001). Expected: the two-half split above, not
   a crash or empty validation. Test in Task 1.
2. **Classes other than 2.** Expected: every combiner and gate works for C = 3 and 4, with no
   2-class-only bias concatenation (loop C hit exactly this bug). Tests in Tasks 3, 6, 7, 8.
3. **Fingerprint distance scale differing across datasets** (13 vs 14 vs 22 channels → different
   feature dimensions). Expected: A's distance feature is relative to each dataset's own
   calibration median, so a gate fitted on two datasets transfers to the third. Tests in Tasks 2
   and 6.
4. **A test window whose fingerprint posterior is uniform.** Expected: A's gate moves weight to
   the pooled model; C's code equals the dropout (uniform) code it was trained with. Tests in
   Tasks 6 and 8.
5. **Seed / dataset bookkeeping.** Expected: the person score is the mean over the 3 seeds before
   pooling over people (17 dev, or the 8 holdout once), and people are never matched across
   datasets. Test in Task 3.

---

## File Structure

| Path | Branch | Responsibility |
|---|---|---|
| `research/integration-harness/src/data.py` | harness | Load the three cross-day datasets (and Dreyer sim2) into one dict format with calibration / validation / test masks and run units |
| `research/integration-harness/src/pipeline.py` | harness | Train the shared parts per dataset and seed; write the bank `.npz` and pooled-model `.pt` |
| `research/integration-harness/src/metrics.py` | harness | C3 as shipped, reference rows 1–3, per-person scores, pooled comparison with bootstrap CI, the adoption rule |
| `research/integration-harness/src/nets.py` | harness | `split_trunk_head` and `FILM_AFTER` for EEGNet / ShallowFBCSPNet (shared by B and C) |
| `research/integration-harness/tests/` | harness | `unittest` tests for the above |
| `track2/submission.py` | harness | `FBFingerprint.features()` (tangent features before the linear layer); `forward` unchanged |
| `docs/handoff/identity-integration-loops.md` | harness | How to start and run each loop |
| `research/integration-gate/src/gate.py` (+ tests) | `exp/integration-gate` | Option A |
| `research/integration-adapters/src/adapters.py` (+ tests) | `exp/integration-adapters` | Option B |
| `research/integration-conditioned/src/conditioned.py` (+ tests) | `exp/integration-conditioned` | Option C |
| `research/integration-sequence/src/grammar.py` (+ tests) | `exp/integration-sequence` | Option D (Transition Grammar prior) |

Tests for the harness run with:
`.venv/bin/python -W ignore -m unittest discover -s research/integration-harness/tests -v`.
Each loop runs its own tests the same way from its `research/<loop>/tests`.

---

### Task 1: Harness worktree and the cross-day data loader

**Files:**
- Create: `research/integration-harness/src/data.py`
- Create: `research/integration-harness/tests/test_data.py`

**Interfaces:**
- Produces:
  - `load(name: str, part: str = "dev") -> dict`, with `name ∈ {"bnci2014", "bnci2015",
    "zhou2016"}` and `part ∈ {"dev", "holdout", "all"}` (people from the locked holdout JSON). The
    dict's `name` is `name` for dev and `f"{name}_{part}"` otherwise, so banks never collide. Keys:
    - `X (n, chans, 480) float32`, `y (n,) int` in 0..C−1;
    - `person (n,) int` in 0..K−1, `unit (n,) int` (calibration run unit; −1 outside calibration);
    - `calib`, `val`, `test`: bool masks (`val ⊂ calib`);
    - `people (K,) array` (original subject ids), `n_classes`, `sfreq = 120.0`, `name`.
  - `load_dreyer_sim2() -> dict`, with the same keys plus `pool` (bool: training-pool windows for
    the pooled EEGNet besides calibration) and `pool_val` (the kit's val split). Here
    `val = pool_val`.
  - `DATASETS = ("bnci2014", "bnci2015", "zhou2016")`.

- [ ] **Step 1: Create the worktree** (from the main checkout, after loop C merged)

```bash
cd /home/promit/Documents/ephys_fm && git branch --show-current   # main
git pull --ff-only origin main
git worktree add .claude/worktrees/t2-int-harness -b exp/integration-harness main
for d in data external .venv outputs submissions; do
  ln -s /home/promit/Documents/ephys_fm/$d .claude/worktrees/t2-int-harness/$d; done
cd .claude/worktrees/t2-int-harness && mkdir -p research/integration-harness/{src,tests} outputs/t2-int-harness
```
Add a `CLAUDE.md` row for the worktree (`exp/integration-harness`, "Track 2: shared cross-day
harness for the identity-integration loops").

- [ ] **Step 2: Write the failing tests**

`research/integration-harness/tests/test_data.py`:
```python
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data  # noqa: E402

EXPECT = {"bnci2014": (9, 4, 22), "bnci2015": (12, 2, 13), "zhou2016": (4, 3, 14)}
DEV_K = {"bnci2014": 6, "bnci2015": 8, "zhou2016": 3}


class TestData(unittest.TestCase):
    def test_dev_part_excludes_the_holdout_people(self):
        hold = data.holdout()
        for name, K in DEV_K.items():
            d = data.load(name)                                 # part="dev" is the default
            self.assertEqual(len(d["people"]), K, name)
            self.assertFalse(set(d["people"].tolist()) & set(hold[name]), name)
            self.assertEqual(d["name"], name)
            h = data.load(name, part="holdout")
            self.assertEqual(sorted(h["people"].tolist()), sorted(hold[name]))
            self.assertEqual(h["name"], f"{name}_holdout")

    def test_shapes_and_masks(self):
        for name, (K, C, chans) in EXPECT.items():
            d = data.load(name, part="all")
            self.assertEqual((len(d["people"]), d["n_classes"], d["X"].shape[1]), (K, C, chans), name)
            self.assertEqual(d["X"].shape[2], 480)
            self.assertFalse((d["calib"] & d["test"]).any(), name)
            self.assertTrue((d["val"] <= d["calib"]).all(), name)
            for k in range(K):
                m = d["person"] == k
                self.assertGreater((m & d["calib"] & ~d["val"]).sum(), 0, (name, k))
                self.assertGreater((m & d["val"]).sum(), 0, (name, k))
                self.assertGreaterEqual((m & d["test"]).sum(), 150, (name, k))
                self.assertGreaterEqual(len(np.unique(d["unit"][m & d["calib"]])), 2, (name, k))

    def test_single_run_person_split_in_halves(self):
        d = data.load("bnci2015", part="all")
        m = (d["person"] == 0) & d["calib"]
        self.assertEqual(sorted(np.unique(d["unit"][m]).tolist()), [0, 1])
        self.assertEqual((d["unit"][m] == 0).sum(), (d["unit"][m] == 1).sum())
        self.assertTrue((d["val"][m] == (d["unit"][m] == 1)).all())   # the last unit is val

    def test_zhou_calibrates_on_two_sessions(self):
        d = data.load("zhou2016", part="all")
        raw = np.load(data.PATHS["zhou2016"], allow_pickle=True)
        self.assertEqual(sorted(np.unique(raw["session"][d["calib"]]).tolist()), [0, 1])
        self.assertEqual(np.unique(raw["session"][d["test"]]).tolist(), [2])


if __name__ == "__main__":
    unittest.main()
```

Run: `.venv/bin/python -W ignore -m unittest discover -s research/integration-harness/tests -v`
Expected: `ModuleNotFoundError: No module named 'data'`.

- [ ] **Step 3: Implement `data.py`**

```python
"""Cross-day datasets for the identity-integration loops (kit format: 120 Hz, 4-s windows).

Calibrate on every session but the last, test on the last. Within calibration, each person's
last run is validation (pooled-model epoch choice). A person with a single calibration run has
it split into two contiguous halves that act as runs 0 and 1 (BNCI 2015-001).
"""

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PATHS = {
    "bnci2014": REPO / "data" / "experiments" / "tangermann_windows.npz",
    "bnci2015": REPO / "outputs" / "t2-fingerprint" / "cache" / "BNCI2015_001_windows.npz",
    "zhou2016": REPO / "outputs" / "t2-fingerprint" / "cache" / "Zhou2016_windows.npz",
}
SPLITS = {"bnci2014": ([0], 1), "bnci2015": ([0], 1), "zhou2016": ([0, 1], 2)}
DATASETS = tuple(PATHS)
SFREQ = 120.0
HOLDOUT = REPO / "docs" / "superpowers" / "specs" / "2026-10-05-identity-integration-holdout.json"


def holdout():
    import json
    return json.loads(HOLDOUT.read_text())["holdout"]


def load(name, part="dev"):
    full = np.load(PATHS[name], allow_pickle=True)
    keep = {"dev": ~np.isin(full["subject"], holdout()[name]),
            "holdout": np.isin(full["subject"], holdout()[name]),
            "all": np.ones(len(full["subject"]), bool)}[part]
    raw = {k: full[k][keep] for k in ("X", "task", "subject", "session", "run")}
    cal_sessions, test_session = SPLITS[name]
    subj, ses, run = raw["subject"], raw["session"], raw["run"]
    people = np.unique(subj)
    person = np.searchsorted(people, subj)
    y = np.searchsorted(np.unique(raw["task"]), raw["task"])
    calib = np.isin(ses, cal_sessions)
    test = ses == test_session
    unit = np.full(len(y), -1)
    val = np.zeros(len(y), bool)
    for k in range(len(people)):
        idx = np.where(calib & (person == k))[0]
        keys = sorted(set(zip(ses[idx].tolist(), run[idx].tolist())))
        if len(keys) == 1:                                  # single run: two contiguous halves
            half = len(idx) // 2
            unit[idx[:half]], unit[idx[half:]] = 0, 1
        else:
            pos = {key: i for i, key in enumerate(keys)}
            unit[idx] = [pos[(s, r)] for s, r in zip(ses[idx], run[idx])]
        val[idx] = unit[idx] == unit[idx].max()
    return {"X": raw["X"].astype(np.float32), "y": y, "person": person, "unit": unit,
            "calib": calib, "val": val, "test": test, "people": people,
            "n_classes": int(y.max()) + 1, "sfreq": SFREQ,
            "name": name if part == "dev" else f"{name}_{part}"}


def load_dreyer_sim2():
    """Dreyer sim2 (loop C): the 14 locked people are the evaluation people."""
    import json
    import sys
    sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2")]
    from train_mixture import load_windows, masks
    d = load_windows()
    spec = REPO / "research" / "expert-portfolio" / "experiments" / "00-protocol" / "sim2_people.json"
    train, calib, hidden, people = masks(d, json.loads(spec.read_text())["people"])
    person = np.array([people.index(s) if s in people else -1 for s in d["subject"]])
    return {"X": d["X"], "y": d["y"], "person": person, "unit": d["run"].copy(),
            "calib": calib, "val": d["split"] == "val", "test": hidden,
            "pool": train, "pool_val": d["split"] == "val",
            "people": np.asarray(people), "n_classes": int(d["y"].max()) + 1,
            "sfreq": float(d["sfreq"]), "name": "dreyer_sim2"}
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -W ignore -m unittest discover -s research/integration-harness/tests -v`
Expected: 4 tests OK.

Class labels are mapped with `np.unique(raw["task"])` on the loaded part, so a part must contain
every class. Every person in all three datasets has every class, so this holds.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # exp/integration-harness
git add research/integration-harness CLAUDE.md
git commit -m "Add the cross-day data loader for the identity-integration harness

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Shared pipeline and banks (fingerprint, pooled EEGNet, experts, Riemannian experts)

**Files:**
- Modify: `track2/submission.py` (`FBFingerprint.features`)
- Create: `research/integration-harness/src/pipeline.py`
- Create: `research/integration-harness/tests/test_pipeline.py`

**Interfaces:**
- Consumes:
  - `data.load(name)` / `data.load_dreyer_sim2()`;
  - `riemann_parts.fit_fingerprint(X, person, sfreq) -> state`;
  - `riemann_parts.fit_riemann(X, y, person, run, n_people, n_classes, sfreq) -> (riemann_state, combiner_state, rel, n_out)`;
  - `submission.FBFingerprint`, `RiemannExperts`;
  - `train_mixture.fit(model, X, y, *, lr, epochs, seed, bs, X_val=None, y_val=None, name="", log_every=10)`
    and `logits(model, X, bs=512)`;
  - `models.make_model(arch, n_chans, n_outputs, n_times, sfreq)`.
- Produces:
  - `FBFingerprint.features(X) -> (B, F)` (the concatenated tangent features; `forward` =
    `linear(features)`).
  - `pipeline.run(d: dict, seed: int, threads: int, epochs: int, ft_epochs: int = 50) -> Path`,
    which writes `outputs/t2-int-harness/bank_{name}_seed{seed}.npz` with keys:
    - test windows: `y`, `person`, `fp_logp (n, K)`, `fp_dist (n,)`, `fp_dist_ref ()`,
      `eeg_logp (n, K, C)`, `riemann_logp (n, K, C)`, `pooled_logp (n, C)`;
    - per person: `rel (K,)`, already chance-normalised (from `fit_riemann`'s combiner state);
    - calibration windows: `calib_idx`, `y_calib`, `person_calib`, `unit_calib`,
      `fp_cv_logp_calib (n_cal, K)` (leave-one-unit-out).

    It also writes `pooled_{name}_seed{seed}.pt` (the pooled EEGNet state dict).

- [ ] **Step 1: Write the failing tests**

`research/integration-harness/tests/test_pipeline.py`:
```python
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2"), str(HERE / "src")]
import pipeline  # noqa: E402
import riemann_parts  # noqa: E402
from submission import FBFingerprint  # noqa: E402


def toy(K=3, C=3, chans=6, per=40, seed=0):
    """Toy dataset: person-specific spatial scaling (identity) + class-specific band power."""
    rng = np.random.default_rng(seed)
    X, y, person, unit, calib, test = [], [], [], [], [], []
    for k in range(K):
        gain = 1 + 2 * rng.random(chans)
        for sess in (0, 1):
            for i in range(per):
                c = i % C
                t = np.arange(480) / 120.0
                sig = rng.standard_normal((chans, 480)) + np.sin(2 * np.pi * (8 + 6 * c) * t) * (c + 1)
                X.append(sig * gain[:, None]); y.append(c); person.append(k)
                unit.append((i * 2) // per if sess == 0 else -1)
                calib.append(sess == 0); test.append(sess == 1)
    calib = np.array(calib); unit = np.array(unit)
    val = calib & (unit == 1)
    return {"X": np.array(X, np.float32), "y": np.array(y), "person": np.array(person),
            "unit": unit, "calib": calib, "val": val, "test": np.array(test),
            "people": np.arange(K), "n_classes": C, "sfreq": 120.0, "name": "toy"}


class TestPipeline(unittest.TestCase):
    def test_features_then_linear_equals_forward(self):
        d = toy()
        st = riemann_parts.fit_fingerprint(d["X"][d["calib"]], d["person"][d["calib"]], 120.0)
        fp = FBFingerprint(6, 6, 480, 3); fp.load_state_dict(st)
        X = torch.from_numpy(d["X"][:5])
        with torch.inference_mode():
            self.assertTrue(torch.allclose(fp(X), fp.linear(fp.features(X)), atol=1e-5))

    def test_bank_keys_and_shapes(self):
        d = toy()
        path = pipeline.run(d, seed=0, threads=1, epochs=1, ft_epochs=1,
                            out_dir=Path(self._tmp()))
        b = np.load(path)
        n, K, C = int(d["test"].sum()), 3, 3
        self.assertEqual(b["fp_logp"].shape, (n, K))
        self.assertEqual(b["eeg_logp"].shape, (n, K, C))
        self.assertEqual(b["riemann_logp"].shape, (n, K, C))
        self.assertEqual(b["pooled_logp"].shape, (n, C))
        self.assertEqual(b["fp_dist"].shape, (n,))
        self.assertGreater(float(b["fp_dist_ref"]), 0)
        self.assertEqual(b["fp_cv_logp_calib"].shape, (int(d["calib"].sum()), K))
        self.assertEqual(b["rel"].shape, (K,))
        for k in ("fp_logp", "eeg_logp", "riemann_logp", "pooled_logp", "fp_cv_logp_calib"):
            np.testing.assert_allclose(np.exp(b[k]).sum(-1), 1.0, atol=1e-4, err_msg=k)
        self.assertGreater((b["fp_logp"].argmax(1) == b["person"]).mean(), 0.9)   # identity is easy
        self.assertEqual(b["fp_feat_test"].shape[0], n)                           # loop D inputs
        self.assertEqual(b["fp_feat_calib"].shape[0], int(d["calib"].sum()))

    def _tmp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        return self._td.name


if __name__ == "__main__":
    unittest.main()
```

Run the suite; expected: `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 2: Add `FBFingerprint.features`**

In `track2/submission.py`, split `FBFingerprint.forward`:
```python
    def features(self, X):
        """Concatenated per-band tangent features (B, F), before the linear layer."""
        X = X.to(self.filters.dtype)
        return torch.cat([tangent(oas(X @ M), W)
                          for M, W in zip(self.filters, self.cref_isqrt)], -1)

    def forward(self, X):
        """Logits (B, K)."""
        return self.linear(self.features(X).to(self.linear.weight.dtype))
```
Then run the existing Track 2 suite: `.venv/bin/python -W ignore -m unittest discover -s track2/tests -t .`
(expected: OK, because `forward` is unchanged).

- [ ] **Step 3: Implement `pipeline.py`**

```python
"""Train the shared parts once per dataset and seed, and cache per-window predictions."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[2]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2")]
OUT = REPO / "outputs" / "t2-int-harness"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _lsm(z):
    return torch.log_softmax(torch.as_tensor(z, dtype=torch.float64), -1).numpy()


def run(d, seed, threads, epochs, ft_epochs=50, out_dir=OUT):
    import riemann_parts
    from models import make_model
    from submission import FBFingerprint, RiemannExperts
    from train_mixture import fit, logits
    torch.set_num_threads(threads)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X, y, person, unit = d["X"], d["y"], d["person"], d["unit"]
    cal, val, te = d["calib"], d["val"], d["test"]
    K, C, sf = len(d["people"]), d["n_classes"], d["sfreq"]
    n_chans, n_times = X.shape[1:]
    Xt = torch.from_numpy(X[te])
    bank = {"y": y[te], "person": person[te]}

    log(f"{d['name']} seed {seed}: fingerprint")
    st = riemann_parts.fit_fingerprint(X[cal], person[cal], sf)
    fp = FBFingerprint(len(riemann_parts.FP_BANDS), n_chans, n_times, K); fp.load_state_dict(st)
    with torch.inference_mode():
        bank["fp_logp"] = _lsm(fp(Xt).double())
        f_cal = fp.features(torch.from_numpy(X[cal])).double().numpy()
        f_te = fp.features(Xt).double().numpy()
    mu, sd = f_cal.mean(0), f_cal.std(0) + 1e-9
    z_cal, z_te = (f_cal - mu) / sd, (f_te - mu) / sd
    cents = np.stack([z_cal[person[cal] == k].mean(0) for k in range(K)])
    dist = lambda z: np.sqrt(((z[:, None] - cents[None]) ** 2).sum(-1)).min(1) / np.sqrt(z.shape[1])  # noqa: E731
    bank["fp_dist"], bank["fp_dist_ref"] = dist(z_te), np.median(dist(z_cal))
    bank["fp_feat_test"], bank["fp_feat_calib"] = f_te.astype(np.float32), f_cal.astype(np.float32)
    torch.save(st, out_dir / f"fp_{d['name']}_seed{seed}.pt")             # loop D

    log("fingerprint, leave-one-unit-out posteriors on calibration")
    cal_idx = np.where(cal)[0]
    cv = np.zeros((len(cal_idx), K))
    for u in np.unique(unit[cal]):
        tr, ho = cal & (unit != u), cal_idx[unit[cal] == u]
        st_u = riemann_parts.fit_fingerprint(X[tr], person[tr], sf)
        fp_u = FBFingerprint(len(riemann_parts.FP_BANDS), n_chans, n_times, K); fp_u.load_state_dict(st_u)
        with torch.inference_mode():
            cv[unit[cal] == u] = _lsm(fp_u(torch.from_numpy(X[ho])).double())
    bank |= {"calib_idx": cal_idx, "y_calib": y[cal], "person_calib": person[cal],
             "unit_calib": unit[cal], "fp_cv_logp_calib": cv}

    log("pooled EEGNet")
    pool = d.get("pool", np.zeros(len(y), bool)) | (cal & ~val)
    vmask = d.get("pool_val", val)
    pooled, info = fit(make_model("eegnet", n_chans, C, n_times, sf), X[pool], y[pool], lr=1e-3,
                       epochs=epochs, seed=seed, bs=64, X_val=X[vmask], y_val=y[vmask],
                       name="pooled", log_every=25)
    torch.save(pooled.state_dict(), out_dir / f"pooled_{d['name']}_seed{seed}.pt")
    bank["pooled_logp"] = _lsm(logits(pooled, X[te]).double())

    log(f"per-person EEGNet experts ({K})")
    eeg = np.zeros((int(te.sum()), K, C))
    for k in range(K):
        m = cal & (person == k)
        e, _ = fit(copy.deepcopy(pooled), X[m], y[m], lr=1e-4, epochs=ft_epochs, seed=seed, bs=32)
        eeg[:, k] = _lsm(logits(e, X[te]).double())
    bank["eeg_logp"] = eeg

    log("Riemannian experts")
    rs, cs, _, n_out = riemann_parts.fit_riemann(X[cal], y[cal], person[cal], unit[cal], K, C, sf)
    rm = RiemannExperts(K, n_chans, n_times, n_out, C); rm.load_state_dict(rs)
    with torch.inference_mode():
        bank["riemann_logp"] = torch.cat([rm(Xt[i:i + 256]) for i in range(0, len(Xt), 256)]).numpy()
    bank["rel"] = cs["rel"].numpy()
    path = out_dir / f"bank_{d['name']}_seed{seed}.npz"
    np.savez(path, **bank, pooled_info=str(info))
    log(f"saved {path}")
    return path


if __name__ == "__main__":
    import argparse
    import data
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=[*data.DATASETS, "dreyer_sim2"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    d = data.load_dreyer_sim2() if a.dataset == "dreyer_sim2" else data.load(a.dataset)
    for s in [int(x) for x in a.seeds.split(",")]:
        run(d, s, a.threads, epochs=100 if a.dataset == "dreyer_sim2" else 150)
```

`fit_riemann`'s `run` argument receives `unit[cal]`. That's the run-to-run reliability unit,
including BNCI 2015's two halves.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -W ignore -m unittest discover -s research/integration-harness/tests -v`
Expected: all OK (the toy bank test trains for 1 epoch; under a minute).

- [ ] **Step 5: Commit**

```bash
git branch --show-current
git add track2/submission.py research/integration-harness
git commit -m "Add the harness pipeline: shared parts per dataset and seed, cached as banks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Reference rows, metrics and the adoption rule

**Files:**
- Create: `research/integration-harness/src/metrics.py`
- Create: `research/integration-harness/tests/test_metrics.py`

**Interfaces:**
- Consumes: a bank dict (Task 2 keys); `submission.LogLinearCombiner`; `riemann_parts.C3`.
- Produces:
  - `c3_logp(bank) -> (n, K, C)` (E+T per person, as shipped);
  - `route(fp_logp (n,K), logp (n,K,C)) -> (n, C)`;
  - `rows(bank) -> {"pooled", "current", "oracle"}`, each `(n, C)` log-probabilities;
  - `person_scores(y, person, logp, K) -> (K,)`;
  - `pooled_comparison(per_dataset: {name: (opt (K,), ref (K,))}) -> {"mean", "ci", "better", "worse", "n"}`;
  - `adopt(cmp: dict, sim2_delta: float) -> (bool, str)`.

- [ ] **Step 1: Write the failing tests**

`research/integration-harness/tests/test_metrics.py`:
```python
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2"), str(HERE / "src")]
import metrics  # noqa: E402


def logp(*shape, seed=0):
    z = np.random.default_rng(seed).standard_normal(shape)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


class TestMetrics(unittest.TestCase):
    def test_c3_works_for_four_classes(self):
        b = {"eeg_logp": logp(5, 3, 4), "riemann_logp": logp(5, 3, 4, seed=1), "rel": np.array([.5, .7, .9])}
        lp = metrics.c3_logp(b)
        self.assertEqual(lp.shape, (5, 3, 4))
        np.testing.assert_allclose(np.exp(lp).sum(-1), 1, atol=1e-9)

    def test_oracle_row_picks_true_person(self):
        b = {"eeg_logp": logp(4, 2, 2), "riemann_logp": logp(4, 2, 2, seed=1), "rel": np.array([.6, .8]),
             "fp_logp": np.log(np.full((4, 2), .5)), "pooled_logp": logp(4, 2, seed=2),
             "person": np.array([0, 1, 0, 1])}
        r = metrics.rows(b)
        np.testing.assert_allclose(r["oracle"], metrics.c3_logp(b)[np.arange(4), b["person"]])

    def test_pooled_comparison_counts_people_not_datasets(self):
        cmp = metrics.pooled_comparison({"a": (np.array([.8, .9]), np.array([.7, .9])),
                                         "b": (np.array([.6]), np.array([.65]))})
        self.assertEqual(cmp["n"], 3)
        self.assertAlmostEqual(cmp["mean"], (0.1 + 0 - 0.05) / 3)
        self.assertEqual((cmp["better"], cmp["worse"]), (1, 1))

    def test_adopt_rule(self):
        self.assertTrue(metrics.adopt({"mean": .01, "ci": (.002, .02)}, sim2_delta=-.005)[0])
        self.assertFalse(metrics.adopt({"mean": .01, "ci": (-.001, .02)}, sim2_delta=0)[0])
        self.assertFalse(metrics.adopt({"mean": .004, "ci": (.001, .01)}, sim2_delta=0)[0])
        self.assertFalse(metrics.adopt({"mean": .02, "ci": (.01, .03)}, sim2_delta=-.011)[0])


if __name__ == "__main__":
    unittest.main()
```

Run the suite; expected: `ModuleNotFoundError: No module named 'metrics'`.

- [ ] **Step 2: Implement `metrics.py`**

```python
"""Reference rows, per-person scores, the pooled comparison and the adoption rule."""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2")]


def c3_logp(bank):
    """E+T per person with C3 as shipped (Dreyer coefficients; class bias for 2 classes only)."""
    import riemann_parts
    from submission import LogLinearCombiner
    K, C = bank["eeg_logp"].shape[1:]
    comb = LogLinearCombiner(K, C)
    comb.coef.copy_(torch.tensor([riemann_parts.C3[k] for k in ("a", "c0", "c1")], dtype=torch.float64))
    if C == 2:
        comb.class_bias[1] = riemann_parts.C3["b"]
    comb.rel.copy_(torch.as_tensor(bank["rel"], dtype=torch.float64))
    with torch.inference_mode():
        return comb(torch.as_tensor(bank["eeg_logp"]), torch.as_tensor(bank["riemann_logp"])).numpy()


def route(fp_logp, logp):
    p = (np.exp(fp_logp)[:, :, None] * np.exp(logp)).sum(1)
    return np.log(np.clip(p, 1e-12, 1))


def rows(bank):
    lp = c3_logp(bank)
    n = np.arange(len(bank["person"]))
    return {"pooled": bank["pooled_logp"], "current": route(bank["fp_logp"], lp),
            "oracle": lp[n, bank["person"]]}


def person_scores(y, person, logp, K):
    pred = np.asarray(logp).argmax(-1)
    return np.array([balanced_accuracy_score(y[person == k], pred[person == k]) for k in range(K)])


def pooled_comparison(per_dataset, seed=0, n_boot=5000):
    d = np.concatenate([np.asarray(o) - np.asarray(r) for o, r in per_dataset.values()])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"mean": float(d.mean()), "ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "n": int(len(d))}


def adopt(cmp, sim2_delta):
    if cmp["mean"] < 0.005:
        return False, f"pooled gain {cmp['mean']:+.4f} < 0.005"
    if cmp["ci"][0] <= 0:
        return False, f"CI lower bound {cmp['ci'][0]:+.4f} ≤ 0"
    if sim2_delta < -0.01:
        return False, f"sim2 {sim2_delta:+.4f} below row 2 by more than 0.01"
    return True, "adopt"
```

- [ ] **Step 3: Run the tests, then commit**

Run: `.venv/bin/python -W ignore -m unittest discover -s research/integration-harness/tests -v` (all OK)
```bash
git branch --show-current
git add research/integration-harness
git commit -m "Add harness metrics: reference rows, per-person scores, pooled comparison, adoption rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `nets.py`, the shared trunk / head split for B and C

**Files:**
- Create: `research/integration-harness/src/nets.py`
- Create: `research/integration-harness/tests/test_nets.py`

**Interfaces:**
- Produces:
  - `split_trunk_head(model) -> (trunk: nn.Sequential, head: nn.Module)`, where trunk = every
    child before `final_layer` and head = `final_layer`;
  - `FILM_AFTER = {"eegnet": "bnorm_1", "shallow": "bnorm"}` (the layer after which C injects
    FiLM);
  - `feature_channels(arch) -> int` (16 for EEGNet, 40 for ShallowFBCSPNet).

- [ ] **Step 1: Failing test**

`research/integration-harness/tests/test_nets.py`:
```python
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "track2"), str(HERE / "src")]
import nets  # noqa: E402
from models import make_model  # noqa: E402


class TestNets(unittest.TestCase):
    def test_split_reproduces_forward(self):
        for arch in ("eegnet", "shallow"):
            m = make_model(arch, 22, 4, 480, 120.0).eval()
            trunk, head = nets.split_trunk_head(m)
            x = torch.randn(3, 22, 480)
            with torch.inference_mode():
                self.assertTrue(torch.allclose(m(x), head(trunk(x)), atol=1e-6), arch)
                self.assertEqual(trunk(x).shape[1], nets.feature_channels(arch), arch)

    def test_film_layer_exists(self):
        for arch in ("eegnet", "shallow"):
            names = [n for n, _ in make_model(arch, 22, 4, 480, 120.0).named_children()]
            self.assertIn(nets.FILM_AFTER[arch], names)


if __name__ == "__main__":
    unittest.main()
```

Run; expected: `ModuleNotFoundError: No module named 'nets'`.

- [ ] **Step 2: Implement**

```python
"""Trunk / head split and FiLM injection points for EEGNet and ShallowFBCSPNet (loops B, C)."""

from torch import nn

FILM_AFTER = {"eegnet": "bnorm_1", "shallow": "bnorm"}
_CHANNELS = {"eegnet": 16, "shallow": 40}


def split_trunk_head(model):
    children = list(model.named_children())
    names = [n for n, _ in children]
    i = names.index("final_layer")
    return nn.Sequential(*[m for _, m in children[:i]]), children[i][1]


def feature_channels(arch):
    return _CHANNELS[arch]
```

- [ ] **Step 3: Tests OK, then commit** with the message
  `Add trunk / head split and FiLM points for the integration loops`.

---

### Task 5: Build the banks and the reference table; hand-off brief; PR

**Files:**
- Create: `research/integration-harness/src/run_refs.py`
- Create: `research/integration-harness/results/references.md` (+ `.json`)
- Create: `docs/handoff/identity-integration-loops.md`

- [ ] **Step 1: Build the banks (background, ≤ 2 threads per job)**

```bash
L=outputs/t2-int-harness/logs; mkdir -p $L
E="env OMP_NUM_THREADS=2 .venv/bin/python -W ignore research/integration-harness/src/pipeline.py"
setsid nohup bash -c "$E bnci2014 > $L/bnci2014.log 2>&1 && $E bnci2015 > $L/bnci2015.log 2>&1 && $E zhou2016 > $L/zhou2016.log 2>&1 && echo DONE > $L/crossday.done" < /dev/null > /dev/null 2>&1 &
setsid nohup bash -c "$E dreyer_sim2 --seeds 0 > $L/sim2.log 2>&1 && echo DONE > $L/sim2.done" < /dev/null > /dev/null 2>&1 &
```
Expected:
- 9 cross-day banks plus 1 sim2 bank in `outputs/t2-int-harness/`;
- cross-day pooled models train 150 epochs (minutes each); sim2 takes about 1.5 h.

Sanity checks:
- BNCI 2014-001 pooled EEGNet test accuracy in the 0.68–0.72 range (loop B's earlier runs:
  0.692 ± 0.007);
- fingerprint top-1 on the test session well above chance.

- [ ] **Step 2: Write `run_refs.py` and produce the reference table**

```python
"""Reference rows 1–3 per dataset (seeds 0–2) and on sim2; person scores averaged over seeds."""
import json, sys
from pathlib import Path
import numpy as np
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import data, metrics  # noqa: E402
OUT = SRC.parents[2] / "outputs" / "t2-int-harness"
RES = SRC.parent / "results"


def scores(name, seeds):
    per = {"pooled": [], "current": [], "oracle": []}
    for s in seeds:
        b = dict(np.load(OUT / f"bank_{name}_seed{s}.npz"))
        K = b["eeg_logp"].shape[1]
        for k, lp in metrics.rows(b).items():
            per[k].append(metrics.person_scores(b["y"], b["person"], lp, K))
        fp_right = b["fp_logp"].argmax(1) == b["person"]
        per.setdefault("fp_top1", []).append(float(fp_right.mean()))
    return {k: (np.mean(v, 0) if k != "fp_top1" else float(np.mean(v))) for k, v in per.items()}


def main():
    RES.mkdir(exist_ok=True)
    out = {name: scores(name, (0, 1, 2)) for name in data.DATASETS}
    out["dreyer_sim2"] = scores("dreyer_sim2", (0,))
    rows = ["| Dataset | People | Fingerprint top-1 | Pooled | Current (row 2) | Oracle (row 3) |",
            "|---|---|---|---|---|---|"]
    for name, r in out.items():
        rows.append(f"| {name} | {len(r['current'])} | {r['fp_top1']:.3f} | {r['pooled'].mean():.3f} | "
                    f"{r['current'].mean():.3f} | {r['oracle'].mean():.3f} |")
    cmp = metrics.pooled_comparison({n: (out[n]["current"], out[n]["pooled"]) for n in data.DATASETS})
    rows.append(f"\nCurrent − pooled, 17 dev cross-day people: {cmp['mean']:+.4f} "
                f"[{cmp['ci'][0]:+.4f}, {cmp['ci'][1]:+.4f}], {cmp['better']}/{cmp['worse']}")
    (RES / "references.md").write_text("# Reference rows (balanced accuracy, person mean)\n\n" + "\n".join(rows) + "\n")
    (RES / "references.json").write_text(json.dumps(
        {n: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for n, r in out.items()}, indent=2))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
```
Run it once all banks exist. Expected: four rows, plus the current-vs-pooled line (loop B's
BNCI-only number was about +0.05).

- [ ] **Step 3: Write the hand-off brief**

`docs/handoff/identity-integration-loops.md` must contain, for each of A, B and C:
- the worktree / branch / workspace / artifacts names from the spec's Sections 2–4;
- the start commands:
  `git worktree add …`, then the five symlinks, then `cd … && claude`;
- the prompt: *"Read docs/superpowers/specs/2026-10-05-identity-integration-design.md and
  docs/superpowers/plans/2026-10-05-identity-integration.md (Task 6, 7 or 8), and run the loop
  with the autoresearch skill. Set up its 20-minute heartbeat first."*;
- the shared rules: the banks in `outputs/t2-int-harness/` are read-only; ≤ 2 threads;
  protocols before results; R4–R6 unused; how to compute the primary metric with `metrics.py`.

- [ ] **Step 4: Commit, push, PR**

```bash
git branch --show-current
git add research/integration-harness docs/handoff/identity-integration-loops.md
git commit -m "research(results): harness banks and reference rows for the identity-integration loops

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin exp/integration-harness
gh pr create --base main --title "Track 2: shared cross-day harness for the identity-integration loops" --body "…"
```
The loops start once this PR is merged.

---

### Task 6: Loop A bootstrap: gated pooled fallback (`int-gate`)

Run in its own session in `.claude/worktrees/t2-int-gate` (branch `exp/integration-gate`), per the
hand-off brief. The tasks below are the loop's first protocol and code; the autoresearch loop
continues from its findings.

**Files:**
- Create: `research/integration-gate/{research-state.yaml, research-log.md, findings.md}`
- Create: `research/integration-gate/experiments/00-protocol/protocol.md`
- Create: `research/integration-gate/src/gate.py`, `research/integration-gate/tests/test_gate.py`

**Interfaces:**
- Consumes: `metrics.c3_logp`, `metrics.route`, `metrics.person_scores`,
  `metrics.pooled_comparison`, `metrics.adopt`; bank keys `fp_logp`, `fp_dist`, `fp_dist_ref`,
  `pooled_logp`.
- Produces:
  - `features(bank) -> (H (n,), D (n,))`: H = fingerprint entropy / log K;
    D = fp_dist / fp_dist_ref;
  - `Gate(use_dist: bool)` with `.fit(banks: list[dict]) -> self`, `.g(bank) -> (n,)` and
    `.predict(bank) -> (n, C)` log-probabilities;
  - `lodo(names, seeds, use_dist) -> {name: (K,) person scores}`.

- [ ] **Step 1: Protocol 00 (commit before any result)** with:
  - variants A1 (`use_dist=False`) and A2 (`use_dist=True`);
  - LODO over the three datasets (fit on all seeds of two, score each seed of the third);
  - for sim2, the gate fitted on all three datasets;
  - the primary metric and adoption rule copied from the spec;
  - the diagnostic: accuracy on fingerprint-wrong vs fingerprint-right windows.

- [ ] **Step 2: Failing tests** `research/integration-gate/tests/test_gate.py`:
```python
import sys, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "research" / "integration-harness" / "src"), str(HERE / "src")]
import gate  # noqa: E402


def bank(n=60, K=3, C=4, seed=0, uniform_fp=False):
    rng = np.random.default_rng(seed)
    lsm = lambda z: z - np.log(np.exp(z).sum(-1, keepdims=True))  # noqa: E731
    y = rng.integers(0, C, n); person = rng.integers(0, K, n)
    noise = rng.standard_normal((n, K))                   # always drawn: same rng stream either way
    fp = np.full((n, K), -np.log(K)) if uniform_fp else lsm(5 * np.eye(K)[person] + noise)
    good = lsm(4 * np.eye(C)[y] + rng.standard_normal((n, C)))
    eeg = np.repeat(good[:, None], K, 1)
    return {"y": y, "person": person, "fp_logp": fp, "eeg_logp": eeg, "riemann_logp": eeg.copy(),
            "rel": np.full(K, .8), "pooled_logp": lsm(rng.standard_normal((n, C))),
            "fp_dist": rng.random(n), "fp_dist_ref": np.float64(.5)}


class TestGate(unittest.TestCase):
    def test_features_normalised(self):
        b = bank(uniform_fp=True)
        H, D = gate.features(b)
        np.testing.assert_allclose(H, 1.0, atol=1e-9)          # uniform posterior → max entropy
        np.testing.assert_allclose(D, b["fp_dist"] / .5)

    def test_predict_is_a_distribution_for_four_classes(self):
        g = gate.Gate(use_dist=True).fit([bank(seed=1), bank(seed=2)])
        lp = g.predict(bank(seed=3))
        self.assertEqual(lp.shape, (60, 4))
        np.testing.assert_allclose(np.exp(lp).sum(-1), 1, atol=1e-9)

    def test_gate_rises_with_entropy(self):
        g = gate.Gate(use_dist=False)
        g.theta = np.array([0.0, 4.0, 0.0])                    # α, β, γ
        self.assertGreater(g.g(bank(uniform_fp=True)).mean(), g.g(bank()).mean())

    def test_fit_prefers_routing_when_experts_are_right(self):
        g = gate.Gate(use_dist=False).fit([bank(seed=s) for s in range(3)])
        self.assertLess(g.g(bank(seed=9)).mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
```
Run: `.venv/bin/python -W ignore -m unittest discover -s research/integration-gate/tests -v`.
Expected: `ModuleNotFoundError: No module named 'gate'`.

- [ ] **Step 3: Implement `gate.py`**
```python
"""Option A: p = (1 − g)·Σ_k p(k|x)·p_k + g·p_pooled, g = σ(α + β·H + γ·D)."""
import sys
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research" / "integration-harness" / "src"))
import metrics  # noqa: E402
OUT = REPO / "outputs" / "t2-int-harness"


def features(bank):
    p = np.exp(bank["fp_logp"])
    H = -(p * bank["fp_logp"]).sum(1) / np.log(p.shape[1])
    return H, bank["fp_dist"] / float(bank["fp_dist_ref"])


def _parts(bank):
    route = metrics.route(bank["fp_logp"], metrics.c3_logp(bank))
    H, D = features(bank)
    return [torch.as_tensor(a, dtype=torch.float64) for a in (route, bank["pooled_logp"], H, D)]


class Gate:
    def __init__(self, use_dist):
        self.use_dist = use_dist
        self.theta = np.zeros(3)

    def _g(self, th, H, D):
        return torch.sigmoid(th[0] + th[1] * H + (th[2] * D if self.use_dist else 0.0))

    def _mix(self, th, route, pooled, H, D):
        g = self._g(th, H, D)[:, None]
        return torch.logaddexp(torch.log1p(-g + 1e-12) + route, torch.log(g + 1e-12) + pooled)

    def fit(self, banks):
        parts = [_parts(b) for b in banks]
        ys = [torch.as_tensor(b["y"]) for b in banks]
        th = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = sum(-self._mix(th, *p)[torch.arange(len(y)), y].mean() for p, y in zip(parts, ys)) / len(parts)
            loss = loss + 1e-3 * (th ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.theta = th.detach().numpy()
        return self

    def g(self, bank):
        _, _, H, D = _parts(bank)
        return self._g(torch.as_tensor(self.theta), H, D).numpy()

    def predict(self, bank):
        with torch.no_grad():
            return self._mix(torch.as_tensor(self.theta), *_parts(bank)).numpy()


def load(name, seed):
    return dict(np.load(OUT / f"bank_{name}_seed{seed}.npz"))


def lodo(names, seeds, use_dist):
    out = {}
    for held in names:
        g = Gate(use_dist).fit([load(n, s) for n in names if n != held for s in seeds])
        per = [metrics.person_scores(b["y"], b["person"], g.predict(b), b["eeg_logp"].shape[1])
               for b in (load(held, s) for s in seeds)]
        out[held] = np.mean(per, 0)
    return out
```

- [ ] **Step 4: Tests OK; commit protocol and code (separately: protocol first).**

- [ ] **Step 5: First experiment, the loop's first result**
  - Run A1 and A2 with LODO.
  - Compare with row 2 via `metrics.pooled_comparison` (row-2 person scores from
    `research/integration-harness/results/references.json`).
  - Fit on all three datasets and score sim2 (seed 0).
  - Apply `metrics.adopt`.
  - Write `experiments/01-a1-a2/analysis.md`, including the fingerprint-wrong / fingerprint-right
    diagnostic.
  - Commit the results. The loop continues from its findings (e.g. a gate using the
    fingerprint's top-2 margin, or a log-linear fallback), each with its own protocol first.

---

### Task 7: Loop B bootstrap: shared network with per-person adapters (`int-adapters`)

Run in its own session in `.claude/worktrees/t2-int-adapters` (branch `exp/integration-adapters`).

**Files:**
- Create: `research/integration-adapters/{research-state.yaml, research-log.md, findings.md}`
- Create: `research/integration-adapters/experiments/00-protocol/protocol.md`
- Create: `research/integration-adapters/src/adapters.py`, `research/integration-adapters/tests/test_adapters.py`

**Interfaces:**
- Consumes:
  - `nets.split_trunk_head`, `nets.feature_channels`;
  - `data.load` / `load_dreyer_sim2`;
  - harness pooled models `pooled_{name}_seed{s}.pt`;
  - `models.make_model`;
  - bank keys `fp_logp`, `riemann_logp`, `rel`.
- Produces:
  - `Adapters(trunk, head, K, film: bool, in_shape: (n_chans, n_times))`, an `nn.Module` with
    `forward(X) -> (B, K, C)`
    log-probabilities: one trunk pass, then per-person FiLM (optional) and head;
  - `fit_adapters(model: Adapters, X, y, person, lam, steps, lr, seed) -> Adapters`;
  - `run_dataset(name, seed, film, lam) -> (n_test, K, C)` adapter log-probabilities, aligned
    with the bank's test windows.

- [ ] **Step 1: Protocol 00 (commit first):**
  - variants B1 (head only) and B2 (head + FiLM on trunk features); trunk EEGNet first;
  - λ ∈ {0.01, 0.1, 1.0}, chosen LODO (the λ maximising the other two datasets' mean gain);
  - integration: soft-route over per-person adapters by the bank's fingerprint, then combined
    with the Riemannian experts as C3 (shipped coefficients) and, as a second row, C3
    coefficients refitted LODO;
  - the ShallowFBCSPNet trunk is a second, pre-registered experiment after the EEGNet results;
  - the metric and adoption rule from the spec.

- [ ] **Step 2: Failing tests** `research/integration-adapters/tests/test_adapters.py`:
```python
import sys, unittest
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parents[1]; REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "track2"), str(REPO / "research" / "integration-harness" / "src"), str(HERE / "src")]
import adapters  # noqa: E402
import nets  # noqa: E402
from models import make_model  # noqa: E402

SHAPE = (8, 480)


class TestAdapters(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.trunk, self.head = nets.split_trunk_head(make_model("eegnet", 8, 3, 480, 120.0).eval())

    def test_identity_init_equals_pooled(self):
        a = adapters.Adapters(self.trunk, self.head, K=4, film=True, in_shape=SHAPE).eval()
        x = torch.randn(5, *SHAPE)
        with torch.inference_mode():
            ref = torch.log_softmax(self.head(self.trunk(x)), -1)
            out = a(x)
        self.assertEqual(tuple(out.shape), (5, 4, 3))
        for k in range(4):
            self.assertTrue(torch.allclose(out[:, k], ref, atol=1e-6))

    def test_fit_moves_only_that_persons_adapter(self):
        a = adapters.Adapters(self.trunk, self.head, K=2, film=True, in_shape=SHAPE)
        h0 = [p.detach().clone() for p in a.heads[0].parameters()]
        h1 = [p.detach().clone() for p in a.heads[1].parameters()]
        X = torch.randn(30, *SHAPE); y = torch.randint(0, 3, (30,))
        person = torch.zeros(30, dtype=torch.long)                     # only person 0 has data
        adapters.fit_adapters(a, X, y, person, lam=0.1, steps=20, lr=1e-2, seed=0)
        self.assertTrue(all(torch.equal(b, p) for b, p in zip(h1, a.heads[1].parameters())))
        self.assertFalse(all(torch.equal(b, p) for b, p in zip(h0, a.heads[0].parameters())))
        self.assertTrue(torch.equal(a.gamma[1], torch.zeros_like(a.gamma[1])))
        self.assertFalse(torch.equal(a.gamma[0], torch.zeros_like(a.gamma[0])))

    def test_trunk_is_frozen(self):
        a = adapters.Adapters(self.trunk, self.head, K=2, film=False, in_shape=SHAPE)
        w = [p.detach().clone() for p in a.trunk.parameters()]
        adapters.fit_adapters(a, torch.randn(10, *SHAPE), torch.randint(0, 3, (10,)),
                              torch.zeros(10, dtype=torch.long), lam=0.1, steps=5, lr=1e-2, seed=0)
        self.assertTrue(all(torch.equal(x, p) for x, p in zip(w, a.trunk.parameters())))


if __name__ == "__main__":
    unittest.main()
```
Run: `.venv/bin/python -W ignore -m unittest discover -s research/integration-adapters/tests -v`.
Expected: `ModuleNotFoundError: No module named 'adapters'`.

- [ ] **Step 3: Implement `adapters.py`**
```python
"""Option B: one frozen pooled trunk; per-person head (+ FiLM on the trunk's features)."""
import copy
import torch
from torch import nn


class Adapters(nn.Module):
    def __init__(self, trunk, head, K, film, in_shape):
        super().__init__()
        self.trunk = trunk.eval()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        self.heads = nn.ModuleList(copy.deepcopy(head) for _ in range(K))
        self._pooled = [p.detach().clone() for p in head.parameters()]
        self.film = film
        if film:
            with torch.no_grad():
                ch = self.trunk(torch.zeros(1, *in_shape)).shape[1]
            self.gamma = nn.Parameter(torch.zeros(K, ch))       # scale = 1 + gamma
            self.beta = nn.Parameter(torch.zeros(K, ch))

    def person_logits(self, f, k):
        if self.film:
            shape = (1, -1) + (1,) * (f.dim() - 2)
            f = f * (1 + self.gamma[k].view(shape)) + self.beta[k].view(shape)
        return self.heads[k](f)

    def forward(self, X):
        self.trunk.eval()
        f = self.trunk(X)                                      # one trunk pass for every person
        return torch.stack([torch.log_softmax(self.person_logits(f, k), -1)
                            for k in range(len(self.heads))], 1)


def fit_adapters(model, X, y, person, lam, steps, lr, seed):
    """Per person: full-batch Adam on its head (+ its FiLM row), L2-pulled to the pooled head."""
    torch.manual_seed(seed)
    with torch.no_grad():
        f_all = model.trunk(X)
    for k in torch.unique(person).tolist():
        m = person == k
        params = list(model.heads[k].parameters()) + ([model.gamma, model.beta] if model.film else [])
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model.person_logits(f_all[m], k), y[m])
            loss = loss + 0.5 * lam * sum(((p - p0) ** 2).sum()
                                         for p, p0 in zip(model.heads[k].parameters(), model._pooled))
            if model.film:
                loss = loss + 0.5 * lam * ((model.gamma[k] ** 2).sum() + (model.beta[k] ** 2).sum())
            loss.backward()
            if model.film:                                     # only person k's FiLM row moves
                for p in (model.gamma, model.beta):
                    keep = torch.zeros_like(p); keep[k] = 1
                    p.grad *= keep
            opt.step()
    return model
```
With Adam, a parameter row whose gradient stays zero keeps zero moments, so it does not move.
Masking person k's FiLM row is therefore enough to keep every other person's adapter fixed.

- [ ] **Step 4: Tests OK; commit protocol, then code.**
- [ ] **Step 5: First experiment: B1 and B2, EEGNet trunk, three seeds, all datasets plus sim2.**
  Compare with row 2 and apply `metrics.adopt`. Write the analysis, commit results, and continue
  as an autoresearch loop (next: the ShallowFBCSPNet trunk, per protocol 00).

---

### Task 8: Loop C bootstrap: identity as an input (`int-conditioned`)

Run in its own session in `.claude/worktrees/t2-int-conditioned` (branch
`exp/integration-conditioned`).

**Files:**
- Create: `research/integration-conditioned/{research-state.yaml, research-log.md, findings.md}`
- Create: `research/integration-conditioned/experiments/00-protocol/protocol.md`
- Create: `research/integration-conditioned/src/conditioned.py`, `research/integration-conditioned/tests/test_conditioned.py`

**Interfaces:**
- Consumes:
  - `nets.FILM_AFTER`, `nets.feature_channels`;
  - harness pooled models;
  - bank keys `fp_cv_logp_calib`, `calib_idx`, `fp_logp`, `riemann_logp`, `rel`;
  - `data.load`.
- Produces:
  - `Conditioned(base, arch, K, emb_dim=8)`, an `nn.Module` with
    `forward(X, q: (B, K) identity posterior) -> (B, C)` logits;
  - `train_conditioned(net, X, y, q, id_dropout=0.2, lr=1e-4, epochs=50, bs=32, seed=0)`;
  - `run_dataset(name, seed) -> (n_test, C)` log-probabilities.

- [ ] **Step 1: Protocol 00 (commit first):**
  - C-alone and C+T (log-linear with the fingerprint-routed Riemannian experts, 2 coefficients
    + class bias, fitted LODO);
  - identity dropout 0.2; embedding dimension 8; init from the harness pooled EEGNet; FiLM
    initialised to identity;
  - the train–test gap reported per dataset;
  - the metric and adoption rule from the spec.

- [ ] **Step 2: Failing tests** `research/integration-conditioned/tests/test_conditioned.py`:
```python
import sys, unittest
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parents[1]; REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "track2"), str(REPO / "research" / "integration-harness" / "src"), str(HERE / "src")]
import conditioned  # noqa: E402
from models import make_model  # noqa: E402


class TestConditioned(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.base = make_model("eegnet", 8, 3, 480, 120.0).eval()

    def test_identity_film_init_equals_base(self):
        net = conditioned.Conditioned(self.base, "eegnet", K=4).eval()
        x = torch.randn(5, 8, 480); q = torch.softmax(torch.randn(5, 4), -1)
        with torch.inference_mode():
            self.assertTrue(torch.allclose(net(x, q), self.base(x), atol=1e-6))

    def test_uniform_code_is_the_dropout_code(self):
        net = conditioned.Conditioned(self.base, "eegnet", K=4)
        u = torch.full((2, 4), 0.25)
        self.assertTrue(torch.allclose(net.code(u), net.code(conditioned.uniform_like(u))))

    def test_training_changes_output_and_handles_three_classes(self):
        net = conditioned.Conditioned(self.base, "eegnet", K=2)
        X = torch.randn(24, 8, 480); y = torch.randint(0, 3, (24,)); q = torch.softmax(torch.randn(24, 2), -1)
        before = net(X[:4], q[:4]).detach().clone()
        conditioned.train_conditioned(net, X, y, q, epochs=2, bs=8, seed=0)
        out = net(X[:4], q[:4])
        self.assertEqual(tuple(out.shape), (4, 3))
        self.assertFalse(torch.allclose(before, out))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Implement `conditioned.py`**
```python
"""Option C: one network; identity code e(x) = Σ_k q_k(x)·E_k injected by FiLM."""
import copy
import torch
from torch import nn
import nets


def uniform_like(q):
    return torch.full_like(q, 1.0 / q.shape[1])


class Conditioned(nn.Module):
    def __init__(self, base, arch, K, emb_dim=8):
        super().__init__()
        self.layers = nn.ModuleDict(copy.deepcopy(base).named_children())
        self.order = [n for n, _ in base.named_children()]
        self.after = nets.FILM_AFTER[arch]
        ch = nets.feature_channels(arch)
        self.E = nn.Parameter(0.1 * torch.randn(K, emb_dim))
        self.film = nn.Linear(emb_dim, 2 * ch)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def code(self, q):
        return q.to(self.E.dtype) @ self.E

    def forward(self, X, q):
        gb = self.film(self.code(q))
        gamma, beta = gb.chunk(2, -1)
        h = X
        for n in self.order:
            h = self.layers[n](h)
            if n == self.after:
                shape = (h.shape[0], -1) + (1,) * (h.dim() - 2)
                h = h * (1 + gamma.view(shape)) + beta.view(shape)
        return h


def train_conditioned(net, X, y, q, id_dropout=0.2, lr=1e-4, epochs=50, bs=32, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(X), generator=g).split(bs):
            qb = q[idx].clone()
            drop = torch.rand(len(idx), generator=g) < id_dropout
            qb[drop] = uniform_like(qb[drop]) if drop.any() else qb[drop]
            opt.zero_grad()
            nn.functional.cross_entropy(net(X[idx], qb), y[idx]).backward()
            opt.step()
    return net.eval()
```

- [ ] **Step 4: Tests OK; commit protocol, then code.**
- [ ] **Step 5: First experiment.** C-alone and C+T, EEGNet base, three seeds, all datasets plus
  sim2:
  - train on calibration windows with the bank's `fp_cv_logp_calib` posteriors;
  - test with the bank's `fp_logp`;
  - compare with row 2 and apply `metrics.adopt`;
  - report the train–test gap;
  - commit results and continue as an autoresearch loop.

---

### Task 8b: Loop D bootstrap: batch-level identity prior (`int-sequence`)

Run in its own session in `.claude/worktrees/t2-int-sequence` (branch `exp/integration-sequence`).
Spec: Section 4b. **Not shippable** until the organisers confirm sealed-phase batch composition.

**Files:**
- Create: `research/integration-sequence/{research-state.yaml, research-log.md, findings.md}`
- Create: `research/integration-sequence/experiments/00-protocol/protocol.md`
- Create: `research/integration-sequence/src/grammar.py`, `research/integration-sequence/tests/test_grammar.py`

**Interfaces:**
- Consumes:
  - bank keys `fp_feat_calib`, `person_calib`, `unit_calib`, `fp_feat_test`, `fp_logp`,
    `person`, `y`;
  - `metrics.c3_logp`, `metrics.route`, `metrics.person_scores`, `metrics.pooled_comparison`.
- Produces:
  - `fit_states(Z, K=16, seed=0) -> (mu, sd, centroids)`;
  - `memberships(Z, mu, sd, centroids, gamma=0.05) -> (n, K)`;
  - `transition_matrix(W, eps=1e-3) -> (K, K)`;
  - `grammars(W_cal, person_cal, unit_cal, P) -> (P, K, K)`: per person, pooled over calibration
    units, never across a unit boundary;
  - `batch_prior(W_b, fp_logp_b, G, arm) -> (P,)` log-prior, arm ∈ {"D1", "D2", "D3"};
  - `gate(fp_logp_b, thresh=0.8) -> bool`;
  - `routed(bank, arm, lam, batches) -> (n, C)` log-probabilities;
  - `make_batches(n, size, mode, person=None, seed=0) -> list[np.ndarray]`, mode ∈ {"kit",
    "shuffled", "mixed"}.

- [ ] **Step 1: Protocol 00 (commit first).**
  - K = 16 and γ = 0.05 fixed a priori (neuralprint's non-overlapping settings); arms D1, D2, D3;
    λ fitted leave-one-dataset-out on the kit batching.
  - The robustness suite: kit batches of 64; shuffled; two-person mixed; sizes 8 and 16.
  - The metric and candidate rule from the spec, plus the gate check: D must not score below row 2
    in any robustness condition.
  - Attribution: the grammar code is adapted from neuralprint (commit `fad313c`).

- [ ] **Step 2: Failing tests** `research/integration-sequence/tests/test_grammar.py`:
```python
import sys, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import grammar  # noqa: E402


def chain(P, T, rng):
    s = [0]
    for _ in range(T - 1):
        s.append(rng.choice(len(P), p=P[s[-1]]))
    return np.eye(len(P))[s]


class TestGrammar(unittest.TestCase):
    def test_transition_rows_sum_to_one(self):
        W = np.random.default_rng(0).dirichlet(np.ones(4), 30)
        np.testing.assert_allclose(grammar.transition_matrix(W).sum(1), 1, atol=1e-9)

    def test_batch_prior_prefers_the_generating_grammar(self):
        rng = np.random.default_rng(1)
        Pa = np.array([[.9, .1, 0, 0], [0, .9, .1, 0], [0, 0, .9, .1], [.1, 0, 0, .9]]) + 1e-3
        Pb = Pa[:, ::-1].copy()
        Pa /= Pa.sum(1, keepdims=True); Pb /= Pb.sum(1, keepdims=True)
        G = np.stack([grammar.transition_matrix(chain(Pa, 400, rng)),
                      grammar.transition_matrix(chain(Pb, 400, rng))])
        W = chain(Pa, 64, rng)
        lp = grammar.batch_prior(W, np.log(np.full((64, 2), .5)), G, arm="D2")
        self.assertGreater(lp[0], lp[1])

    def test_gate_rejects_mixed_batches(self):
        a = np.log(np.tile([.9, .1], (32, 1))); b = np.log(np.tile([.1, .9], (32, 1)))
        self.assertTrue(grammar.gate(a))
        self.assertFalse(grammar.gate(np.vstack([a, b])))

    def test_lambda_zero_equals_per_window_routing(self):
        rng = np.random.default_rng(2)
        n, P, C = 40, 3, 2
        lsm = lambda z: z - np.log(np.exp(z).sum(-1, keepdims=True))  # noqa: E731
        bank = {"fp_logp": lsm(rng.standard_normal((n, P))), "eeg_logp": lsm(rng.standard_normal((n, P, C))),
                "riemann_logp": lsm(rng.standard_normal((n, P, C))), "rel": np.full(P, .7),
                "fp_feat_test": rng.standard_normal((n, 5)), "y": rng.integers(0, C, n)}
        st = grammar.fit_states(rng.standard_normal((60, 5)), K=4)
        G = np.stack([np.full((4, 4), .25)] * P)
        out = grammar.routed(bank, "D3", 0.0, grammar.make_batches(n, 8, "kit"), st, G)
        sys.path.insert(0, str(HERE.parents[1] / "integration-harness" / "src"))
        import metrics
        np.testing.assert_allclose(out, metrics.route(bank["fp_logp"], metrics.c3_logp(bank)), atol=1e-9)

    def test_mixed_batches_alternate_people(self):
        person = np.repeat([0, 1], 20)
        b = grammar.make_batches(40, 8, "mixed", person=person, seed=0)
        self.assertTrue(all(len(set(person[i].tolist())) == 2 for i in b))


if __name__ == "__main__":
    unittest.main()
```
Run: `.venv/bin/python -W ignore -m unittest discover -s research/integration-sequence/tests -v`.
Expected: `ModuleNotFoundError: No module named 'grammar'`.

- [ ] **Step 3: Implement `grammar.py`**
```python
"""Option D: batch-level identity prior from a Transition Grammar Biometric Prior.

Adapted from neuralprint (`experiments/h-bnci2015-transition-fingerprint/code/run_experiment.py`,
commit fad313c): soft k-means states on fingerprint tangent features, Dirichlet-smoothed Markov
transition grammars. Here a grammar scores the short window sequence inside one predict() batch."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integration-harness" / "src"))


def fit_states(Z, K=16, seed=0):
    from sklearn.cluster import KMeans
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit((Z - mu) / sd)
    return mu, sd, km.cluster_centers_


def memberships(Z, mu, sd, centroids, gamma=0.05):
    d = np.linalg.norm(((Z - mu) / sd)[:, None] - centroids[None], axis=-1)
    e = np.exp(-gamma * (d - d.min(1, keepdims=True)))          # stable softmax of −γ·d
    return e / e.sum(1, keepdims=True)


def transition_matrix(W, eps=1e-3):
    if len(W) < 2:
        return np.full((W.shape[1], W.shape[1]), 1.0 / W.shape[1])
    M = W[:-1].T @ W[1:] + eps
    return M / M.sum(1, keepdims=True)


def grammars(W_cal, person_cal, unit_cal, P):
    K = W_cal.shape[1]
    G = np.zeros((P, K, K))
    for k in range(P):
        M = np.full((K, K), 1e-3)
        for u in np.unique(unit_cal[person_cal == k]):
            W = W_cal[(person_cal == k) & (unit_cal == u)]
            if len(W) > 1:
                M += W[:-1].T @ W[1:]                            # no transition across units
        G[k] = M / M.sum(1, keepdims=True)
    return G


def batch_prior(W_b, fp_logp_b, G, arm):
    lp = np.zeros(G.shape[0])
    if arm in ("D2", "D3") and len(W_b) > 1:
        T = W_b[:-1].T @ W_b[1:]                                 # expected transition counts
        lp = lp + (T[None] * np.log(G)).sum((1, 2))
    if arm in ("D1", "D3"):
        lp = lp + fp_logp_b.sum(0)
    return lp - np.logaddexp.reduce(lp)


def gate(fp_logp_b, thresh=0.8):
    top = fp_logp_b.argmax(1)
    return np.bincount(top).max() / len(top) >= thresh


def make_batches(n, size, mode, person=None, seed=0):
    idx = np.arange(n)
    if mode == "shuffled":
        idx = np.random.default_rng(seed).permutation(n)
    elif mode == "mixed":                                        # interleave two people per batch
        people = np.unique(person)
        rng = np.random.default_rng(seed)
        pools = {p: list(rng.permutation(np.where(person == p)[0])) for p in people}
        out, order = [], list(people)
        while any(pools.values()):
            a, b = order[0], order[1 % len(order)]
            take = [pools[a].pop() for _ in range(min(size // 2, len(pools[a])))]
            take += [pools[b].pop() for _ in range(min(size - len(take), len(pools[b])))]
            if take:
                out.append(np.array(take))
            order = [p for p in order[2:] + order[:2] if pools[p]] or order
            if not any(pools[p] for p in order):
                break
        return out
    return [idx[i:i + size] for i in range(0, n, size)]


def routed(bank, arm, lam, batches, states, G):
    import metrics
    per_person = metrics.c3_logp(bank)                           # (n, P, C)
    W = memberships(bank["fp_feat_test"], *states)
    fp = bank["fp_logp"].copy()
    for b in batches:
        if lam == 0 or not gate(bank["fp_logp"][b]):
            continue
        prior = batch_prior(W[b], bank["fp_logp"][b], G, arm)
        z = bank["fp_logp"][b] + lam * prior[None]
        fp[b] = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
    return metrics.route(fp, per_person)
```
D2 scores the batch's expected transition counts against log G, i.e. the sequence log-likelihood
under each person's grammar. This differs from neuralprint's cosine match of whole-session matrices
because a 64-window batch is too short to estimate its own K × K matrix.

- [ ] **Step 4: Tests OK; commit protocol, then code.**
- [ ] **Step 5: First experiment.**
  - D1, D2 and D3 on all three datasets (seeds 0–2) plus sim2.
  - Run the full robustness suite and compare with row 2; λ fitted LODO on kit batching.
  - Report whether D2/D3 beat D1, which tests whether the grammar adds anything beyond pooling.
  - Commit results; continue as an autoresearch loop.

---

### Task 9: Final comparison and the one holdout confirmation (after all four loops conclude)

- [ ] **Step 0: Option D.** D is ship-eligible only after the organisers confirm sealed-phase
  `predict()` batch composition, and only if it passed every robustness condition. If D is a
  candidate, also score "D + the A/B/C winner" on dev, pre-registered like the rest.
- [ ] **Step 1: Pick on dev.** Collect each loop's dev verdict (candidate or not). If several are
  candidates, choose the one with the highest dev CI lower bound (ties → cheaper: A < B < C). If
  none, record that the current integration stands, and stop here; the holdout stays unused.
- [ ] **Step 2: Pre-register (commit before running).**
  `research/integration-harness/experiments/holdout-confirm/protocol.md` names:
  - the one candidate, with its exact variant and parameters, fitted on all three datasets' dev
    people;
  - the comparison against row 2 on the 8 holdout people;
  - the adoption bar (holdout mean gain ≥ +0.005; CI reported, not required);
  - the BNCI 2015-001 session 1 → 3 check for people 8–11 (reported).
- [ ] **Step 3: Rebuild the shared parts with everyone.**
  `pipeline.run(data.load(name, part="all"), seed, …)` for seeds 0–2, so the fingerprint and
  pooled model know all people, as the sealed phase would. This writes `bank_{name}_all_seed{s}.npz`.
  Then apply the candidate and score **only** the holdout people, plus row 2 on them.
- [ ] **Step 4: Decide and record.** Adopt if the bar is met. The adopted option gets its own
  short plan for `submission.py` integration and the offline kit contract check. Either way,
  commit the result with its CI, and update `track2/README.md`.
