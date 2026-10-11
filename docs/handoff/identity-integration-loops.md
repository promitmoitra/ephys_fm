# Hand-off: starting the identity-integration loops (A, B, C, D)

The shared harness (`exp/integration-harness`) is merged. Each option now runs as its own
autoresearch loop, in its own worktree, branch, workspace and Claude Code session.

- **Spec:** `docs/superpowers/specs/2026-10-05-identity-integration-design.md` (Sections 1–5).
- **Plan:** `docs/superpowers/plans/2026-10-05-identity-integration.md` (Tasks 6, 7, 8, 8b; Task 9 at
  the end).
- **Reference rows:** `research/integration-harness/results/references.md` (+ `.json`).
- **Order:** A starts first; B, C and D start in parallel once the harness is on `main`.

## What the reference rows say (read before writing a protocol)

From `research/integration-harness/results/references.md` (17 dev people, seeds 0–2):
- The current design (row 2) beats the pooled EEGNet (row 1) by **+0.061** [+0.038, +0.083], with
  15 people better and 2 worse.
- Perfect identity (row 3) beats the current design by only **+0.007** [0.000, +0.021], with 4 people
  better and 1 worse. BNCI 2014-001 has no headroom at all: its fingerprint is 1.000 across days.
- So, on dev, any gain from routing more accurately is capped at about +0.007. That is barely above
  the candidate rule's +0.005. **D**, and the routing side of **A**, are unlikely to clear the rule on
  routing alone. The room to improve is in *how* identity is used: A's pooled fallback where
  experts are weak, and B's and C's shared networks.
- On Dreyer sim2, row 2 is 0.936 (fingerprint 1.000; pooled 0.930).

## The four loops

| Option | Loop | Branch | Worktree | Workspace | Artifacts | Plan task |
|---|---|---|---|---|---|---|
| A, gated pooled fallback | `int-gate` | `exp/integration-gate` | `.claude/worktrees/t2-int-gate` | `research/integration-gate/` | `outputs/t2-int-gate/` | 6 |
| B, per-person adapters | `int-adapters` | `exp/integration-adapters` | `.claude/worktrees/t2-int-adapters` | `research/integration-adapters/` | `outputs/t2-int-adapters/` | 7 |
| C, identity as an input | `int-conditioned` | `exp/integration-conditioned` | `.claude/worktrees/t2-int-conditioned` | `research/integration-conditioned/` | `outputs/t2-int-conditioned/` | 8 |
| D, batch-level identity prior | `int-sequence` | `exp/integration-sequence` | `.claude/worktrees/t2-int-sequence` | `research/integration-sequence/` | `outputs/t2-int-sequence/` | 8b |

**D is not shippable** until the competition organisers confirm that sealed-phase `predict()`
batches hold consecutive windows from one recording, in order. It is still evaluated like A, B
and C. K = 16 and γ = 0.05 are fixed a priori (spec Section 4b).

## Start commands

From the root of the main checkout, on an up-to-date `main` (example for A; substitute the
worktree and branch from the table):

```bash
cd /home/promit/Documents/ephys_fm && git branch --show-current   # main
git pull --ff-only origin main
git worktree add .claude/worktrees/t2-int-gate -b exp/integration-gate main
for d in data external .venv outputs submissions; do
  ln -s /home/promit/Documents/ephys_fm/$d .claude/worktrees/t2-int-gate/$d; done
mkdir -p outputs/t2-int-gate
cd .claude/worktrees/t2-int-gate && claude
```

Add a `CLAUDE.md` row for the worktree when it is created (`CLAUDE.md` rule 6), and remove it after
the loop's merge.

## Prompt for each session

> Read docs/superpowers/specs/2026-10-05-identity-integration-design.md and
> docs/superpowers/plans/2026-10-05-identity-integration.md (Task 6, 7 or 8), and run the loop
> with the autoresearch skill. Set up its 20-minute heartbeat first.

Use Task 6 for A, 7 for B, 8 for C and 8b for D.

## Shared rules (binding for every loop)

- **The banks in `outputs/t2-int-harness/` are read-only.** No loop retrains the shared parts
  (fingerprint, pooled EEGNet, per-person EEGNet experts, Riemannian experts). Files:
  `bank_<dataset>_seed<s>.npz`, `pooled_<dataset>_seed<s>.pt`, `fp_<dataset>_seed<s>.pt`, for
  `bnci2014`, `bnci2015`, `zhou2016` (seeds 0, 1, 2) and `dreyer_sim2` (seed 0).
- **Dataset caches are read-only** too.
- **≤ 2 CPU threads per job** (`OMP_NUM_THREADS=2`, `--threads 2`). Check `uptime` before
  launching. Long jobs: `setsid nohup .venv/bin/python … > log 2>&1 < /dev/null &`, then track the
  PID with `ps -C python -o pid,args`. Never `A=… && B=… && cmd &`.
- **Protocols before results:** each loop commits `experiments/00-protocol/protocol.md` before it
  runs anything that produces a number it decides on.
- **Holdout:** the 8 people in `docs/superpowers/specs/2026-10-05-identity-integration-holdout.json`
  are never loaded by a loop. `data.load(name)` defaults to `part="dev"`; never pass
  `part="holdout"` or `part="all"` outside Task 9.
- **Dreyer R4–R6 are not used** (reused hidden runs; upper bounds only).
- **No leakage:** learned integration parameters are fitted on calibration data, or
  leave-one-dataset-out (fit on two datasets' dev test sessions, score on the third).
- **Classes:** every combiner, gate or head is tested with C = 2 **and** C ≥ 3.

## The primary metric, with `metrics.py`

`research/integration-harness/src/metrics.py`:
- `rows(bank)` gives the reference rows as `(n, C)` log-probabilities: `"pooled"` (row 1),
  `"current"` (row 2, E+T with C3 as shipped, soft-routed by the real fingerprint) and `"oracle"`
  (row 3).
- `person_scores(y, person, logp, K)` gives per-person balanced accuracy on the test session.
- `pooled_comparison({name: (option_scores, row2_scores)})` gives the paired gain over the
  17 dev people, with a person-bootstrap 95% CI (5,000 resamples, rng seed 0) and better/worse
  counts. People are never matched across datasets.
- `adopt(cmp, sim2_delta)` applies the candidate rule: gain ≥ 0.005, CI lower bound > 0, and Dreyer
  sim2 not worse than row 2 by more than 0.01.

Recipe, per dataset:

```python
import numpy as np, data, metrics
for s in (0, 1, 2):
    b = dict(np.load(f"outputs/t2-int-harness/bank_{name}_seed{s}.npz"))
    K = b["eeg_logp"].shape[1]
    ref = metrics.person_scores(b["y"], b["person"], metrics.rows(b)["current"], K)
    opt = metrics.person_scores(b["y"], b["person"], my_option_logp(b), K)
# person score = mean over the 3 seeds, then
cmp = metrics.pooled_comparison({name: (opt_mean, ref_mean) for name in data.DATASETS})
```

`research/integration-harness/src/run_refs.py` is the worked example.

## Finish

Each loop concludes with findings, a report in `research/<loop>/to_human/` and a PR. Task 9 then
picks at most one candidate on dev and confirms it **once** on the holdout people, with its
protocol committed first.
