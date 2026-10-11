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

**For loop A: is the pooled EEGNet undertrained?** Checked, no. Its best validation epoch is late on
Zhou 2016 (143–145 of 150) and on BNCI 2014-001 seed 0 (149), but not elsewhere (BNCI 2014 seeds
1–2 at 130, BNCI 2015 at 85–132, sim2 at 89 of 100). A 300-epoch rerun, scored on the test session
(scratch only, banks unchanged), gave:
- Zhou, seeds 0–2: pooled +0.020 (all 3 people up), row 2 +0.004, row 3 −0.004;
- BNCI 2014, seed 0: pooled −0.025, row 2 −0.009.

The banks keep 150 epochs. If Zhou's pooled gain is real, it would shrink the current − pooled
gap by about 0.0035 (3 of 17 people). A's fallback may be slightly pessimistic on Zhou only.

**For loop C: the calibration posteriors are nearly perfect.** The banked leave-one-unit-out
fingerprint posteriors on calibration (`fp_cv_logp_calib`) have top-1 of 1.000 (BNCI 2014-001),
0.998 (BNCI 2015-001) and 0.999 (Zhou 2016). Across days, the test session's top-1 is 1.000, 0.952
and 0.747. The held-out unit always comes from the same session as the training units: Zhou's units
are session × run, and BNCI 2015's two "runs" are halves of one session. So these posteriors carry
almost none of the cross-day identity noise that C is meant to learn to tolerate. C's protocol
must handle this before training on them, for example with leave-one-calibration-session-out
posteriors for Zhou, or a temperature or noise level matched to cross-day top-1, fitted
leave-one-dataset-out.

## The four loops

| Option | Loop | Branch | Worktree | Workspace | Artifacts | Plan task |
|---|---|---|---|---|---|---|
| A, gated pooled fallback | `int-gate` | `exp/integration-gate` | `.claude/worktrees/t2-int-gate` | `research/integration-gate/` | `outputs/t2-int-gate/` | 6 |
| B, per-person adapters | `int-adapters` | `exp/integration-adapters` | `.claude/worktrees/t2-int-adapters` | `research/integration-adapters/` | `outputs/t2-int-adapters/` | 7 |
| C, identity as an input | `int-conditioned` | `exp/integration-conditioned` | `.claude/worktrees/t2-int-conditioned` | `research/integration-conditioned/` | `outputs/t2-int-conditioned/` | 8 |
| D, pooled batch prior | `int-sequence` | `exp/integration-sequence` | `.claude/worktrees/t2-int-sequence` | `research/integration-sequence/` | `outputs/t2-int-sequence/` | 8b |

**D is not shippable** until the competition organisers confirm that sealed-phase `predict()`
batches hold consecutive windows from one recording, in order. It is still evaluated like A, B
and C.

D pools the batch's per-window fingerprint posteriors into a gated prior (spec Section 4b,
revision 2026-10-11). Its transition-grammar arms were dropped: neuralprint's runtime audit of
2026-10-09 showed the grammar carried no temporal information (shuffling order changed nothing;
occupancy matched transitions), and its 100% identification figures came from a projection that
leaks identity. D's gain is capped by the +0.007 routing headroom above.

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

- `seed_mean_scores(name, logp_fn, seeds=(0, 1, 2))` loads `bank_<name>_seed<s>.npz` for each seed,
  scores `logp_fn(bank) -> (n, C)` per person and returns the mean over seeds. **Use it**, so the
  person score is always the seed mean.

Recipe (the primary metric for an option):

```python
import data, metrics
cmp = metrics.pooled_comparison({
    name: (metrics.seed_mean_scores(name, my_option_logp),
           metrics.seed_mean_scores(name, lambda b: metrics.rows(b)["current"]))
    for name in data.DATASETS})
ok, why = metrics.adopt(cmp, sim2_delta)   # the dev candidate rule
```

`research/integration-harness/src/run_refs.py` is the worked example for the reference rows.

## Finish

Each loop concludes with findings, a report in `research/<loop>/to_human/` and a PR. Task 9 then
picks at most one candidate on dev and confirms it **once** on the holdout people, with its
protocol committed first.
