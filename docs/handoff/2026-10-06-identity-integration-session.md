# Handoff: identity-integration work (session of 2026-09-30 → 2026-10-06)

Read this first, then the spec and the plan below. Everything here is committed on `main`
(00b4a08), except this file, which is the first commit on `exp/integration-harness`.

## Start here

```bash
cd /home/promit/Documents/ephys_fm/.claude/worktrees/t2-int-harness   # branch exp/integration-harness
claude
```
Prompt: *"Read docs/handoff/2026-10-06-identity-integration-session.md, then execute
docs/superpowers/plans/2026-10-05-identity-integration.md from Task 1 with
superpowers:executing-plans (native execution, as the user chose for loop C)."*

- **Spec:** `docs/superpowers/specs/2026-10-05-identity-integration-design.md`.
- **Plan:** `docs/superpowers/plans/2026-10-05-identity-integration.md`.
- **Holdout (locked):** `docs/superpowers/specs/2026-10-05-identity-integration-holdout.json`.

## What the next work is

The shipped Track 2 model routes each window, by a fingerprint's p(person | window), over K
per-person copies of each expert. The user wants identity used **more robustly**, with
**cross-day drift** as the primary target. Three options run as three parallel autoresearch
loops, each with its own branch, worktree and workspace:

- **A, gate:** a pooled fallback weighted by the fingerprint's uncertainty. Branch
  `exp/integration-gate`, worktree `t2-int-gate`. **A starts first** (user's choice).
- **B, adapters:** one shared trunk with per-person heads, optionally FiLM on the trunk's
  features. Branch `exp/integration-adapters`, worktree `t2-int-adapters`. Also settles whether a
  ShallowFBCSPNet trunk helps.
- **C, conditioned:** one network with an identity code from the fingerprint posterior, injected
  by FiLM and trained with identity dropout. Branch `exp/integration-conditioned`, worktree
  `t2-int-conditioned`.

**Plan order:**
1. **Tasks 1–5:** the shared harness, in this worktree. Data loader; pipeline that caches banks to
   `outputs/t2-int-harness/`; metrics and adoption rule; trunk/head split; reference rows; the
   loop hand-off brief; PR.
2. **Tasks 6–8:** each loop's bootstrap, in its own session and worktree, after the harness PR
   merges. The user starts those sessions (`cd .claude/worktrees/<name> && claude`) with the
   prompt in the Task 5 hand-off brief.
3. **Task 9:** pick the winner on dev; confirm it **once** on the holdout.

**Evaluation (binding):**
- Datasets: BNCI 2014-001 (s1 → s2), BNCI 2015-001 (s1 → s2), Zhou 2016 (s1–2 → s3).
- Real fingerprint errors; no oracle identity.
- **Decisions on 17 dev people only.** The 8 holdout people (BNCI14 2, 3, 9; BNCI15 8–11; Zhou 1)
  are never loaded by a loop: `data.load(name)` defaults to `part="dev"`.
- Candidate rule (dev): pooled gain over row 2 ≥ 0.005 with person-bootstrap CI lower bound > 0,
  and Dreyer sim2 not worse by more than 0.01.
- Adoption: the chosen candidate's holdout mean gain ≥ +0.005.
- Learned integration parameters are fitted leave-one-dataset-out.

**Already verified in the plan:** the option code (gate, adapters, conditioned), the harness
metrics, nets and the data loader were dry-run from the plan's code blocks (20 tests pass; the
data loader ran on the real caches). Task 2's pipeline test is the first code not yet run.

## State of the repo

- **Merged:** #7 (loop A fingerprint), #8 (loop B combiner C3), #9 (both shipped in
  `submission.py`), #10 (onboarding docs), #11 (identity-integration spec + plan + holdout),
  #12 (loop C expert portfolio).
- **Worktrees:**
  - `t2-int-harness` (this one, new);
  - `t2-fingerprint`, another session's line of work (now on `exp/expert-occlusion`): **don't
    touch**;
  - `data-atlas`, another line of work, with atlas jobs often running: **don't touch**;
  - `track2`, a general worktree.
- **Untracked, the user's own (do not commit or move):** `docs/alt-arch.md`,
  `docs/imagery-window-and-cue-evidence.md` and `docs/model-summary-and-assumptions.md` in the
  main checkout.
- **Artifacts used by the harness:**
  - `data/experiments/tangermann_windows.npz`;
  - `outputs/t2-fingerprint/cache/{BNCI2015_001,Zhou2016}_windows.npz`;
  - `data/experiments/dreyer_windows.npz`;
  - sim2 people: `research/expert-portfolio/experiments/00-protocol/sim2_people.json`.

  All read-only.

## What we know (numbers to carry)

| Fact | Value | Source |
|---|---|---|
| Shipped `predict`, Dreyer R4–R6, seed 0 | 0.921 (EEGNet experts alone 0.907) | #9 |
| **R4–R6 are upper bounds** (reused by loops A, B, C) | — | loop C |
| Dreyer sim2 (14 fresh people), honest | EEGNet experts 0.9355; + Riemannian C3 0.9365 (+0.001, CI [−0.004, +0.007]) | loop C |
| C3 across days (BNCI 2014-001, 5 seeds) | +0.052 over EEGNet experts | loop B |
| ShallowFBCSPNet as a 2nd per-person stream | dev +0.011; sim2 −0.005; BNCI −0.004 → **not shipped** (user decision) | loop C |
| Leakage-free page (warm-up train split only) | pooled ShallowFBCSPNet **0.844**, kit EEGNet 0.795, filter-bank Riemann 0.685, MeanLogReg 0.681, REVE probe (mean-pooled) 0.547 | `track2/bench/README.md` |
| Fingerprint across days (loop A) | 1 calibration day 0.79–0.94; 2 days 0.975–1.000 | loop A |

## Gotchas learned the hard way

- **Background chains:** `A=… && B=… && cmd &` backgrounds the whole list, so later lines don't
  see A and B. Set the variables on their own lines, then `setsid nohup bash -c "…" < /dev/null &`.
  Find the Python PID with `ps -C python -o pid,args`. Harness waits are capped at 2 h; re-arm
  them.
- **`external/` is a symlink to the main checkout.** `../../track2` from inside the kit resolves to
  the main checkout, not your worktree: pass solver files by absolute path.
- **REVE offline:** `HF_HUB_OFFLINE` is read when `huggingface_hub` is imported, so setting it
  later does nothing. `load_reve_encoder` passes `local_files_only=True`, and ships
  `reve_positions.json` + `reve_kwargs.json` (via `REVE_POSITIONS_PATH`).
- **Class counts:** every combiner, bias or fingerprint must be tested with C = 2 **and** C ≥ 3.
  Two bugs were of this kind: the research combiner bias (fixed), and `fit_fingerprint` for
  2 classes (fixed).
- **`train_mixture.py`:**
  - Only the original design may write `outputs/track2_dreyer_sim`; others need `--out` (sim2
    defaults to `…/sim2`).
  - Extra streams require `--combiner portfolio --portfolio-coef FILE`, where the JSON carries
    `"streams"`.
  - `masks()` rejects unknown and val people.
- **CPU:** 8 threads, shared with the atlas and loop A sessions. ≤ 2 threads per job for loop
  work; check `uptime` before launching.
- **Usage limits** hit several times. Keep ledgers and state files current, so a fresh session can
  resume from `git log` plus the ledger (`.superpowers/sdd/<plan>/progress.md` in each worktree).
- **Every protocol is committed before its results.** R4–R6 are never used for decisions.

## Deferred (known, small)

From loop C's final review (`research/expert-portfolio/findings.md` has the context):
- **M5:** `ReveProbe.state_dict` override only applies at the top level.
- **M6:** a flat non-zero-DC channel gets resampling edge transients on the REVE path.
- **M7:** the bench `riemann_solver` lacks a channel-order assert.
- **M8:** P0's `offline_load: ok` predates the `local_files_only` fix.

Open research questions:
- a REVE probe that keeps per-channel features;
- a warm-up-legitimate submission built on a pooled ShallowFBCSPNet (0.844 cross-subject).
