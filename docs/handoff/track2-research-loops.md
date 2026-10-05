# Handoff: Track 2 autoresearch loops

> **Status: concluded.** Both loops ran 30 Sep – 1 Oct 2026 and merged (#7, #8); their results
> ship in `track2/submission.py` (#9). This brief stays as the record of their protocol. Thread
> counts and timings below are from the machine the loops ran on.

Two parallel research loops improve the Track 2 (BCI decoding) submission. Each runs in a fresh
Claude Code session inside its own worktree (see `CLAUDE.md`). Read this whole brief, then
`track2/README.md` ("Current best model"), before starting.

| Loop | Worktree | Branch | Goal |
|---|---|---|---|
| A | `.claude/worktrees/t2-fingerprint` | `exp/fingerprint-model` | A better per-window **fingerprint** (person ID) model with well-calibrated probabilities |
| B | `.claude/worktrees/t2-expert-weights` | `exp/expert-weights` | **Learned weights** for combining each person's EEGNet expert with their Riemannian (and CSP) experts |

## Start a loop

From the root of your main checkout (always on `main`):

```bash
git pull --ff-only origin main
git worktree add .claude/worktrees/t2-fingerprint -b exp/fingerprint-model main   # loop B: t2-expert-weights / exp/expert-weights
for d in data external .venv outputs submissions; do
  ln -s "$PWD/$d" .claude/worktrees/t2-fingerprint/$d
done
cd .claude/worktrees/t2-fingerprint && claude
```

Then prompt: *"Read docs/handoff/track2-research-loops.md and run loop A with the autoresearch
skill."* Invoke the skill as `autoresearch:autoresearch` (the bare `0-autoresearch-skill` name does
not resolve). It is a Claude Code plugin installed separately; it does not ship with this repo. Set up its 20-minute `/loop` heartbeat first; it lives only as long as the session.

## Why these two loops

The current best model (`track2/README.md`) is a pooled EEGNet fine-tuned once per evaluation
person, mixed in `predict(X)` by a fingerprint model's p(person | window), since `predict` gets no
IDs. On BNCI this beat an epoch-matched pooled control by +0.034 ± 0.007 (5/5 seeds). Two things
limit it:

- **The fingerprint bounds the gain.** Soft routing kept 62–88% of the oracle-ID gain across BNCI
  seeds, lowest for the weakest fingerprint (0.813). On the Dreyer simulation the fingerprint drops
  from 0.857 on the run it was tuned on (R3) to 0.744 on the hidden runs.
- **The Riemannian expert helped only under a plain average.** Per-person Riemannian models alone
  score 0.688 (oracle) vs 0.908 for the EEGNet experts, yet averaging them lifted the soft-routed
  mixture from 0.901 to 0.909 (11 people better, 9 worse, one seed). CSP added nothing; averaging
  all three hurt (0.833).

## The shared setting: Dreyer sealed-phase simulation

Dreyer 2023 (2-class MI, 27 EEG channels, kit preprocessing: 120 Hz, per-recording robust scaling,
0–4 s from the cue, 480 samples). The kit's test people (subjects 61–81, **21 people**) play the
sealed phase's evaluation participants: runs **R1–R3 = calibration** (labeled), **R4–R6 = hidden
test**. The training pool is the kit's train split (52 people, 12,392 windows); the kit's val
split (14 people, 3,360 windows) picks the pooled model's epoch.

- Window cache (kit-identical): `data/experiments/dreyer_windows.npz`, keys `X (20792, 27, 480)`,
  `y`, `subject` (str), `run` (0–5), `split` (train/val/test), `ch_names`, `sfreq`.
  Test people have 240 windows each, 40 per run.
- Trained mixture (seed 0): `outputs/track2_dreyer_sim/submission/` (`submission.py`,
  `mixture.pt` = `{"fingerprint": sd, "experts": [21 sd]}`, `config.json`; expert order = people
  sorted numerically). Load it with `track2/submission.py:build_model`.
- Reference numbers (seed 0, balanced accuracy on R4–R6, chance 0.5):

| Model | R4–R6 |
|---|---|
| Pooled EEGNet | 0.873 |
| Epoch-matched control (pooled + same fine-tune on all calibration) | 0.890 |
| EEGNet experts, oracle ID | 0.908 |
| **EEGNet experts, soft-routed (current best)** | **0.901** |
| EEGNet + Riemannian experts averaged, soft-routed | 0.909 |
| Fingerprint (21-way) | 0.744 (chance 0.048) |

Reusable code: `track2/train_mixture.py` (data loading, `fit`, packaging), `track2/classical_experts.py`
(per-person CSP / Riemannian experts on 8–30 Hz, 0.5–4 s, and combination scoring),
`track2/submission.py` (the shipped model). Recipes: EEGNet via braindecode, AdamW, weight decay
1e-3; pooled lr 1e-3, batch 64; per-person fine-tune lr 1e-4, batch 32, 50 epochs, **last epoch**.

## Locked evaluation (write it into each protocol before running)

**The hidden runs R4–R6 are never used for a decision.** Iterating against them would overfit
them. Decide on calibration data; confirm on R4–R6 only at outer-loop checkpoints, for a few
pre-registered candidates.

### Loop A: fingerprint

- **Inner-loop metric:** train on R1–R2 of the 21 people (1,680 windows), evaluate on **R3**
  (840 windows): 21-way balanced accuracy **and** negative log-likelihood (soft routing uses the
  probabilities). **Fixed epoch budget, no epoch selection on R3.** The seed-0 fingerprint in
  `mixture.pt` picked its epoch on R3, so its R3 score (0.857) is biased; re-run the baseline
  recipe under this protocol first.
- **Confirmation:** retrain the recipe on R1–R3, score on R4–R6: fingerprint accuracy, and the
  soft-routed mixture with the **fixed** EEGNet experts from `mixture.pt` (baseline 0.744 / 0.901);
  3 seeds. External check: BNCI cross-session subject ID (`data/experiments/tangermann_windows.npz`,
  code in `experiments/fingerprint_tangermann/`; EEGNet 0.886 ± 0.046 over 5 seeds).
- **Hypotheses to start from** (write each as mechanism → prediction):
  1. Train on all of R1–R3 with a fixed epoch budget: 50% more data per person.
  2. Pretrain the trunk on the training pool's 52 identities (66 with the val split), then
     fine-tune on the 21: more
     identity diversity should give features that generalize across runs.
  3. Riemannian fingerprint (covariance → tangent space → logistic regression, broadband and per
     band): spatial covariance is strongly subject-specific and cheap on CPU.
  4. Temperature scaling on a held-out calibration run: better-calibrated probabilities improve
     soft routing even at equal accuracy.
  5. Augmentation for run-to-run drift (channel dropout, amplitude scaling, time shifts, noise).
  6. Ensembles (EEGNet + Riemannian fingerprints, product of experts).

### Loop B: expert weights

- **One-time setup (the expensive part, ~1.5 h):** a *prediction bank*. Weights must be learned
  from calibration data only, so the experts that predict the dev set must not have seen it:
  - dev bank: pooled EEGNet trained on pool + calibration **R1–R2** (epoch on val) → per-person
    fine-tunes on R1–R2 → predict **R3**; Riemannian and CSP experts fitted on R1–R2 → predict R3;
  - test bank: pooled on pool + R1–R3 → fine-tunes on R1–R3 → predict **R4–R6**; classical
    experts fitted on R1–R3 → predict R4–R6;
  - **save** the pooled models and all per-person probabilities (e.g. under
    `outputs/t2-expert-weights/`); `train_mixture.py` does not save the pooled model today.
- **Inner-loop metric:** combiners are learned and scored on the dev bank under the **true ID**
  (identity is known during calibration): leave-one-person-out cross-validation for global
  combiners, within-person 2-fold for per-person ones. Every experiment then takes seconds.
- **Confirmation:** fit the combiner on the full dev bank, apply it to the test bank, soft-routed
  with the fixed fingerprint from `mixture.pt` (baseline 0.901; plain average 0.909); then with
  loop A's best fingerprint.
- **Hypotheses to start from:**
  1. One global weight between EEGNet and Riemannian, chosen by log-loss.
  2. Per-person weights (flexible but noisy: 40 windows per person in R3).
  3. Stacking: logistic regression on the experts' log-probabilities.
  4. Log-linear pooling (weighted sum of log-probabilities) vs linear pooling.
  5. Per-expert temperature scaling before pooling.
  6. Confidence gating (weight each expert by its margin or entropy per window).
  7. Include CSP with learned weights (it should get a small weight).

## Operating rules and gotchas

- **Git:** follow `CLAUDE.md`. Work only in your worktree, check `git branch --show-current`
  before every commit, never bare `git stash`, integrate via PRs into `main` (protected). Per the
  autoresearch skill, commit each protocol **before** its results. A branch created with
  `git switch -c <b> origin/main` tracks `origin/main`: run `git branch --unset-upstream`, then
  push with `git push -u origin <b>`.
- **Workspace:** keep the autoresearch files (`research-state.yaml`, `research-log.md`,
  `findings.md`, `experiments/<hypothesis>/…`, `to_human/`) under `research/` in your worktree.
  Write run artifacts to `outputs/t2-fingerprint/` or `outputs/t2-expert-weights/`; only read
  `data/`.
- **CPU** (the machine the loops ran on): 8 threads, no GPU, 16 GB RAM, shared by both loops: use `--threads 4` /
  `torch.set_num_threads(4)` each. At 8 threads a pooled Dreyer EEGNet epoch took ~19 s (100
  epochs ≈ 31 min) and a fingerprint epoch ~2.3 s; expect slower at 4 threads.
- **Long jobs:** harness background tasks are killed after ~30 min. Launch with
  `setsid nohup python … > log 2>&1 < /dev/null &`, then find the **Python** PID with
  `ps -C python -o pid,args` (`pgrep -f` and `$!` can match the launching shell, which exits
  early and fools a monitor). Monitors expire after 30 min; re-arm them.
- **Reproducibility:** training is deterministic given seed, epochs **and thread count**; changing
  `--threads` changes the numbers slightly.
- **Data traps:** MNE filters skip EOG/EMG unless `picks="all"`. The kit's 0.1 Hz high-pass is
  zero-phase (non-causal), so large post-cue signals smear into pre-cue time; pre-cue baselines of
  the EOG are contaminated. The kit's windows start at the cue, so there is no pre-cue data at
  inference. Dreyer's EOG montage is inverted between parts A (1–60) and B/C (61–87).
- **Never upload** anything trained on the Dreyer simulation to the Codabench warm-up: it uses the
  warm-up test subjects' labels (inflated, leaky score).
- **What the model uses on Dreyer** (`experiments/dreyer_eog/README.md`): mostly an early (~250 ms)
  cue-locked response, plus eye position and some wrist EMG; genuine motor imagery is present but
  person-specific. Within a person, CSP / Riemannian spatial filters are the best pure-imagery
  decoders (CSP + LDA 0.628 on 1.25–4 s).
- **Already ruled out** (don't re-test without a new reason): per-person heads trained from
  scratch, BatchNorm-statistics-only adaptation, epoch selection on small validation sets, hard
  routing (≤ soft), explicit hemispheric-asymmetry features, per-trial pre-cue baseline correction.

## When a loop concludes

Update `track2/README.md` ("Current best model") and the experiments table in `README.md`, open a
PR from your branch, and remove the worktree after merge (`CLAUDE.md` rule 9). If both loops
improve, the next step is integrating the best fingerprint and combiner into `track2/submission.py`
(weights as tensors, no sklearn/pyriemann objects at inference) and re-running the kit contract
check.
