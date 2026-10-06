# Onboarding

Welcome. This repository is our entry for the
[EEG/EMG Foundation Challenge 2026](https://neural-interfaces26.github.io/) (NeurIPS 2026 Brain &
Body Workshop). This page covers what to set up, how we work together, and where things stand.
State as of 4 Oct 2026.

## 1. The competition in brief

Four independent tracks, each its own Codabench competition, ranked separately. A team may enter
any or all of them (table and metrics in the [README](../README.md)).

| Track | Task | Status here |
|---|---|---|
| 1 EEG-to-Image | retrieve the viewed image's DINOv2 embedding from EEG | not started |
| 2 BCI decoding | classify the user's mental command from a short EEG window | main effort; submission built and validated |
| 3 Sleep onset | predict seconds to the first N2 sleep stage | data prepared, constant floor only |
| 4 EMG-to-Pose | predict hand joint angles from forearm EMG | not started |

**Dates.** Registration closes **24 Oct**. Warm-up runs until 25 Oct (5 submissions per person
per day, public data, scores are indicative only). The **sealed phase**, 28 Oct – 21 Nov (AoE),
is the only one that ranks: 1 submission per person per day, ranked on the best of the team's
last five sealed submissions. Winners are announced at NeurIPS (11–12 Dec).

**Rules that shape our work** (Codabench *Terms* and *Submission Guide*; the kit is
[neural-interfaces26/2026-competition](https://github.com/neural-interfaces26/2026-competition)):

- A submission is a ZIP with `submission.py` (`class Solver(CompetSolver)`, all inference code)
  and its weights at the ZIP root. It may only import what the worker image carries (`torch`,
  `braindecode` and the rest of the kit's `requirements.txt`). Nothing is trained at evaluation.
- Inference must finish in 60 min on one A100. 15 GB of submission storage per Codabench profile,
  shared across tracks.
- Pretraining on any public dataset is allowed; **declare every external dataset and a compute
  estimate**.
- **Reproducibility audit:** for the top three teams per track, the organisers rerun our training
  pipeline, and the rerun must score within ±2σ of the submitted score. Keep every result
  reproducible from committed code, a seed and a thread count.

**Track 2 in the sealed phase:** 3 classes (kinesthetic motor imagery, mental calculation, word
association), 47 channels (43 EEG, 2 EOG, 2 EMG) at 500 Hz. Each evaluation participant's early
sessions come labeled for calibration; the test is their later sessions. The metric is balanced
accuracy averaged over subject × session × context cells. This 2026 Graz + BrainHero data is not
released yet; warm-up uses Dreyer 2023 (2-class motor imagery) as a proxy.

## 2. Register (each person, before 24 Oct)

1. Create a [Codabench account](https://www.codabench.org/accounts/signup).
2. On each track you enter, open **My Submissions**, accept the terms and click **Register**
   ([Track 1](https://www.codabench.org/competitions/17974/#/participate-tab),
   [2](https://www.codabench.org/competitions/17982/#/participate-tab),
   [3](https://www.codabench.org/competitions/17983/#/participate-tab),
   [4](https://www.codabench.org/competitions/17984/#/participate-tab)).
3. Submit the [registration form](https://forms.gle/p3t2V25nuQtVXyj9A) once, with your Codabench
   email, ticking every track you registered for.
4. Send the team leader your Codabench username. The leader invites you to the team's Codabench
   organization; you receive an email "You have been invited to join <team>" and must click accept
   while logged in to your own account. Until you accept you are listed as **INVITED** and cannot
   submit for the team. No email? Check that organization invite emails are allowed in your
   Codabench notification settings.
5. **Submit as the team, every time.** The upload form under **My Submissions** has a dropdown to
   submit as yourself or as an organization; pick the team, or the entry counts as individual. The
   dropdown appears only after you accept the invite.

Each person can belong to one team only; submission quotas stay individual, and submissions count
for the team. GitHub access to this repository does not make you a team member on Codabench.

**Team leader, once.** The leader handles the reproducibility audit, the workshop presentation and
the prize if we win, so pick someone available through December.

1. Complete steps 1–3 above.
2. Create the organization at
   [codabench.org/profiles/organization/create](https://www.codabench.org/profiles/organization/create/):
   **Organization Name** (shown on the leaderboard) and **Organization Email** are what matter;
   the rest is optional. Save.
3. On the organization's page, click **Edit**, then **Invite Users**. The invite box searches
   existing Codabench accounts by username, so teammates need an account first.
4. Check the member list: everyone should move from INVITED to MEMBER once they accept.

## 3. Set up your machine

You need git, Python ≥ 3.12, [uv](https://docs.astral.sh/uv/) and, for Track 2 alone, about
11 GB of disk (Dreyer raw data, the kit's cache and our window cache; all prepared tracks
together: 30+ GB). A GPU is optional: everything in the repo so
far was developed on CPU.

```bash
git clone https://github.com/promitmoitra/ephys_fm.git
cd ephys_fm
git clone https://github.com/neural-interfaces26/2026-competition external/2026-competition
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
uv pip install -r external/2026-competition/requirements.txt
```

Then follow the README's [Setup](../README.md#setup) and [Usage](../README.md#usage) sections:
run benchopt from the kit root with `BENCHOPT_DATA_HOME` pointing at our `data/`, check the
zero-download smoke test, and prepare the data you need (Dreyer: about 2 h to download,
8 min to extract; see the README's *Data* section for the deduplication notes).

**Not in git, so not in your clone:** `data/`, `outputs/`, `external/`, `.venv/`,
`submissions/`, and every trained weight (`*.pt`, `*.zip`). Regenerate what you need locally:

```bash
python track2/train_mixture.py --seed 0   # the full Track 2 package; ~55 min on 8 CPUs
```

Its first run also builds the Dreyer window cache `data/experiments/dreyer_windows.npz` (about
1 GB) from the prepared data. Commands in the docs that point into `outputs/` (for example
`--reuse-eegnet outputs/track2_dreyer_sim/submission/mixture.pt`) assume you have run this first.

## 4. How we work

The full protocol is [`CLAUDE.md`](../CLAUDE.md); it applies to people and Claude Code agents
alike. In short:

- **`main` is protected.** Every change reaches it through a pull request.
- **One worktree per line of work.** Your main checkout stays on `main`; each line of work lives
  in `.claude/worktrees/<name>` on its own branch, with symlinks to the main checkout's shared
  folders (commands in `CLAUDE.md` rule 6). This keeps parallel agents from switching branches
  under each other and shares the 30+ GB of data instead of copying it.
- **Branch names:** `exp/<topic>`, `track<N>/<topic>`, `research/<topic>`, `setup/<topic>`,
  `docs/<topic>`.
- **Git hygiene:** run `git branch --show-current` before every commit; never use bare
  `git stash` (the stash is shared by all worktrees; make a WIP commit instead).
- **Experiments:** commit the protocol (question, setting, metric, prediction) before the
  results. Record seed **and** thread count with every number: training is deterministic only for
  a fixed thread count.
- **Held-out data:** decisions are made on calibration data. In the Dreyer simulation, the hidden
  runs R4–R6 are scored only at pre-registered checkpoints, never iterated on.
- **Never upload a build trained on the Dreyer simulation to the warm-up leaderboard.** It trains
  on the warm-up test subjects' labeled runs, so its warm-up score is leaky.

## 5. Claude Code (optional)

- `CLAUDE.md` loads automatically and keeps agents inside their worktree's protocol.
- The project skill `.claude/skills/mne-python-guide` ships with the repo.
- The Track 2 research loops used the `autoresearch` plugin, which is installed separately and is
  not part of the repo.
- Per-machine settings go in `.claude/settings.local.json` (git-ignored).
- Jobs longer than ~30 min: start them with `setsid nohup python … > log 2>&1 < /dev/null &`, since
  harness background tasks are killed after about 30 min.

## 6. Where things stand

**Track 2.** The shipped design (`track2/README.md`, "Current best model"):

1. a pooled EEGNet, trained on everyone's labeled data;
2. per participant, a fine-tuned copy of it and a Riemannian (covariance) classifier, combined in
   log space and weighted by how reliably the Riemannian classifier decodes that person;
3. a filter-bank covariance "fingerprint" that recognises the participant from the window;
4. `predict` weights each participant's prediction by the fingerprint's probability for them
   (`predict` gets no participant IDs).

On the Dreyer simulation (calibration runs R1–R3, hidden runs R4–R6) the package scores 0.921
balanced accuracy (seed 0) and passes the kit's contract check. Read the caveats before trusting
that number for the sealed phase:

- R1–R3 and R4–R6 come from one session, while the sealed test uses later sessions.
- Most of the Dreyer score comes from a cue-evoked response, not motor imagery
  (`experiments/dreyer_eog/`).
- The full pipeline has one seed; only the expert-combination gain has three.

**Open work:**

- Track 2: retrain and re-validate on the 2026 Graz + BrainHero data once it is released (3
  classes, 47 channels, 500 Hz, cross-session).
- Tracks 1, 3 and 4: no models yet.
- The data atlas research track lives on `research/data-atlas` and is not on GitHub yet.

## 7. Reading order

1. [`README.md`](../README.md): tracks, setup, data, and the experiments table.
2. [`track2/README.md`](../track2/README.md): the current Track 2 model, its evidence, the
   simulation and the contract check.
3. `experiments/*/`: the studies behind the design (fingerprinting on BNCI2014_001, the Dreyer
   confound and EOG/EMG analyses).
4. [`research/findings.md`](../research/findings.md) and
   [`research/expert-weights/findings.md`](../research/expert-weights/findings.md): what the two
   research loops found.
5. [`docs/handoff/track2-research-loops.md`](handoff/track2-research-loops.md): how those loops
   were run, including the data traps and the ideas already ruled out.
