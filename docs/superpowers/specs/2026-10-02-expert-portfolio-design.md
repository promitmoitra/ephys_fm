# Track 2 loop C: an expert portfolio (REVE probe, ShallowFBCSPNet) behind a learned combiner

Date: 2026-10-02 · Branch `exp/expert-portfolio` · Worktree `.claude/worktrees/t2-portfolio`
Background: `docs/alt-arch.md` (untracked, in the main checkout). Its statement that the loop A
and loop B parts are "not yet in `submission.py`" is out of date: PR #9 shipped them.

## Goal

1. **Primary: a hedge for the unknown sealed data.** Add expert streams with different inductive
   biases next to the shipped ones, and let a combiner fitted on calibration data weight them per
   dataset. A stream ships only if it earns weight on dev and does not hurt across days.
2. **Secondary: a comparison page.** A benchopt results page that shows several models side by
   side, trained without leakage, for reporting and understanding.

**Not the goal:** picking a single "best architecture" on Dreyer. Dreyer rewards an early
cue-locked response that may not exist in the sealed task.

## Current state (main @ cfd7402)

`submission.py` routes per-person experts by a filter-bank covariance fingerprint (0.996 on Dreyer
R4–R6). Per person it combines two streams with the reliability-weighted log-linear combiner C3:
- the EEGNet expert (pooled EEGNet, whole-network fine-tune on the person's calibration runs);
- the Riemannian expert (ts_C0.1).

Dreyer simulation seed 0: `predict` 0.921 (EEGNet experts alone 0.907). All EEGNets are stock
braindecode defaults (2,018 parameters).

## Constraints

- **Inference imports:** `submission.py` imports only torch and braindecode. Weights ship as
  tensors; no sklearn, pyriemann or scipy at inference.
- **Budget:** ≤ 60 min on one A100 for all windows, streams and the fingerprint. Submission
  storage is 15 GB per profile, shared by four tracks.
- **Pretrained models:** offline at inference. `brain-bzh/reve-base` is pre-staged on the
  worker (`tools/staged_hf_models.txt`); nothing else is. Declare external data and compute
  (Rule 04).
- **Local compute:** 8 CPU threads, 16 GB RAM, no GPU, shared with other sessions (≤ 4 threads
  for this loop).
- **Evaluation discipline:** R4–R6 are report-only upper bounds (already reused by loops A and B);
  decisions use the dev bank and BNCI; honest Dreyer numbers come from sim2; protocols are committed
  before results.

## Components

Each stream maps a window to per-person class log-probabilities `(B, K, C)`, so routing and
combination stay architecture-agnostic.

### S1. REVE frozen probe (new)

- **Encoder:** `brain-bzh/reve-base` (≈ 69M parameters, 277 MB), frozen, eval mode.
- **Input:**
  - The kit's window (27 ch × 480 samples at 120 Hz) is resampled to 200 Hz (800 samples) by one
    fixed linear map: a (480, 800) matrix made at training time by applying the resampler to the
    identity. The same trick is used for the fingerprint filters, so no scipy is needed at
    inference.
  - Input scaling follows REVE's expected preprocessing, settled in probe P0.
- **Electrode positions:** looked up once at training time from `brain-bzh/reve-positions` for
  the training channel names, and shipped as a `(n_chans, 3)` tensor. At inference the
  encoder is constructed **without** downloading the position bank (that repo is not
  pre-staged), and the positions are passed explicitly.
- **Embedding:** the encoder's output tokens are mean-pooled into one 512-d vector per window
  (the pooling choice is confirmed in P0).
- **Heads:**
  - A pooled multinomial logistic head is trained on all labelled windows (training pool +
    calibration).
  - A per-person head starts from the pooled head and is fine-tuned on that person's calibration
    windows with a fixed budget, keeping the last epoch (the EEGNet recipe's analogue).
  - Each head is 512 × C weights.
- **Cost:** the embeddings for every window are computed once and cached (`outputs/t2-portfolio/`).
  Training heads takes seconds. At inference: one encoder pass per window, then K heads.

### S2. ShallowFBCSPNet experts (new)

- braindecode `ShallowFBCSPNet` as the pooled model, plus a whole-network fine-tune per person.
  The recipe is identical to the EEGNet experts (AdamW lr 1e-3 → 1e-4, 100 + 50 epochs, epoch on
  the val people, last epoch for fine-tunes).
- **Time constants:** its defaults assume 250 Hz. They are rescaled to the data's sampling rate
  (filter length, pool length and stride × sfreq / 250), set once in the protocol.
- **Purpose:** band power learned end to end (square → mean-pool → log), the neural counterpart
  of the Riemannian expert. It is expected to lean less on the evoked cue response than EEGNet.

### S3. N-stream combiner (generalises C3)

- **Form,** per person k:
  `z = Σ_s w_s · log p_s,k + (c_cls,0 + c_cls,1 · (rel_k − 0.5)) · log p_riemann,k + b`
  with one weight per neural stream, the reliability slope for the classical stream (as in C3),
  and a class bias.
- **Fitting:** by NLL with a small L2 penalty, on the cross-fitted dev bank (below).
- **Code:** extends loop B's `RelLogLinear` / `LogLinearCombiner`; the torch inference version
  goes in `submission.py`.

### Unchanged

The fingerprint, routing, EEGNet experts and the Riemannian experts.

## Evaluation protocol

### Banks

1. **Dreyer cross-fitted dev bank.** For each held-out calibration run r ∈ {R1, R2, R3}:
   - Each new stream's pooled part is trained on the training pool + the other two runs (epoch on
     val), and per-person parts on those two runs; then every person's stream predicts run r.
   - The EEGNet and Riemannian folds already exist (`outputs/t2-expert-weights/`). Only S1 and S2
     are new: S1 costs head training only; S2 needs 3 pooled trainings.
   - 2,520 windows.
2. **Dreyer test bank.** Each stream trained on R1–R3, predicting R4–R6. Report-only (upper bound),
   only.
3. **BNCI 2014-001 cross-day bank.** Session 1 runs 0–4 train, run 5 val, session 2 test; 5
   seeds for trained parts (S1's frozen encoder is seed-free).

### Metrics (true ID unless stated)

- Each stream alone: balanced accuracy and NLL.
- The combination: all streams, and drop-one ablations (leave-one-person-out on the dev bank).
- Per-person paired differences with person-bootstrap 95% CIs.
- **Shortcut diagnostic:** each stream scored on the early window only (0–1.25 s, rest zeroed)
  and the late window only (1.25–4 s, early zeroed). Pooled and fine-tuned parts are trained on
  full windows; the diagnostic only measures. A stream whose accuracy is mostly early is reading
  the cue response.

### Decision rule (fixed in this spec; repeated in the loop's first protocol)

A new stream is a **ship candidate** if both hold:
- **(a) Dev bank:** dropping it from the full combination costs ≥ 0.005 balanced accuracy, or
  ≥ 0.005 NLL with the bootstrap CI of the NLL difference excluding zero.
- **(b) BNCI:** adding it to the shipped two-stream combination does not lower balanced accuracy
  (mean over 5 seeds ≥ −0.005) with Dreyer-fitted weights.

**The ship decision is made on (a) and (b) alone.** Dreyer's R4–R6 have already gated decisions
in loops A and B, so their scores are now optimistic upper bounds and no longer choose anything.

### Honest estimates (revised 2026-10-02)

- **Fresh simulation, `sim2`:**
  - 14 people drawn from the 52-person training pool, with a fixed seed **before** any training,
    and never used for any choice.
  - Their R1–R3 are calibration; their R4–R6 are hidden.
  - The training pool is the other 38 training-pool people plus the 21 original evaluation people
    (all six runs, as ordinary training people). The pooled model's epoch is still chosen on the
    kit's val people.
  - The whole pipeline runs on it (fingerprint, EEGNet experts, Riemannian experts, new streams;
    combiner coefficients transferred from the dev bank) for seeds 0–2.
  - This is the honest Dreyer estimate for the shipped design and for the candidates.
- **R4–R6:** the candidates are also scored there (seeds 0–2), **report-only, labelled as upper
  bounds**.
- **Veto (pre-registered, gross failure only):** a candidate is withdrawn if the full combination
  falls more than 0.02 below the two-stream combination on sim2 (mean over seeds) or on R4–R6.
  Such a drop would point to a bug or a broken transfer, not a close call.

## Deliverables

1. **Loop C workspace:** `research/expert-portfolio/` (state, log, findings, protocols,
   experiments, reports), so it cannot collide with other loops' `research/` files.
2. **Probe P0 (first task, a spike):** load `reve-base` offline with shipped positions (no
   position-bank download); confirm input scaling, pooling and output shapes; time the embedding
   of 1,000 Dreyer windows on 4 CPU threads. It also confirms the frozen encoder's ZIP footprint
   is zero (loaded from the staged cache).
   - **Exit:** a go / no-go for S1, with the estimated full-cache time.
3. **Stream code:**
   - `track2/` training-side helpers for S1 and S2 (like `riemann_parts.py`).
   - Torch inference modules and config flags in `submission.py`. Old packages keep loading
     unchanged.
4. **Integration:** shippable streams packaged by `train_mixture.py`, scored through `predict`,
   and passing the kit contract check offline (`HF_HUB_OFFLINE=1`).
5. **Comparison page (by-product):**
   - **Solvers:** each model is a standalone benchopt solver under `track2/bench/`. It loads
     weights from the kit's per-solver folder, `tracks/bci_decoding/outputs/<Solver.name>/`
     (the kit's default when `COMPET_SUBMISSION_DIR` is unset), so several solvers can run in
     one `benchopt run`.
   - **Leakage-free weights:** trained on the warm-up train split only, with no evaluation-people
     labels; safe to show.
   - **Line-up:** Constant, MeanLogReg, EEGNet (kit), ShallowFBCSPNet, the REVE probe, and the
     filter-bank Riemannian classifier (pooled).
   - The calibrated mixture appears only on the leaky simulation page, clearly labelled.
   - One benchopt run renders the HTML page.
6. **Docs:** `track2/README.md` current-best section, the experiments table, and progress
   reports in `research/expert-portfolio/to_human/`.

## Risks and how the plan handles them

| Risk | Handling |
|---|---|
| REVE needs the position bank online | P0 builds the encoder with a stub position bank and explicit positions; no-go stops S1 before any bank work |
| REVE expects other input scaling (e.g. µV, z-scored per channel) | P0 checks the documented preprocessing; the probe is retrained if the scaling changes |
| Embedding all windows on CPU is slow | P0 measures; Dreyer ~21k windows + BNCI ~5k are embedded once and cached; ≤ 4 threads |
| ShallowFBCSPNet pooled training time (3 dev folds + test + BNCI × 5 seeds) | Run in the background with `setsid nohup`; ~1 h per Dreyer pooled model expected at 4 threads |
| A stream wins on Dreyer by reading the cue response more | Shortcut diagnostic per stream; the BNCI condition (b) is required |
| 500 Hz sealed data | S1's resampling matrix and S2's rescaled time constants are derived from `meta["sfreq"]` at packaging time; the matrix size at 500 Hz is reported |

## Out of scope

- Fine-tuning REVE: no GPU, and storage would be needed per person.
- Other foundation models (LaBraM, CBraMod, EEGPT): not pre-staged. Revisit through a PR to the
  kit's staged list if S1 pays off.
- Changes to the fingerprint or to the EEGNet recipe.
- Sealed-data retraining: data not released.
