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

## sim2 people (drawn with seed 20261002; `sim2_people.json`)
3, 8, 10, 14, 19, 20, 40, 45, 46, 47, 50, 55, 58, 82. Hidden runs R4–R6 are complete (40
windows per run) and class-balanced for all 14. Person 40 has 32 windows in R3 (calibration),
so sim2 calibration has 1,672 windows (not 1,680); the draw is kept as drawn.
