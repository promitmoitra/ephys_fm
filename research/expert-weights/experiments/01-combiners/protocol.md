# Protocol 01: first combiner sweep on the dev bank (H1–H7)

Locked 2026-09-30, before the dev bank exists. Metric and CV as in protocol 00
(true ID; LOPO for global combiners, within-person stratified 2-fold × 5 for per-person).

## Combiners (all on per-person experts; EEGNet = `eegnet`, Riemannian = `ts`, CSP = `csp`)

| ID | Combiner | Experts | Scope |
|---|---|---|---|
| R0 | EEGNet only | eegnet | — |
| R1 | Equal linear average (current "+ts" rule) | eegnet, ts | — |
| R2 | Equal linear average | eegnet, ts, csp | — |
| H1 | Linear pool, learned weights (softmax), NLL | eegnet, ts | global |
| H2 | same as H1 | eegnet, ts | per person |
| H3 | Log-linear with class bias (stacking on log-probs) | eegnet, ts | global |
| H4 | Log-linear, no bias | eegnet, ts | global |
| H5a | Per-expert temperature, then equal average | eegnet, ts | global |
| H5b | Per-expert temperature, then learned linear pool | eegnet, ts | global |
| H6 | Confidence gate (tempered, weight ∝ softmax(a + g·max-prob)) | eegnet, ts | global |
| H7 | Linear pool, learned | eegnet, ts, csp | global |
| H7b | Log-linear, no bias | eegnet, ts, csp | global |

Also reported, for context only: pooled, control, ts only, csp only; and every combiner
soft-routed with `fp_biased` (exploratory, optimistic fingerprint).

## Predictions (mechanism → prediction)

- The fine-tuned EEGNet expert (80 windows, 50 epochs) is over-confident, so H5 finds
  T_eegnet > 1 and the equal average of *tempered* experts gains over R1 mostly in NLL.
- ts carries some complementary imagery information (8–30 Hz, 0.5–4 s vs EEGNet's early
  cue-locked response), so the learned EEGNet weight in H1 is < 1 but ≥ 0.5, and H1 ≥ R1 in
  accuracy by a small margin (≤ 0.01).
- Per-person weights (H2) lose to global ones (forecast-combination puzzle: 20 windows per
  fold).
- CSP gets a small weight (H7 ≈ H1); R2 < R1, as on R4–R6.

## Decision rule for checkpoint 1 (confirmation on R4–R6)

Pre-register at most three candidates, chosen on the dev bank only: R1 (reference), the
global combiner with the best dev balanced accuracy, and the one with the best dev NLL
(if different; ties broken toward fewer parameters). A candidate "wins" on the dev bank only
if it beats R1 by ≥ 0.005 balanced accuracy *or* by NLL without losing accuracy, with more
people better than worse. If nothing wins, the finding is "the equal average is not
improvable with 840 windows" and the loop moves to richer dev data (cross-fitted banks) or
to better experts.
