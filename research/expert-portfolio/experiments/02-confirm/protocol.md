# Protocol 02: honest confirmation (sim2) and the R4–R6 report

Locked 2026-10-03, before any sim2 model is trained or any R4–R6 number for the new stream is
computed.

- **Streams shipped by Task 7:** ShallowFBCSPNet (S). REVE is not shipped.
- **Combiner coefficients:** `outputs/t2-portfolio/portfolio_coef.json` (fitted on the dev bank;
  w = [0.496, 0.374], c = [0.791, 1.297], b = −0.051), transferred unchanged. E+T uses C3 as
  shipped.
- **sim2:**
  - The 14 people in `00-protocol/sim2_people.json`; seeds 0, 1, 2.
  - The full pipeline is trained from scratch:
    `train_mixture.py --eval-people … --shallow-experts --combiner portfolio`.
  - Reported, oracle and soft-routed: EEGNet experts, E+T (C3), full combination (E+T+S).
  - **These are the honest Dreyer estimates.**
- **R4–R6:** seeds 0–2 (loop B's test banks plus `bank_test_shallow_seed{0,1,2}`); E+T vs E+T+S,
  oracle. **Upper bounds** (these runs already gated loops A and B).
- **Veto (the only decision here):** withdraw ShallowFBCSPNet if E+T+S is more than 0.02 below
  E+T on sim2 (mean of seeds 0–2) or on R4–R6 (mean of seeds 0–2). Everything else is reported.
