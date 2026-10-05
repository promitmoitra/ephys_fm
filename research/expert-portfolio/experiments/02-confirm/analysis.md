# Protocol 02 analysis: honest confirmation (sim2) and R4–R6 (upper bounds)

Sources: `track2/results/dreyer_sim_seed{0,1,2}_fb_riemann_c3_sim2_shallow_portfolio.{md,json}`
(sim2) and `results_r4r6.json` (R4–R6). Balanced accuracy, true person ID unless marked soft.

## sim2: 14 fresh people, full pipeline, seeds 0–2 (**honest**)

| Seed | Fingerprint | EEGNet experts | E+T (C3) | E+T+S | `predict` (soft E+T+S) |
|---|---|---|---|---|---|
| 0 | 1.000 | 0.9375 | 0.9369 | 0.9310 | 0.9315 |
| 1 | 1.000 | 0.9375 | 0.9381 | 0.9298 | 0.9298 |
| 2 | 1.000 | 0.9315 | 0.9345 | 0.9351 | 0.9351 |
| **mean** | 1.000 | **0.9355** | **0.9365** | **0.9319** | 0.9321 |

Per-person differences (mean over seeds, person bootstrap, 14 people):
- E+T+S − E+T: **−0.0046 [−0.016, +0.004]**, 4 better / 5 worse.
- E+T − EEGNet: **+0.0010 [−0.004, +0.007]**, 5 / 5.
- E+T+S − EEGNet: −0.0036 [−0.017, +0.008], 4 / 8.

## R4–R6, seeds 0–2 (**upper bounds**: these runs gated loops A and B)

| Seed | E+T | E+T+S | NLL E+T → E+T+S |
|---|---|---|---|
| 0 | 0.9214 | 0.9294 | 0.209 → 0.190 |
| 1 | 0.9127 | 0.9286 | 0.222 → 0.198 |
| 2 | 0.9183 | 0.9369 | 0.215 → 0.189 |

Mean E+T+S − E+T = +0.014.

## Veto (the only decision in this protocol)

E+T+S − E+T: sim2 mean −0.0046 and R4–R6 mean +0.014; neither is below −0.02. **No veto**, so
under the locked rules ShallowFBCSPNet remains a ship candidate.

## What the evidence says (beyond the locked rule)

| Evidence | E+T+S − E+T | E+T − EEGNet alone |
|---|---|---|
| Dev bank (decision data) | +0.011 [+0.000, +0.017] | +0.008 |
| R4–R6, 3 seeds (upper bound) | +0.014 (3/3) | +0.017 (3/3, loop B) |
| **sim2, 3 seeds (honest)** | **−0.005 [−0.016, +0.004]** | **+0.001 [−0.004, +0.007]** |
| **BNCI, 5 seeds (honest, cross-day)** | **−0.004 (5/5 negative)** | +0.052 (loop B, 5/5) |

- **ShallowFBCSPNet:** both honest checks are slightly negative and neither excludes zero; only
  the decision data and the reused hidden runs are positive. The fair summary is "no detectable
  benefit on fresh data". The locked rule lets it through because the veto only catches drops
  beyond 0.02.
- **The shipped C3 (loop B), as a by-product:** on fresh Dreyer people it adds only +0.001 over
  EEGNet alone, against +0.017 on the reused R4–R6. Its strong cross-day gain on BNCI (+0.05)
  still stands; its within-session Dreyer gain does not replicate. The R4–R6 numbers reported for
  loops A/B/C should be read as optimistic.
- sim2 scores are higher overall (0.93) than the original simulation's (0.90): different, easier
  people. Only within-sim2 differences are comparable.
