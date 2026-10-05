# Protocol 01 analysis: the ship decision (CONFIRMATORY; amendment 1 applied)

Results: `results/dev.json`, `results/bnci.json`. λ_person for REVE = 1.0 (protocol rule:
0.5575 / 0.5639 / 0.5810 for 0.01 / 0.1 / 1.0).

## Streams alone (dev, true ID)

| Stream | Bal acc | NLL | Early-only | Late-only |
|---|---|---|---|---|
| E: EEGNet experts | 0.875 | 0.303 | 0.838 | 0.712 |
| S: ShallowFBCSPNet experts | 0.869 | 0.311 | 0.749 | 0.626 |
| R: REVE probe (mean-pooled) | 0.581 | 0.754 | 0.539 | 0.563 |
| T: Riemannian ts_C0.1 | 0.668 | 0.597 | — | — |

## Combinations (leave-one-person-out)

| Combination | Bal acc | NLL |
|---|---|---|
| E+T (shipped) | 0.883 | 0.278 |
| E+T+R | 0.885 | 0.277 |
| E+T+S | **0.894** | **0.253** |
| E+T+R+S | 0.893 | 0.253 |

## Ship rule

| Stream | (a) drop-one cost from E+T+R+S | (b) BNCI Δ vs E+T, mean of 5 seeds (per seed) | Decision |
|---|---|---|---|
| REVE | acc −0.0016, CI [−0.0048, +0.0012]; NLL +0.0004, CI [−0.0008, +0.0015] → **fails** | −0.0032 (−0.004, −0.004, −0.008, +0.001, −0.001) | **not shipped** |
| ShallowFBCSPNet | acc **+0.0083, CI [+0.0004, +0.0167]**; NLL **+0.0247, CI [+0.0114, +0.0380]** → passes (both routes) | **−0.0042** (−0.009, −0.003, −0.002, −0.002, −0.006) → passes (≥ −0.005) | **ships** |

Coefficients for E+T+S, fitted on the full dev bank (`outputs/t2-portfolio/portfolio_coef.json`,
unrounded): w = [0.496 EEGNet, 0.374 ShallowFBCSPNet], c = [0.791, 1.297], b = −0.051.

## Reading

- **ShallowFBCSPNet** is about as good as EEGNet alone (0.869 vs 0.875) but makes different
  errors, so the two-CNN ensemble gains on dev (+0.011 over E+T; NLL 0.278 → 0.253). It reads the
  early cue-locked window less than EEGNet (early-only 0.749 vs 0.838), consistent with its
  band-power design.
- **Caveat, stated plainly:** on BNCI (cross-day), adding it is slightly negative on **all five
  seeds** (mean −0.004). That passes the locked "does not hurt" threshold of −0.005, but only
  narrowly, and BNCI is the regime the sealed phase resembles. With Dreyer-fitted weights, the
  extra CNN neither helps nor clearly hurts across days. sim2 (Task 9) checks Dreyer honestly; it
  cannot settle the cross-day question.
- **REVE, mean-pooled, earns nothing.** 0.581 alone is far below the kit's REVE-probe reference
  (Stieger: 68.0 vs EEGNet 58.6). Likely cause: averaging the embedding over channels discards the
  electrode layout, which carries left-vs-right imagery. A channel-preserving probe (mean over
  time patches only, 27 × 512 features), or REVE's pretrained attention pooling, is a separate,
  untested hypothesis. It is **not** evaluated here, so this result says nothing about REVE with
  a spatial probe.
