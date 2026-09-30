# Research Findings (loop A: fingerprint)

## Research Question

Which per-window person-ID model gives the most accurate and best-calibrated p(person | window)
for the 21 Dreyer evaluation people, so that soft routing in the Track 2 mixture keeps more of
the oracle-ID gain (0.901 soft vs 0.908 oracle on R4–R6, seed 0)?

## Current Understanding

Nothing measured yet under the locked protocol. Prior evidence (before this loop):
- The shipped fingerprint (EEGNet, epoch picked on R3) scored 0.857 on R3 (biased) and 0.744
  on R4–R6: accuracy falls with distance in time from the training runs.
- On BNCI, soft routing kept 62–88% of the oracle gain, lowest for the weakest fingerprint.
- Dreyer is one session per person: the problem is cross-run within a session.

## Key Results

**Riemannian fingerprints are near-perfect within a Dreyer session** (H3, H3.1). Tangent space +
multinomial LR, deterministic:

| Benchmark (dev only) | ts_broad_C1 | ts_fb_C1 (6 bands) |
|---|---|---|
| R1–R2 → R3 (locked inner loop) | 0.987 / NLL 0.107 | **1.000 / NLL 0.027** |
| R1 → R2 (one training run) | 0.967 / 0.153 | 0.996 / 0.065 |
| R1 → R3 (two runs later) | 0.976 / 0.188 | 1.000 / 0.052 |
| BNCI session 1 → 2 (cross-day, 9-way) | 0.872 / 0.403 | (running) |

No decay with run distance inside a session: R1 → R3 is as good as R1 → R2. The shipped
EEGNet fingerprint scored 0.857 on R3 with epoch selection *on R3* (biased) and 0.744 on R4–R6.

## Patterns and Insights

## Lessons and Constraints

- R4–R6 are never used for decisions; R3 is the dev set; fixed epoch budgets, last epoch.
- 4 CPU threads (loop B shares the machine); training is deterministic given seed, epochs and
  thread count.

## Open Questions

- How much of the R3 → R4–R6 drop is temporal drift that R3 cannot reveal? (R1 → R3 vs
  R1–R2 → R3 gives a drift-distance proxy.)

## Optimization Trajectory

(none yet)
