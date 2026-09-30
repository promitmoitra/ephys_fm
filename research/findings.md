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

(none yet)

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
