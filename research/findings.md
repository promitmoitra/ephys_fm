# Findings: loop B (learned expert weights)

## Current understanding

Starting point (seed 0, R4–R6): EEGNet experts soft-routed 0.901; equal average with the
per-person Riemannian (ts) expert 0.909 (11 people better / 9 worse); adding CSP to the
average hurt. The ts expert alone is far weaker (0.688 oracle vs 0.908), so an equal
average is almost certainly not the best weighting, and its gain is within noise.

## Patterns and insights

(none yet)

## Lessons and constraints

- R4–R6 are never used for decisions; confirm only pre-registered candidates.
- Dev experts are trained on 2 calibration runs, test experts on 3.

## Open questions

- Is the ts expert's contribution complementary (errors on different windows) or just
  regularisation of an over-confident EEGNet expert? Temperature scaling (H5) separates these.
