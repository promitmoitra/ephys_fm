# Dreyer EOG: cue-triggered change vs prior gaze

> **Inconclusive:** the pre-cue control is not at chance. The pipeline's zero-phase 0.1 Hz high-pass smears the post-cue eye deflection into the baseline (opposite sign, growing toward the cue). Needs a causal high-pass re-extraction.

Baseline -2.5 to -1.0 s (fixation, before the beep). Logistic regression on 250 ms EOG bin means; train subjects → val (A/C), test (B). Chance 0.5.

| Features | Val | Test |
|---|---|---|
| precue_control | 0.565 | 0.408 |
| cue_0-1.25s/raw | 0.696 | 0.267 |
| cue_0-1.25s/change | 0.696 | 0.267 |
| full_0-4s/raw | 0.701 | 0.256 |
| full_0-4s/change | 0.686 | 0.263 |

Right-minus-left **change from baseline** (0–1.25 s), subjects positive / n (median):

| Part | EOG1 | EOG2 | EOG3 |
|---|---|---|---|
| A | 36/60 (+0.059) | 58/60 (+0.373) | 49/60 (+0.133) |
| B | 10/21 (-0.015) | 0/21 (-0.410) | 4/21 (-0.161) |
| C | 1/6 (-0.151) | 1/6 (-0.341) | 3/6 (+0.035) |
