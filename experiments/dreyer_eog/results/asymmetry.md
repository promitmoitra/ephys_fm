# Dreyer 2023: hemispheric-asymmetry and baseline-relative features

`asym` = log P(left) − log P(right) for 11 homologous pairs × 4 bands (θ, μ, β, γ); `bandpower` = per-channel log power (27 × 4); `_rel` = minus the same trial's pre-cue baseline (-2.5 to -1.0 s). Chance 0.5.

## `eeg_reg`: cross-subject (train subjects → val A/C, test B)

| Window | Features | Val | Test |
|---|---|---|---|
| cue_0-1.25s | bandpower | 0.562 | 0.555 |
| cue_0-1.25s | asym | 0.556 | 0.552 |
| cue_0-1.25s | bandpower_rel | 0.555 | 0.531 |
| cue_0-1.25s | asym_rel | 0.543 | 0.546 |
| cue_0-1.25s | asym_rel+bandpower_rel | 0.554 | 0.531 |
| sustained_1.25-4s | bandpower | 0.558 | 0.567 |
| sustained_1.25-4s | asym | 0.561 | 0.574 |
| sustained_1.25-4s | bandpower_rel | 0.531 | 0.544 |
| sustained_1.25-4s | asym_rel | 0.534 | 0.557 |
| sustained_1.25-4s | asym_rel+bandpower_rel | 0.531 | 0.542 |
| precue_control | bandpower | 0.491 | 0.504 |
| precue_control | asym | 0.503 | 0.503 |

Time-resolved peak (val): bandpower 0.576 @ 0.25 s (test 0.573); asym 0.561 @ 0.25 s (test 0.565); bandpower_rel 0.568 @ 0.25 s (test 0.558); asym_rel 0.564 @ 0.25 s (test 0.566)

## `eeg_reg`: within-subject (train R1–R3 → test R4–R6), mean ± SEM

| Window / features | All (87) | A (60) | B (21) | C (6) |
|---|---|---|---|---|
| cue_0-1.25s/bandpower | 0.571 ± 0.007 | 0.569 ± 0.009 | 0.570 ± 0.016 | 0.597 ± 0.031 |
| cue_0-1.25s/asym | 0.576 ± 0.007 | 0.577 ± 0.009 | 0.571 ± 0.015 | 0.589 ± 0.020 |
| cue_0-1.25s/bandpower_rel | 0.561 ± 0.008 | 0.559 ± 0.010 | 0.574 ± 0.016 | 0.542 ± 0.011 |
| cue_0-1.25s/asym_rel | 0.556 ± 0.008 | 0.556 ± 0.010 | 0.561 ± 0.015 | 0.536 ± 0.009 |
| cue_0-1.25s/asym_rel+bandpower_rel | 0.564 ± 0.008 | 0.563 ± 0.010 | 0.573 ± 0.016 | 0.538 ± 0.019 |
| cue_0-1.25s/csp_lda | 0.582 ± 0.009 | 0.586 ± 0.010 | 0.578 ± 0.018 | 0.564 ± 0.032 |
| sustained_1.25-4s/bandpower | 0.590 ± 0.011 | 0.591 ± 0.013 | 0.578 ± 0.023 | 0.625 ± 0.035 |
| sustained_1.25-4s/asym | 0.594 ± 0.011 | 0.592 ± 0.014 | 0.586 ± 0.023 | 0.636 ± 0.029 |
| sustained_1.25-4s/bandpower_rel | 0.558 ± 0.008 | 0.555 ± 0.011 | 0.563 ± 0.015 | 0.565 ± 0.038 |
| sustained_1.25-4s/asym_rel | 0.559 ± 0.009 | 0.556 ± 0.011 | 0.563 ± 0.015 | 0.569 ± 0.039 |
| sustained_1.25-4s/asym_rel+bandpower_rel | 0.563 ± 0.009 | 0.561 ± 0.010 | 0.569 ± 0.016 | 0.560 ± 0.044 |
| sustained_1.25-4s/csp_lda | 0.628 ± 0.016 | 0.613 ± 0.017 | 0.646 ± 0.039 | 0.713 ± 0.063 |

| Paired comparison | Mean diff | Better / worse (of 87) | Wilcoxon p |
|---|---|---|---|
| cue_0-1.25s: asym vs bandpower | +0.005 | 47 / 37 | 0.25 |
| cue_0-1.25s: bandpower_rel vs bandpower | -0.010 | 33 / 52 | 0.16 |
| cue_0-1.25s: asym_rel vs asym | -0.020 | 26 / 58 | 0.0051 |
| cue_0-1.25s: asym_rel+bandpower_rel vs csp_lda | -0.019 | 36 / 46 | 0.038 |
| cue_0-1.25s: asym_rel vs csp_lda | -0.027 | 29 / 52 | 0.0074 |
| sustained_1.25-4s: asym vs bandpower | +0.004 | 43 / 39 | 0.78 |
| sustained_1.25-4s: bandpower_rel vs bandpower | -0.032 | 26 / 57 | 8.1e-05 |
| sustained_1.25-4s: asym_rel vs asym | -0.035 | 24 / 58 | 1.1e-05 |
| sustained_1.25-4s: asym_rel+bandpower_rel vs csp_lda | -0.065 | 27 / 58 | 3.1e-05 |
| sustained_1.25-4s: asym_rel vs csp_lda | -0.069 | 23 / 60 | 3.9e-06 |

## `eeg`: cross-subject (train subjects → val A/C, test B)

| Window | Features | Val | Test |
|---|---|---|---|
| cue_0-1.25s | bandpower | 0.570 | 0.554 |
| cue_0-1.25s | asym | 0.567 | 0.559 |
| cue_0-1.25s | bandpower_rel | 0.540 | 0.537 |
| cue_0-1.25s | asym_rel | 0.544 | 0.553 |
| cue_0-1.25s | asym_rel+bandpower_rel | 0.541 | 0.538 |
| sustained_1.25-4s | bandpower | 0.568 | 0.570 |
| sustained_1.25-4s | asym | 0.564 | 0.576 |
| sustained_1.25-4s | bandpower_rel | 0.540 | 0.559 |
| sustained_1.25-4s | asym_rel | 0.546 | 0.558 |
| sustained_1.25-4s | asym_rel+bandpower_rel | 0.539 | 0.559 |
| precue_control | bandpower | 0.496 | 0.504 |
| precue_control | asym | 0.493 | 0.499 |

Time-resolved peak (val): bandpower 0.605 @ 0.25 s (test 0.575); asym 0.612 @ 0.25 s (test 0.585); bandpower_rel 0.592 @ 0.25 s (test 0.564); asym_rel 0.603 @ 0.25 s (test 0.583)

## `eeg`: within-subject (train R1–R3 → test R4–R6), mean ± SEM

| Window / features | All (87) | A (60) | B (21) | C (6) |
|---|---|---|---|---|
| cue_0-1.25s/bandpower | 0.576 ± 0.006 | 0.574 ± 0.007 | 0.585 ± 0.013 | 0.567 ± 0.012 |
| cue_0-1.25s/asym | 0.576 ± 0.007 | 0.579 ± 0.009 | 0.569 ± 0.014 | 0.575 ± 0.020 |
| cue_0-1.25s/bandpower_rel | 0.559 ± 0.006 | 0.551 ± 0.008 | 0.579 ± 0.009 | 0.572 ± 0.016 |
| cue_0-1.25s/asym_rel | 0.552 ± 0.006 | 0.550 ± 0.008 | 0.564 ± 0.012 | 0.542 ± 0.015 |
| cue_0-1.25s/asym_rel+bandpower_rel | 0.565 ± 0.007 | 0.561 ± 0.009 | 0.581 ± 0.008 | 0.553 ± 0.022 |
| cue_0-1.25s/csp_lda | 0.601 ± 0.010 | 0.609 ± 0.012 | 0.602 ± 0.024 | 0.519 ± 0.025 |
| sustained_1.25-4s/bandpower | 0.596 ± 0.011 | 0.600 ± 0.014 | 0.583 ± 0.023 | 0.597 ± 0.020 |
| sustained_1.25-4s/asym | 0.606 ± 0.012 | 0.602 ± 0.014 | 0.606 ± 0.025 | 0.654 ± 0.034 |
| sustained_1.25-4s/bandpower_rel | 0.566 ± 0.008 | 0.561 ± 0.010 | 0.578 ± 0.016 | 0.569 ± 0.022 |
| sustained_1.25-4s/asym_rel | 0.573 ± 0.009 | 0.564 ± 0.012 | 0.590 ± 0.020 | 0.610 ± 0.027 |
| sustained_1.25-4s/asym_rel+bandpower_rel | 0.572 ± 0.009 | 0.565 ± 0.011 | 0.587 ± 0.017 | 0.587 ± 0.023 |
| sustained_1.25-4s/csp_lda | 0.659 ± 0.015 | 0.656 ± 0.017 | 0.661 ± 0.040 | 0.689 ± 0.049 |

| Paired comparison | Mean diff | Better / worse (of 87) | Wilcoxon p |
|---|---|---|---|
| cue_0-1.25s: asym vs bandpower | +0.000 | 42 / 37 | 0.88 |
| cue_0-1.25s: bandpower_rel vs bandpower | -0.016 | 35 / 49 | 0.016 |
| cue_0-1.25s: asym_rel vs asym | -0.024 | 28 / 56 | 0.00025 |
| cue_0-1.25s: asym_rel+bandpower_rel vs csp_lda | -0.036 | 28 / 55 | 0.0019 |
| cue_0-1.25s: asym_rel vs csp_lda | -0.049 | 30 / 56 | 3.5e-05 |
| sustained_1.25-4s: asym vs bandpower | +0.011 | 46 / 38 | 0.1 |
| sustained_1.25-4s: bandpower_rel vs bandpower | -0.030 | 27 / 56 | 0.00011 |
| sustained_1.25-4s: asym_rel vs asym | -0.033 | 29 / 56 | 0.00013 |
| sustained_1.25-4s: asym_rel+bandpower_rel vs csp_lda | -0.088 | 22 / 64 | 5.2e-09 |
| sustained_1.25-4s: asym_rel vs csp_lda | -0.086 | 18 / 67 | 1.8e-09 |

