# Dreyer 2023: EOG, EMG and EOG-regressed EEG decoding

Left vs right hand motor imagery; EEGNet trained on the kit's train subjects, best of 30 epochs on val subjects, tested on unseen test subjects 61–81. Chance 0.5. Window from the cue: arrow 0–1.25 s (`cue`), full 0–4 s (`full`). Seed 2.

| Signal / window | Channels × samples | Val bal. acc | Test bal. acc | Acquisition R1–R2 | Online R3–R6 |
|---|---|---|---|---|---|
| eeg_full | 27 × 480 | 0.872 | 0.801 | 0.804 | 0.800 |
| eeg_cue | 27 × 150 | 0.865 | 0.782 | 0.793 | 0.777 |
| eog_full | 3 × 480 | 0.713 | 0.247 | 0.262 | 0.239 |
| eog_cue | 3 × 150 | 0.714 | 0.266 | 0.277 | 0.261 |
| heog_cue | 1 × 150 | 0.622 | 0.378 | 0.374 | 0.379 |
| emg_full | 2 × 480 | 0.596 | 0.637 | 0.595 | 0.658 |
| emg_cue | 2 × 150 | 0.539 | 0.594 | 0.564 | 0.609 |
| eeg_reg_full | 27 × 480 | 0.837 | 0.662 | 0.663 | 0.661 |
| eeg_reg_cue | 27 × 150 | 0.830 | 0.650 | 0.660 | 0.645 |

Kit check (our EEG windows vs the kit's, correlation): s1 r0: 1.0000, s1 r5: 1.0000, s61 r0: 1.0000, s61 r5: 1.0000, s85 r0: 1.0000, s85 r5: 1.0000
