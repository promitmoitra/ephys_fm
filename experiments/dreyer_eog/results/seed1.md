# Dreyer 2023: EOG, EMG and EOG-regressed EEG decoding

Left vs right hand motor imagery; EEGNet trained on the kit's train subjects, best of 30 epochs on val subjects, tested on unseen test subjects 61–81. Chance 0.5. Window from the cue: arrow 0–1.25 s (`cue`), full 0–4 s (`full`). Seed 1.

| Signal / window | Channels × samples | Val bal. acc | Test bal. acc | Acquisition R1–R2 | Online R3–R6 |
|---|---|---|---|---|---|
| eeg_full | 27 × 480 | 0.883 | 0.796 | 0.798 | 0.795 |
| eeg_cue | 27 × 150 | 0.872 | 0.776 | 0.780 | 0.774 |
| eog_full | 3 × 480 | 0.713 | 0.248 | 0.260 | 0.242 |
| eog_cue | 3 × 150 | 0.711 | 0.259 | 0.262 | 0.258 |
| heog_cue | 1 × 150 | 0.627 | 0.382 | 0.389 | 0.379 |
| emg_full | 2 × 480 | 0.597 | 0.629 | 0.596 | 0.646 |
| emg_cue | 2 × 150 | 0.535 | 0.587 | 0.565 | 0.598 |
| eeg_reg_full | 27 × 480 | 0.839 | 0.661 | 0.674 | 0.655 |
| eeg_reg_cue | 27 × 150 | 0.829 | 0.650 | 0.651 | 0.650 |

Kit check (our EEG windows vs the kit's, correlation): s1 r0: 1.0000, s1 r5: 1.0000, s61 r0: 1.0000, s61 r5: 1.0000, s85 r0: 1.0000, s85 r5: 1.0000
