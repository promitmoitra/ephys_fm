# Dreyer 2023 confound screen: motor imagery or the screen?

Pooled EEGNet per condition: train on the kit's train subjects, best of 30 epochs on val subjects, tested on unseen test subjects 61–81 (warm-up split). 2 classes, chance 0.5. Seed 0. Window 0–4 s from the cue: arrow 0–1.25 s, feedback bar 1.25–4 s.

| Condition | Channels × samples | Test bal. acc | Acquisition R1–R2 | Online R3–R6 | Per-subject range |
|---|---|---|---|---|---|
| full | 27 × 480 | 0.799 | 0.802 | 0.797 | 0.50–0.93 |
| cue_only | 27 × 150 | 0.780 | 0.784 | 0.778 | 0.46–0.93 |
| feedback_only | 27 × 330 | 0.710 | 0.686 | 0.722 | 0.60–0.85 |
| no_frontal | 24 × 480 | 0.794 | 0.797 | 0.793 | 0.51–0.91 |
| motor_strip | 14 × 480 | 0.732 | 0.718 | 0.738 | 0.47–0.90 |
| shuffled | 27 × 480 | 0.503 | 0.495 | 0.507 | 0.46–0.55 |
