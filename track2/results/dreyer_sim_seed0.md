# Track 2 prototype: fingerprint mixture on Dreyer 2023 (sealed-phase simulation)

21 evaluation people (subjects 61–81); calibration = runs R1–R3, hidden test = R4–R6 (2520 windows). 2-class MI, chance 0.5. Seed 0. Fingerprint (21-way) on hidden runs: 0.744 (chance 0.048).

| Model | Balanced acc (windows) | Mean over people |
|---|---|---|
| pooled | 0.873 | 0.873 |
| control | 0.890 | 0.890 |
| oracle | 0.908 | 0.908 |
| soft (shipped mixture) | 0.901 | 0.901 |
| hard | 0.901 | 0.901 |

`control` = pooled model fine-tuned on all calibration runs with the same recipe as the experts (epoch-matched). The soft row is computed by reloading the packaged `submission.py` + `mixture.pt` + `config.json`. Inference on 2520 windows (CPU): 18.58 s. ZIP: 0.222 MB. Pooled best epoch 96, fingerprint best epoch 135. Runtime 2569.1 s.
