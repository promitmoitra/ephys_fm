# Track 2: fingerprint mixture on Dreyer 2023 (sealed-phase simulation)

14 evaluation people (sim2: 3, 8, 10, 14, 19, 20, 40, 45, 46, 47, 50, 55, 58, 82); calibration = runs R1–R3, hidden test = R4–R6 (1680 windows). 2-class MI, chance 0.5. Seed 0. Fingerprint `fb_riemann`: 1.000 14-way on the hidden runs (chance 0.071). Riemannian experts on (combiner portfolio). EEGNet experts trained. Extra streams: ShallowFBCSPNet.

| Model | Balanced acc (windows) | Mean over people |
|---|---|---|
| pooled | 0.932 | 0.932 |
| control | 0.939 | 0.939 |
| EEGNet experts, oracle | 0.938 | 0.937 |
| EEGNet experts, soft | 0.938 | 0.937 |
| E+T (C3), oracle | 0.937 | 0.937 |
| E+T (C3), soft | 0.937 | 0.937 |
| full combination, oracle | 0.931 | 0.931 |
| full combination, soft | 0.932 | 0.932 |
| shipped mixture (predict) | 0.932 | 0.932 |

`soft` rows mix the per-person experts by the packaged fingerprint's p(person | window); `oracle` rows use the true person. The last row runs the packaged `submission.py` + `mixture.pt` + `config.json` through `predict`. Inference on 1680 windows (CPU): 30.4 s. ZIP: 6.425 MB. Runtime 5288.4 s.
