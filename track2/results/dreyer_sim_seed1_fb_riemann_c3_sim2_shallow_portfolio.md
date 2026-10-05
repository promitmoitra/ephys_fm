# Track 2: fingerprint mixture on Dreyer 2023 (sealed-phase simulation)

14 evaluation people (sim2: 3, 8, 10, 14, 19, 20, 40, 45, 46, 47, 50, 55, 58, 82); calibration = runs R1–R3, hidden test = R4–R6 (1680 windows). 2-class MI, chance 0.5. Seed 1. Fingerprint `fb_riemann`: 1.000 14-way on the hidden runs (chance 0.071). Riemannian experts on (combiner portfolio). EEGNet experts trained. Extra streams: ShallowFBCSPNet.

| Model | Balanced acc (windows) | Mean over people |
|---|---|---|
| pooled | 0.929 | 0.929 |
| control | 0.935 | 0.935 |
| EEGNet experts, oracle | 0.938 | 0.937 |
| EEGNet experts, soft | 0.938 | 0.937 |
| E+T (C3), oracle | 0.938 | 0.938 |
| E+T (C3), soft | 0.938 | 0.938 |
| full combination, oracle | 0.930 | 0.930 |
| full combination, soft | 0.930 | 0.930 |
| shipped mixture (predict) | 0.930 | 0.930 |

`soft` rows mix the per-person experts by the packaged fingerprint's p(person | window); `oracle` rows use the true person. The last row runs the packaged `submission.py` + `mixture.pt` + `config.json` through `predict`. Inference on 1680 windows (CPU): 30.75 s. ZIP: 6.424 MB. Runtime 4353.6 s.
