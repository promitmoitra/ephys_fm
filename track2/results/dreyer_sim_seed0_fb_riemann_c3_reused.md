# Track 2: fingerprint mixture on Dreyer 2023 (sealed-phase simulation)

21 evaluation people (subjects 61–81); calibration = runs R1–R3, hidden test = R4–R6 (2520 windows). 2-class MI, chance 0.5. Seed 0. Fingerprint `fb_riemann`: 0.996 21-way on the hidden runs (chance 0.048). Riemannian experts on (combiner C3). EEGNet experts reused from `outputs/track2_dreyer_sim/submission/mixture.pt`.

| Model | Balanced acc (windows) | Mean over people |
|---|---|---|
| EEGNet experts, oracle | 0.908 | 0.908 |
| EEGNet experts, soft | 0.907 | 0.907 |
| + Riemannian (C3), oracle | 0.921 | 0.921 |
| + Riemannian (C3), soft | 0.921 | 0.921 |
| shipped mixture (predict) | 0.921 | 0.921 |

`soft` rows mix the per-person experts by the packaged fingerprint's p(person | window); `oracle` rows use the true person. The last row runs the packaged `submission.py` + `mixture.pt` + `config.json` through `predict`. Inference on 2520 windows (CPU): 34.47 s. ZIP: 4.037 MB. Runtime 127.2 s.
