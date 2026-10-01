# checkpoint-1: test bank (R4–R6), combiners fitted on the dev bank

| ID | Combiner | oracle bal acc / NLL | soft_fp_packaged bal acc / NLL | soft_fp_loopA_tsfb bal acc / NLL | oracle vs R0: Δ, better / worse |
|---|---|---|---|---|---|
| R0 | EEGNet only | 0.908 / 0.235 | 0.901 / 0.244 | 0.907 / 0.234 | — |
| R1 | equal average eegnet+ts | 0.921 / 0.325 | 0.909 / 0.346 | 0.921 / 0.325 | +0.014, 11 / 7 |
| H5a | temperature, then equal average | 0.917 / 0.307 | 0.909 / 0.329 | 0.917 / 0.307 | +0.010, 9 / 6 |
| H3 | log-linear + bias (stacking) | 0.921 / 0.214 | 0.908 / 0.236 | 0.921 / 0.215 | +0.013, 11 / 6 |

2,520 windows (21 people × 120). Params: R0 `{}`; R1 `{}`; H5a `{"logT": [-1.147, 0.001]}`; H3 `{"w": [0.827, 0.397], "b": [0.277]}`
