# Track 2 prototype: per-person CSP / Riemannian experts (Dreyer simulation)

21 evaluation people; experts fitted on calibration runs R1–R3, scored on hidden runs R4–R6. `csp` = CSP(6) + shrinkage LDA; `ts` = OAS covariance → tangent space → logistic regression; both 8–30 Hz, 0.5–4.0 s. `eegnet` = the fine-tuned EEGNet experts of the packaged seed-0 mixture; combinations average class probabilities per person. Routing by the same fingerprint. Chance 0.5. Reference (seed 0): pooled EEGNet 0.873, epoch-matched control 0.89.

| Experts | Oracle ID | Soft routing | Hard routing | Soft vs `eegnet`: Δ, people better / worse |
|---|---|---|---|---|
| eegnet | 0.908 | 0.901 | 0.901 | — |
| csp | 0.671 | 0.646 | 0.643 | -0.255, 2 / 18 |
| ts | 0.688 | 0.677 | 0.673 | -0.223, 4 / 17 |
| eegnet+csp | 0.889 | 0.896 | 0.875 | -0.004, 10 / 10 |
| eegnet+ts | 0.921 | 0.909 | 0.904 | +0.008, 11 / 9 |
| csp+ts | 0.688 | 0.673 | 0.664 | -0.228, 3 / 17 |
| eegnet+csp+ts | 0.824 | 0.833 | 0.807 | -0.068, 6 / 14 |

Runtime 88.6 s.
