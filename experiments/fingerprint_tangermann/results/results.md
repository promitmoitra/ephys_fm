# Cross-session fingerprinting: BNCI2014_001

9 subjects, chance = 0.111. Windows: 2592 (session 1), 2592 (session 2). Metric: balanced accuracy, per 4-s window.

## Subject identification

| Model | A. within session 1 (LORO) | B. session 1 → 2 |
|---|---|---|
| riemann | 0.999 | 0.872 |
| psd | 0.937 | 0.877 |
| eegnet | — | 0.909 |
| riemann, shuffled labels (control) | | 0.065 |

### Per-subject recall, session 1 → 2

| Subject | riemann | psd | eegnet |
|---|---|---|---|
| 1 | 1.00 | 0.86 | 0.91 |
| 2 | 0.33 | 0.39 | 0.85 |
| 3 | 0.98 | 0.86 | 0.77 |
| 4 | 0.85 | 0.97 | 0.94 |
| 5 | 0.78 | 0.91 | 0.88 |
| 6 | 0.91 | 0.97 | 0.96 |
| 7 | 0.99 | 0.99 | 0.92 |
| 8 | 1.00 | 0.96 | 0.97 |
| 9 | 1.00 | 1.00 | 0.98 |

## Does routing by inferred identity help MI decoding?

4-class MI, session 1 → 2, Riemannian decoders; routing uses the `eegnet` fingerprint. Chance = 0.25.

| Strategy | Balanced acc (pooled windows) | Mean over subjects |
|---|---|---|
| pooled | 0.490 | 0.490 |
| oracle_subject | 0.572 | 0.572 |
| routed_hard | 0.546 | 0.546 |
| routed_soft | 0.553 | 0.553 |

Runtime: 272.0 s.
