# Protocol 01 results: dev bank (experts on R1–R2 → R3, true ID)

| ID | Combiner | Experts | Scope | Bal acc | NLL | vs R1: Δ, better / worse | Soft (fp_biased) | Params (full fit) |
|---|---|---|---|---|---|---|---|---|
| — | pooled EEGNet | pooled | — | 0.860 | 0.355 | | | |
| — | control (fine-tuned on all calib) | control | — | 0.888 | 0.297 | | | |
| — | ts only | ts | — | 0.681 | 0.624 | | | |
| — | csp only | csp | — | 0.645 | 1.310 | | | |
| R0 | EEGNet only | eegnet | fixed | 0.887 | 0.312 | +0.018, 10 / 6 | 0.877 | `{}` |
| R1 | equal average eegnet+ts | eegnet+ts | fixed | 0.869 | 0.377 | +0.000, 0 / 0 | 0.875 | `{}` |
| R2 | equal average eegnet+ts+csp | eegnet+ts+csp | fixed | 0.750 | 0.438 | -0.119, 2 / 18 | 0.760 | `{}` |
| H1 | linear pool, learned | eegnet+ts | global | 0.887 | 0.305 | +0.018, 9 / 6 | 0.877 | `{"a": [1.19, -1.19]}` |
| H2 | linear pool, learned per person | eegnet+ts | person | 0.891 ± 0.002 | 0.290 | +0.022, 9 / 7 | 0.877 | `{"a": [1.19, -1.19]}` |
| H3 | log-linear + bias (stacking) | eegnet+ts | global | 0.882 | 0.300 | +0.013, 9 / 4 | 0.883 | `{"w": [0.827, 0.397], "b": [0.277]}` |
| H4 | log-linear | eegnet+ts | global | 0.880 | 0.301 | +0.011, 8 / 3 | 0.882 | `{"w": [0.815, 0.385]}` |
| H5a | temperature, then equal average | eegnet+ts | global | 0.888 | 0.362 | +0.019, 11 / 3 | 0.883 | `{"logT": [-1.147, 0.001]}` |
| H5b | temperature, then learned linear pool | eegnet+ts | global | 0.887 | 0.310 | +0.018, 9 / 6 | 0.880 | `{"logT": [-0.308, 0.099], "a": [0.967, -0.967]}` |
| H6 | confidence gate | eegnet+ts | global | 0.886 | 0.310 | +0.017, 9 / 5 | 0.882 | `{"a": [0.888, -0.888], "g": [0.654], "logT": [-0.295, 0.107]}` |
| H7 | linear pool, learned, +csp | eegnet+ts+csp | global | 0.886 | 0.305 | +0.017, 9 / 6 | 0.881 | `{"a": [1.984, -0.923, -1.062]}` |
| H7b | log-linear, +csp | eegnet+ts+csp | global | 0.879 | 0.305 | +0.010, 9 / 3 | 0.881 | `{"w": [0.82, 0.504, -0.048]}` |

840 windows (21 people × 40); one window ≈ 0.0012. Runtime 4 s.
