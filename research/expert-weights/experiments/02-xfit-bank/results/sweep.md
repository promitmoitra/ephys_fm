# Protocol 02 results: cross-fitted dev bank (each calibration run held out; true ID)

| ID | Combiner | Scope | Bal acc | NLL | vs R0: Δ, better / worse | Fold R1 | Fold R2 | Fold R3 | Δ vs R0 per fold | Params (full fit) |
|---|---|---|---|---|---|---|---|---|---|---|
| — | pooled alone | — | 0.849 | 0.362 | | 0.857 | 0.830 | 0.860 | | |
| — | control alone | — | 0.877 | 0.310 | | 0.885 | 0.860 | 0.888 | | |
| — | ts alone | — | 0.680 | 0.609 | | 0.665 | 0.694 | 0.681 | | |
| — | csp alone | — | 0.651 | 1.202 | | 0.638 | 0.669 | 0.645 | | |
| R0 | EEGNet only | fixed | 0.875 | 0.303 | +0.000, 0 / 0 | 0.883 | 0.855 | 0.887 | +0.000, +0.000, +0.000 | `{}` |
| R1 | equal average eegnet+ts | fixed | 0.867 | 0.371 | -0.008, 9 / 11 | 0.868 | 0.864 | 0.869 | -0.015, +0.010, -0.018 | `{}` |
| R2 | equal average eegnet+ts+csp | fixed | 0.768 | 0.429 | -0.107, 5 / 16 | 0.765 | 0.789 | 0.750 | -0.118, -0.065, -0.137 | `{}` |
| H1 | linear pool, learned | global | 0.875 | 0.298 | -0.000, 3 / 5 | 0.882 | 0.856 | 0.886 | -0.001, +0.001, -0.001 | `{"a": [1.228, -1.228]}` |
| H2 | linear pool, learned per person | person | 0.882 | 0.288 | +0.007, 5 / 3 | 0.894 | 0.864 | 0.887 | +0.011, +0.010, +0.000 | `{"a": [1.228, -1.228]}` |
| H3 | log-linear + bias (stacking) | global | 0.876 | 0.288 | +0.001, 12 / 7 | 0.880 | 0.869 | 0.879 | -0.004, +0.014, -0.008 | `{"w": [0.826, 0.422], "b": [-0.036]}` |
| H4 | log-linear | global | 0.877 | 0.287 | +0.002, 12 / 7 | 0.879 | 0.874 | 0.879 | -0.005, +0.019, -0.008 | `{"w": [0.826, 0.42]}` |
| H5a | temperature, then equal average | global | 0.877 | 0.360 | +0.002, 11 / 7 | 0.879 | 0.870 | 0.882 | -0.005, +0.015, -0.005 | `{"logT": [-0.823, -0.074]}` |
| H5b | temperature, then learned linear pool | global | 0.875 | 0.300 | -0.000, 3 / 5 | 0.882 | 0.856 | 0.887 | -0.001, +0.001, +0.000 | `{"logT": [-0.124, -0.207], "a": [1.101, -1.101]}` |
| H6 | confidence gate | global | 0.876 | 0.298 | +0.001, 8 / 7 | 0.883 | 0.857 | 0.887 | -0.000, +0.002, +0.000 | `{"a": [0.932, -0.932], "g": [1.254], "logT": [-0.138, -0.176]}` |
| H7 | linear pool, learned, +csp | global | 0.876 | 0.298 | +0.001, 4 / 4 | 0.882 | 0.857 | 0.888 | -0.001, +0.002, +0.001 | `{"a": [2.01, -0.959, -1.05]}` |
| H7b | log-linear, +csp | global | 0.878 | 0.289 | +0.003, 12 / 7 | 0.882 | 0.870 | 0.881 | -0.001, +0.015, -0.006 | `{"w": [0.823, 0.365, 0.024]}` |

2,520 windows (21 people × 120; 840 per fold). Per-person rows (H2): leave-one-run-out within person; their 'full fit' params are global. Runtime 4 s.
