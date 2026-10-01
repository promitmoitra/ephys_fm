# Literature notes (loop B: combining per-person experts)

Short pass, 2026-09-30. Question: how to weight a strong EEGNet expert with a weak but
different Riemannian expert when the combiner is learned on ~840 windows (40 per person).

| Source | Point that matters here |
|---|---|
| Forecast-combination puzzle (Claeson/Stock–Watson line; review: Wang et al. 2022, [arXiv 2205.04216](https://arxiv.org/pdf/2205.04216); "When to choose the simple average", J. Bus. Res. 2016, [link](https://www.sciencedirect.com/science/article/abs/pii/S0148296316303952)) | Equal weights are hard to beat when weights are estimated from few points: estimation error in the weights eats the theoretical gain. Predicts that per-person weights (20 windows per fold) will lose to global ones, and that a global 1-parameter weight is the most we can hope to learn reliably. |
| Gneiting & Ranjan 2013 / Ranjan & Gneiting 2010, JRSS-B 72(1):71, [pdf](https://academic.oup.com/jrsssb/article-pdf/72/1/71/49515284/jrsssb_72_1_71.pdf) | A linear pool of distinct calibrated forecasts is necessarily *un*calibrated (under-confident); fix with a beta-transformed linear pool (BLP). Implication: judge linear pooling by NLL only after recalibration; accuracy is unaffected by a monotone recalibration in the 2-class case only if the transform is symmetric. |
| Log-linear (geometric) pooling, e.g. Neyman & Roughgarden, [arXiv 2202.11219](https://arxiv.org/pdf/2202.11219) | Log pooling takes confident experts more seriously: (0.1%, 99.9%) + (50%, 50%) → (3%, 97%) vs linear (25%, 75%). With a strong, over-confident expert (fine-tuned EEGNet) and a weak one, the two pools can behave very differently: linear pooling lets a weak expert flip decisions only where EEGNet is unsure; log pooling adds a log-odds nudge everywhere. Hölder pools interpolate (α = 1 linear, α = 0 log). |
| Large, Lines & Bagnall 2019, CAWPE, DMKD, [link](https://link.springer.com/article/10.1007/s10618-019-00638-y) | Heterogeneous ensemble weighted by cross-validated accuracy raised to a power (α = 4) beats stacking and equal voting on many small datasets. A cheap, low-variance alternative to fitted weights: weights = (CV accuracy)^α computed per person on calibration data. |
| Kuncheva, *Combining Pattern Classifiers* (2nd ed., Wiley 2014) | Fixed rules (mean, product, max) vs trained combiners; trained combiners need data separate from the base classifiers' training data — here R3 for the dev bank, which is why the bank exists. |
| MI-specific fusion (Riemannian + DL): e.g. TSF-PDER, J. Neural Eng. 2026 ([PubMed 42392147](https://pubmed.ncbi.nlm.nih.gov/42392147/)); Dempster-Shafer fusion of Riemannian features over windows | Mostly feature-level fusion; late fusion of a CNN and a Riemannian classifier is usually a plain average. No strong prior for learned late-fusion weights in MI. |

## Hypotheses this adds

- **H8 (CAWPE):** per-person weights ∝ (cross-validated accuracy on calibration)^α, α
  fixed at 4 as in the paper; low variance because accuracy is estimated from all calibration
  windows, not fitted by NLL. Needs within-calibration CV predictions for each expert, which
  the classical experts can provide cheaply (and EEGNet only with extra fine-tunes).
- **H9 (Hölder / BLP):** a recalibrated linear pool (beta transform) vs log pool; mostly
  relevant to NLL, which matters for downstream soft routing only weakly.
