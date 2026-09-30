# Literature notes (loop A: fingerprint)

Light survey at bootstrap (2026-09-30). The aim was to find what matters for a per-window EEG
person-ID model that must hold up over time, not to be exhaustive.

## EEG biometrics across sessions

- **Cross-session identification is the hard case.** EEG-ID papers consistently report
  near-perfect within-session identification that drops across sessions and days. The causes
  are changes in spatial and spectral structure (physiology, electrode re-placement, impedance).
  - Frontiers 2021, selective cross-subject TL in tangent space:
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.779231/full
  - "Interpretable Spectral Features for Cross-Session EEG Biometric Identification and
    Verification", Sensors 2026: https://doi.org/10.3390/s26154811
  - "A Spatial-Spectral-Temporal Representation Method Based on Riemannian Manifold for EEG
    Individual Identification" (TBME 2025, SPD-manifold ID): https://doi.org/10.1109/tbme.2025.3628167
  - Cross-session person ID with hierarchical graph embedding (2024):
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11564420/
- **Our own BNCI result** (`experiments/fingerprint_tangermann/`): tangent space + LR scored
  0.999 within a session and 0.872 across days; EEGNet scored 0.909 across days. Riemann wins
  within a session and loses across days, so **the Dreyer simulation (one session) may flatter
  covariance fingerprints relative to the sealed phase**, whose hidden sessions are probably
  other days.

## Alignment / re-centering

- Zanini et al. 2018 (Riemannian alignment): re-center each domain's covariances at its own
  Riemannian mean. Rodrigues et al. 2019 (Riemannian Procrustes Analysis, IEEE TBME;
  https://pubmed.ncbi.nlm.nih.gov/30596565/) add re-centering, stretching and rotation.
- Relevance: **re-centering removes exactly the person-specific mean covariance that a
  fingerprint relies on.** It is the opposite of what we want for ID. It is useful only if it
  removes a *session* shift while keeping the between-person differences, which would need
  per-person reference data from the new session (not available: `predict` gets no IDs).

## Calibration

- Guo et al. 2017, "On Calibration of Modern Neural Networks" (arXiv:1706.04599): a single
  temperature fitted on held-out data fixes most of the miscalibration of deep nets, while
  keeping argmax and accuracy. For soft routing, NLL and p(true) matter, not just accuracy.
- The held-out data must share the target's shift: a temperature fitted on the same run
  distribution overshoots when the test is farther away in time (a cross-fitted, out-of-run
  temperature is the principled choice here).

## Implications for hypotheses

1. The covariance fingerprint should be evaluated for **decay with run distance** (R1 → R2 vs
   R1 → R3), and on the BNCI cross-day check, before it is trusted.
2. An EEGNet + Riemann ensemble may be robust in both regimes (within session: Riemann; across
   days: EEGNet).
3. Temperatures should be fitted out-of-run.
