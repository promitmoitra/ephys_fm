# Protocol 01: dev-bank and BNCI evaluation of the new streams

Locked before any new-stream bank is scored.

- **REVE per-person regularisation:** λ_person ∈ {0.01, 0.1, 1.0}, chosen on the dev bank by REVE
  stream-alone true-ID balanced accuracy (ties within 0.002 → larger λ). λ_pool = 1e-3 fixed.
  The chosen λ is used everywhere after (BNCI, test).
- **Streams:** E = EEGNet experts (loop B bank), T = Riemannian ts_C0.1 + reliability,
  R = REVE probe, S = ShallowFBCSPNet experts.
- **Combinations** (PortfolioLogLinear, leave-one-person-out): E+T (shipped), E+T+R, E+T+S,
  E+T+R+S. Drop-one ablations from E+T+R+S.
- **Ship rule:** protocol 00 (a) on the dev bank, (b) on BNCI with weights fitted on the full
  Dreyer dev bank and applied unchanged.
- **Shortcut diagnostic:** early-only and late-only balanced accuracy for E (loop B's saved dev
  experts, via `stream_bank.py eegnet-shortcut`), S and R; reported, not used for the ship
  decision.
