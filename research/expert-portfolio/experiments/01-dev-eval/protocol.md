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

## Amendment 1 (2026-10-02, before any new-stream bank existed or was scored)

Condition (a)'s accuracy route now also requires its person-bootstrap 95% CI to exclude zero,
like the NLL route already did. Reason: loop B measured this dev bank's resolution at about
±0.011 (95% CI of a per-person rule difference), so a 0.005 point estimate alone can be noise.
Shipping REVE adds a 69M-parameter model that must load offline; that cost should buy a gain
that is distinguishable from zero.

(a) now reads: dropping the stream from the full combination costs ≥ 0.005 balanced accuracy
**with the CI of the per-person accuracy difference excluding zero**, or ≥ 0.005 NLL with the CI
of the NLL difference excluding zero. (b) and the sim2 / R4–R6 handling are unchanged.
