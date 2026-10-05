# Research log: loop C (expert portfolio)

## 2026-10-02

- Workspace created; protocol 00 locked (evaluation and ship rule from the spec).
- sim2 people drawn (seed 20261002) and locked: 3, 8, 10, 14, 19, 20, 40, 45, 46, 47, 50, 55, 58, 82.
- **Amendment 1** (before any scoring): ship rule (a)'s accuracy route now also requires its person-bootstrap CI to exclude zero (side-agent note: dev bank resolves only ±0.011).
- λ_person for REVE (protocol 01 rule): 0.01 → 0.5575, 0.1 → 0.5639, **1.0 → 0.5810** (REVE alone, dev). Chosen: 1.0.
- Protocol 01 run (amendment 1 applied). **ShallowFBCSPNet ships; REVE (mean-pooled) does not.**
  E+T+S 0.894 / NLL 0.253 vs E+T 0.883 / 0.278 on dev. BNCI: E+T+S −0.004 vs E+T (5/5 seeds
  negative, passes the −0.005 bar narrowly). Fixed a 4-class bias bug in PortfolioLogLinear
  found by the BNCI run (test added).

## 2026-10-03

- Protocol 02: sim2 (honest) E+T+S − E+T = −0.005 [−0.016, +0.004]; E+T − EEGNet = +0.001;
  R4–R6 (upper bound) +0.014. No veto. Neither honest check (sim2, BNCI) shows ShallowFBCSPNet
  helping; asking the user before packaging (Task 10).

## 2026-10-05

- **User decision:** ShallowFBCSPNet not shipped (honest checks sim2 −0.005, BNCI −0.004). Task 10
  (packaging) skipped; the question moves to identity-integration option B.
- **Loop C concluded.** Findings written; `track2/README.md` marks R4–R6 rows as upper bounds and
  adds the sim2 row; README experiments row added.
