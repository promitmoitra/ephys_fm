# H3.3: filter-bank design for cross-day identity, with held-out cross-day datasets

**Status:** CONFIRMATORY. Parent: H3.2.

## Mechanism → prediction

H3.2 showed that cross-day identity lives in the alpha/beta range, that delta (1–4 Hz) is
day-specific, and that 18 × 2-Hz bands (4–40 Hz) beat the 6 physiological bands on BNCI
cross-day (0.992 / NLL 0.035 vs 0.974 / 0.071). Two effects may be mixed: (a) excluding delta,
and (b) finer spectral resolution, which resolves individual peak frequencies such as the
alpha peak. **Predictions:**
1. Removing delta alone (`fb5_nodelta`) recovers only part of the gain; finer bands add the rest.
2. The gain saturates around 2 Hz: 1-Hz bands (`fine36`) are no better than `fine18`, because
   window-level covariance estimates get noisier in narrow bands (4 s × 1 Hz).
3. Restricting to 8–30 Hz (`fine11_8_30`) loses a little: theta and low gamma add identity.
4. On the two held-out cross-day datasets, the selected bank beats `fb6` on NLL.

## Arms (tangent space + StandardScaler + LR C = 1, OAS; as H3)

| Arm | Bands |
|---|---|
| `fb6` | 1–4, 4–8, 8–13, 13–20, 20–30, 30–45 (reference, = `ts_fb`) |
| `fb5_nodelta` | `fb6` without 1–4 |
| `fine9_4hz` | 4–8, 8–12, …, 36–40 |
| `fine18` | 4–6, 6–8, …, 38–40 (H3.2) |
| `fine36` | 4–5, 5–6, …, 39–40 |
| `fine11_8_30` | 8–10, …, 28–30 |

## Benchmarks and selection rule (locked)

- **Selection set:** D2 (BNCI2014_001 day 1 → day 2) and D1 (Dreyer R1 → R3), plus the
  locked R1–R2 → R3.
- **Rule:** the selected bank has the lowest D2 NLL among arms whose D1 balanced accuracy is
  within 0.005 of `fb6`'s. Ties (NLL within 0.005): the arm with fewer bands.
- **Held-out cross-day sets (never used to select):** D3 = BNCI2015_001 (12 people, 2-class MI,
  13 channels, day 1 → day 2); D4 = Zhou2016 (4 people, 14 channels, session 1 → sessions 2–3).
  Scored for `fb6`, `fine18` and the selected bank only.

## Decision

If the selected bank (or `fine18`) beats `fb6` on D3 and D4 NLL, it replaces `ts_fb` as the lead
candidate and goes to confirm-2 (Dreyer R4–R6). If it does not, `fb6` stays: the D2 gain would
be dataset-specific.
