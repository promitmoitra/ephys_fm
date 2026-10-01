# Checkpoint 1 analysis

Results: `results/confirm.md` (combiners fitted on the full dev bank, applied to R4–R6).

- **ts helps on the test runs:** R1 (equal average) 0.921 oracle vs R0 0.908 (+0.014,
  11 better / 7 worse); soft-routed with the packaged fingerprint 0.909 vs 0.901, with loop
  A's `ts_fb` fingerprint 0.921 vs 0.907.
- **H3 (log-linear + bias, weights 0.83 / 0.40)** matches R1's accuracy (0.921) and has the
  best NLL of every rule (0.214 vs R0 0.235, R1 0.325). The linear average hurts NLL because
  ts is over-confident. H5a (EEGNet sharpened, then averaged) 0.917.
- **Dev and test disagree on sign.** Dev: R1 − R0 = −0.018 (6 better / 10 worse);
  test: +0.014 (11 / 7). H3 − R0: dev −0.005, test +0.013.

**Interpretation.** A per-person accuracy on 40 windows has a standard error of ~0.07; the
paired difference of two rules averaged over 21 people has an SE of roughly 0.01. Differences
of 0.01–0.02 are at the dev bank's resolution limit, so protocol 01's "ts adds nothing" was
not a reliable decision. Two non-noise explanations remain: (i) R3 is special (e.g. fatigue
at the end of the calibration block); (ii) the classical expert gains more from a third
training run than EEGNet does (ts at test is fitted on 120 windows, at dev on 80).

**Direction: DEEPEN the evaluation, not the combiners.** Build a cross-fitted dev bank:
every calibration run held out in turn (2,520 windows, 120 per person, the test bank's size),
which also separates noise from an R3-specific effect (compare the per-fold results).
H3 is the provisional front-runner; it is not yet adopted, because its dev evidence is
negative and it was chosen by the dev NLL rule, not by accuracy.
