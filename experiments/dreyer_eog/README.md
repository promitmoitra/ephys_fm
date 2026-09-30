# Dreyer 2023: what does the Track 2 warm-up decode?

A pooled EEGNet reaches ~0.80 cross-subject on the Dreyer warm-up (left vs
right hand). `experiments/dreyer_confound/` showed that most of that lives in
the first 1.25 s after the cue. This folder asks what the signal is, using
Dreyer's EOG (3 channels) and wrist EMG (2), which the kit drops.

All analyses use the kit's warm-up split (train subjects → val subjects,
parts A/C → unseen test subjects 61–81, part B). The re-extracted windows
match the kit's exactly (r = 1.0000; post-cue max abs diff 0).

## Scripts

| Script | What | Results |
|---|---|---|
| `run.py` | extract EEG / EOG / EMG / EOG-regressed EEG windows; EEGNet per signal and window; time-resolved slow-waveform decoding | `results/seed{0,1,2}.*` |
| `polarity.py` | EOG right-minus-left sign per subject by Dreyer part; time-resolved decoding on val vs test | `results/polarity.*` |
| `bandpower.py` | band-power decoding; C3/C4 lateralization sign test | `results/bandpower.*` |
| `extract_pre.py` | re-extract with a pre-cue period (−3 to +4 s) | cache only |
| `asymmetry.py` | hemispheric-asymmetry and baseline-relative features, cross- and within-subject, vs CSP + LDA; pre-cue control | `results/asymmetry.*` |
| `eog_baseline.py` | EOG change from the pre-cue baseline | `results/eog_baseline.*` (**inconclusive**, see caveats) |

## Results

**EEGNet by signal (3 seeds, mean ± SD, balanced accuracy, chance 0.5)**

| Signal / window | Val (A/C) | Test (B) |
|---|---|---|
| EEG, 0–4 s | 0.875 ± 0.007 | 0.799 ± 0.003 |
| EEG, cue 0–1.25 s | 0.869 ± 0.004 | 0.779 ± 0.003 |
| EOG, 0–4 s | 0.714 ± 0.002 | 0.248 ± 0.001 (inverted) |
| EOG, cue | 0.713 ± 0.001 | 0.263 ± 0.003 (inverted) |
| Horizontal EOG (EOG3), cue | 0.625 ± 0.003 | 0.379 ± 0.003 (inverted) |
| Wrist EMG, 0–4 s | 0.597 ± 0.001 | 0.633 ± 0.004 |
| Wrist EMG, cue | 0.538 ± 0.004 | 0.588 ± 0.006 |
| EEG, EOG regressed out, 0–4 s | 0.838 ± 0.001 | 0.661 ± 0.000 |
| EEG, EOG regressed out, cue | 0.833 ± 0.005 | 0.651 ± 0.002 |

**Findings**

1. **The strongest signal is an early, cue-locked brain response** (~250 ms;
   slow-waveform decoding peaks at 0.807 on val) that largely survives EOG
   removal (0.783). Its timing fits a visual/attentional response to the
   lateralized arrow better than motor imagery.
2. **Sustained eye position** toward the cued side: EOG alone decodes
   0.60–0.67 throughout the trial. The EOG montage differs between parts:
   right-minus-left EOG2 change is positive in 58/60 part-A subjects and 0/21
   in part B (part C behaves like B), hence the inverted test scores.
3. **Some wrist EMG** (0.59–0.63), building up over the trial.
4. **Genuine motor imagery is present but transfers poorly across people.**
   The C3/C4 mu/beta lateralization has the expected sign in 70–80% of people
   from ~0.5 s (p down to 1e-7, survives EOG removal), but cross-subject
   band-power decoding reaches only ~0.55–0.57. Within a person, CSP + LDA is
   the best pure-MI decoder (0.628 ± 0.016 on 1.25–4 s).
5. **Explicit asymmetry features add nothing** over per-channel band power
   (within-subject +0.004, p = 0.78). **Per-trial baseline correction hurts**
   (−0.03, p ≈ 1e-5): a single-trial 1.5 s baseline is noisy, and the kit's
   per-recording scaling already normalizes.

## Caveats

- **The EOG pre-cue control fails** (0.565 val / 0.408 test), so the EOG
  change-from-baseline analysis is inconclusive. The pre-cue EOG difference
  has the opposite sign to the post-cue one and grows toward the cue: the
  signature of the pipeline's zero-phase (non-causal) 0.1 Hz high-pass
  filter smearing the large post-cue eye deflection backwards. Consecutive
  trials also share a class only 42.5% of the time, so some previous-trial
  carryover is possible. A causal high-pass re-extraction would settle it.
  The EEG band-power pre-cue control is clean (0.49–0.50).
- The same non-causal filtering means the kit's windows contain slight
  leakage from later samples (including beyond 4 s).
- EOG regression is conservative: it also removes brain and
  common-reference signal (correlation with the original EEG ~0.64).
- The time-resolved logistic regression and band-power features are simple
  linear models; the EEGNet conditions are 30 epochs, best on val.

## Reproduce

```bash
python track2/train_mixture.py --epochs 1 --fp-epochs 1 --ft-epochs 1   # kit window cache (or any run)
python experiments/dreyer_eog/run.py --seed 0                           # + extraction cache
python experiments/dreyer_eog/run.py --seed 1 --no-time-resolved
python experiments/dreyer_eog/run.py --seed 2 --no-time-resolved
python experiments/dreyer_eog/polarity.py
python experiments/dreyer_eog/bandpower.py
python experiments/dreyer_eog/extract_pre.py
python experiments/dreyer_eog/asymmetry.py
python experiments/dreyer_eog/eog_baseline.py
```
