# Track 2 (BCI decoding): fingerprint-routed mixture

`predict(X)` gets no subject IDs, but every hidden sealed-phase window comes
from one of the 10 evaluation participants, whose sessions 1–3 are released
as labeled calibration. The submission therefore ships:

- **experts**: one copy of a pooled EEGNet per evaluation participant,
  fine-tuned (all weights, lr 1e-4, 50 epochs) on that participant's
  calibration data;
- **fingerprint**: an EEGNet that recognises the participant from the window;
- `predict(X)` mixes the experts by the fingerprint's posterior:
  `p(class | x) = Σ_s p(s | x) · p_s(class | x)`.

On BNCI2014_001 this beat an epoch-matched pooled control by +0.034 ± 0.007
(5/5 seeds) without IDs; see `experiments/fingerprint_tangermann/`.

## Current best model

1. **Input**: the kit's windows as-is (0–4 s from the cue, per-recording
   robust scaling). No extra preprocessing: baseline correction and
   asymmetry features did not help (`experiments/dreyer_eog/`).
2. **Pooled EEGNet** on all labeled data (training people + evaluation
   people's calibration sessions); AdamW lr 1e-3, epoch picked on held-out
   people.
3. **Per-person experts**: whole-network fine-tune of the pooled model per
   evaluation person on their calibration data; lr 1e-4, 50 epochs, **last
   epoch** (no epoch selection).
4. **Fingerprint (winning `complex_6b` architecture, PR #15)**: a 6-band complex
   Hermitian covariance model recognises the person from one window: 6 bands
   (1–4, 4–8, 8–13, 13–20, 20–30, 30–45 Hz) → folded Butterworth + Hilbert
   analytic signals → trace-shrunk Hermitian covariance ($\rho = 0.05$) →
   complex tangent space at the complex Fréchet geometric mean → standardised
   multinomial logistic regression (C = 1), fitted on all calibration sessions
   pooled. Shipped as `submission.ComplexFBFingerprint` (complex64 / float32).
   Lifts cross-day ID BA from **0.9329 → 0.9859** and cuts ID NLL from 0.1720 → 0.0642.
5. **Soft routing** in `predict(X)` using the fingerprint posterior $p(k \mid x)$.
6. **Per-person Riemannian expert with individualized $\mu/\beta$ 2-band splitting**
   (`mu_beta_2band`, Protocol 02, PR #16): estimates each participant's individual
   $\mu$ peak frequency $f_{\mu,k} \in [7, 14]$ Hz from calibration PSD via DPSS
   multitaper spectra, and splits $8–30$ Hz into $\mu_k = [f_{\mu,k}-2.5, f_{\mu,k}+2.5]$ Hz
   and $\beta_k = [f_{\mu,k}+2.5, 30.0]$ Hz (0.5–4 s crop). Raw concatenated tangent
   vectors are fitted with Logistic Regression (C = 0.1). Lifts standalone Riemannian
   accuracy by **+0.0251** static and **+0.0208** with EMA.
7. **Causal Online Tangent EMA Adaptation** (Protocol 01, PR #16):
   Test-time reference covariance adaptation running online in streaming or batch
   mode without task labels. Updates running tangent barycenter:
   $\mu_k^{(t)} = (1 - \alpha \gamma_{tk})\mu_k^{(t-1)} + \alpha \gamma_{tk} f_{tk}$ ($\alpha = 0.05$),
   and subtracts $\mu_k^{(t)}$ before classification. Zero future leakage, verified
   under FM-10 random permutations. Combined with steps 4 & 6, lifts 17-person
   cross-day Track 2 mixture from **0.8264 → 0.8522** (and 0.921 → **0.932** on Dreyer R4–R6).
   Shipped as `submission.RiemannExperts` + `LogLinearCombiner` (float64).

| Evidence | Result |
|---|---|
| **Wired Winning Package (seed 0, Dreyer R4–R6)** | `predict` on R4–R6: **0.932** (previous package 0.921, +0.011); fingerprint 0.991; EEGNet experts soft 0.907, + Riemannian 0.932 |
| **17-Person Cross-Day Dev Benchmark (seeds 0–2)** | `row 2` (shipped baseline): 0.8264 → `complex_6b` + Protocol 01 EMA + Protocol 02 $\mu/\beta$ 2-band: **0.8522** (+0.0258, 15W/2L) |
| **Cross-Day Identity Identification (17 Dev People)** | Real OAS 6-band: 0.9329 (NLL 0.1720) → `complex_6b`: **0.9859** (NLL 0.0642, −63% entropy) |
| BNCI, 5 seeds, cross-session | mixture +0.034 ± 0.007 over the epoch-matched control, 5/5 seeds (oracle ID +0.045) |
| Shipped package, Dreyer simulation, seed 0 (previous) | `predict` on R4–R6: 0.921; fingerprint 0.996; EEGNet experts soft 0.907, + Riemannian 0.921 |
| Dreyer sim2, 14 fresh people, full pipeline, seeds 0–2 | EEGNet experts 0.9355 → + Riemannian (step 6) 0.9365; with online EMA 0.9369 |
| Contract & Speed | Self-contained in pure PyTorch; passes all unit tests (39/39 passing); inference on 2,520 windows (CPU, 2 threads): 46.59 s; ZIP: 50.36 MB |

**Tried and dropped**: EEGNet fingerprints (0.745 ± 0.10 on R3 at a fixed
budget, seed-sensitive; the shipped 0.857 picked its epoch on R3), EEGNet +
Riemann fingerprint ensembles (below Riemann alone), finer filter banks (won
on BNCI2014, lost on two held-out datasets), broadband covariance fingerprints
(delta is day-specific: 0.872 across days); per-person heads trained from scratch (−0.030 vs
pooled even with oracle ID, 0/5 seeds); BatchNorm-statistics-only adaptation
(±0.000); epoch selection on small validation sets; hard routing (≤ soft);
CSP experts or equal averaging of all expert types in the mixture (−0.004,
−0.068); a *plain* average with the Riemannian expert (its sign flips
between calibration folds, −0.018 to +0.010: the unregularised Riemannian
expert is over-confident); confidence gating, filter-bank or broadband
Riemannian experts; explicit hemispheric-asymmetry features (≈ per-channel band power);
per-trial pre-cue baseline correction (−0.03 within-subject); a second CNN
expert stream, ShallowFBCSPNet (dev +0.011, but fresh-people sim2 −0.005 and
BNCI −0.004, 5/5 seeds negative; loop C); a frozen REVE probe with mean pooling
(0.581 alone on Dreyer; adds nothing; loop C).

**What it uses on Dreyer** (`experiments/dreyer_confound/`,
`experiments/dreyer_eog/`): mostly an early cue-locked brain response
(~250 ms), plus sustained eye position and some wrist EMG. Genuine motor
imagery is present but transfers poorly across people, which is where
per-person calibration helps.

**Next, in priority order**: the identity-integration loops
(`docs/superpowers/specs/2026-10-05-identity-integration-design.md`): a gated
pooled fallback (A), one shared network with per-person adapters (B) and
identity as a network input (C), judged across days on 17 dev people and
confirmed once on 8 locked holdout people. Then re-validate on the 2026 Graz + BrainHero data
(47 channels incl. EOG/EMG, 3 classes) once released; there, refit the
step-6 coefficients on a leave-one-calibration-session-out bank (across a day
gap the EEGNet expert becomes over-confident and the best Riemannian weight
rises: BNCI prefers ≈ 0.67 / 0.83 over Dreyer's 0.81 / 0.42). More seeds of
the full pipeline (only the step-6 gain has 3 seeds).

## Files

| File | Role |
|---|---|
| `submission.py` | the uploaded code: `Solver(CompetSolver)` + `FingerprintMixture`; maps evaluation channels to training channels **by name** and fails loudly on missing channels, `n_times` or `n_classes` mismatches |
| `train_mixture.py` | trains pooled EEGNet, experts, fingerprint, Riemannian experts and combiner on Dreyer 2023, packages `<out>/track2_dreyer_sim.zip` (`submission.py`, `mixture.pt`, `config.json` at the ZIP root), and scores the shipped code path with ablations. `--reuse-eegnet` loads the EEGNet experts of an existing `mixture.pt`; `--fingerprint eegnet` / `--no-riemann-experts` build the older designs |
| `riemann_parts.py` | training side of the covariance models: fits loop A's fingerprint and loop B's per-person Riemannian experts with scipy / pyriemann / sklearn and exports them as tensors for `submission.py`; holds the C3 coefficients |
| `models.py` | `make_model`: EEGNet or ShallowFBCSPNet with time constants scaled to the sampling rate |
| `reve_parts.py` | training side of a frozen-REVE probe stream (resampling, offline encoder, heads); not shipped |
| `bench/` | leakage-free benchmark-page solvers (ShallowFBCSPNet, REVE probe, filter-bank Riemann; trained on the warm-up train split) |
| `tests/` | `unittest` tests: `python -m unittest discover -s track2/tests -t .` |
| `classical_experts.py` | the original per-person CSP / Riemannian expert evaluation (plain averaging; superseded by step 6) |
| `results/` | per-seed results |

```bash
python track2/train_mixture.py --seed 0      # everything, ~55 min on 8 CPUs
# new fingerprint + Riemannian experts on the seed-0 EEGNet experts (~2 min)
python track2/train_mixture.py --threads 4 --out outputs/t2-integration/fb_c3 \
    --reuse-eegnet outputs/track2_dreyer_sim/submission/mixture.pt
```

## Dreyer 2023 as a sealed-phase simulation

Dreyer has one session of six runs per person. The kit's test people
(subjects 61–81) play the evaluation participants: runs R1–R3 are their
calibration data, R4–R6 the hidden test. The training pool is the kit's
train split; the kit's val split selects the pooled model's epoch.

> ⚠️ **Do not upload this build to the warm-up leaderboard.** Its training
> uses labeled runs of the warm-up test subjects, so its warm-up score is
> inflated and would contaminate the leaderboard. It validates the design and
> the submission contract. A warm-up-legitimate build can only use the train
> split, where the test people are unseen, so there is nothing to calibrate
> on and the mixture reduces to a similarity-weighted ensemble of other
> people's experts.

## Contract check

The packaged ZIP, unzipped read-only, runs through the kit exactly as
Codabench ingestion does:

```bash
cd external/2026-competition
COMPET_SUBMISSION_DIR=$UNZIPPED benchopt run tracks/bci_decoding \
    -d "BCI[study=dreyer2023]" -s $UNZIPPED/submission.py
```

The current package (`outputs/t2-integration/fb_c3/`) passes: 5,040 warm-up
test windows in 73 s on CPU (21 EEGNets, the filter-bank fingerprint and 21
Riemannian experts). The previous package passed in ~1 min with 22 EEGNets
(budget: 60 min on one A100).

## Before the sealed phase

- Retrain on the 2026 Graz + BrainHero data: 47 channels (43 EEG, 2 EMG,
  2 EOG) at 500 Hz, 3 classes, 10 training + 10 evaluation participants.
  The channel mapping is by name, so the evaluation `meta["ch_names"]` must
  contain every training channel.
- The fingerprint no longer bounds the gain on Dreyer (0.996; soft = oracle).
  Across days it relies on calibration from several days (loop A: one day →
  0.79–0.94 on held-out days, two days → 0.975–1.000).
- The covariance models bake the window length into their filter matrices
  (T × T per fingerprint band). At the sealed data's sampling rate, check
  their size and speed (at 500 Hz, 4 s: ≈ 100 MB for the fingerprint in
  float32), or resample before filtering.
- Declare the external data and a compute estimate (Rule 04).
