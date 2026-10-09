# Track 2: more robust use of neural identity, three integration options (A, B, C)

Date: 2026-10-05 · Branch `docs/identity-integration` · Worktree `.claude/worktrees/t2-int-spec`

## Why

The shipped Track 2 model uses **one** fingerprint model, a filter-bank covariance classifier
with one output per enrolled person. It gives p(person | window) and soft-routes over
**per-person copies** of every expert:

```
window x ──► fingerprint (1 model) ──► p(k | x), k = 1..K
        └──► for every person k: EEGNet_k(x), Riemann_k(x)        ← K copies of each
             combined per person by C3 (shared weights, per-person reliability)
prediction = Σ_k p(k | x) · p_k(class | x)
```

Weak points:

- **Routing errors become class errors,** and there is no fallback. The fingerprint is
  near-perfect within a session (0.996–1.000). Across days it drops to 0.79–0.94 with one
  calibration day (loop A), and the sealed phase is cross-day.
- **Per-person experts see only their own 80–120 calibration windows.** Their benefit shrinks
  on fresh people. On loop C's sim2, E+T beat EEGNet alone by only +0.001, against +0.017 on the
  reused R4–R6.
- **Every stream costs K copies.**

This spec defines the shared evaluation and three alternative integrations. Each is its own
autoresearch loop.

## Goal and success criterion

**Primary:** robustness to **cross-day drift**, the sealed phase's regime. It is measured end to
end with the fingerprint's real errors (no oracle identity).

**Two stages, so that an untouched test remains** (amendment, 2026-10-05, before any code):

1. **Dev decision (17 people).** The loops iterate on the 17 dev people only. An option becomes a
   **candidate** if, pooled over those 17, its gain over the current design is ≥ 0.005 balanced
   accuracy with a person-bootstrap 95% CI excluding zero, and it is not worse than the current
   design on Dreyer sim2 by more than 0.01. If several options are candidates, the one with the
   highest CI lower bound on dev is chosen (ties → cheaper: A < B < C).
2. **Holdout confirmation (8 people, used once).** The 8 holdout people
   (`2026-10-05-identity-integration-holdout.json`) are never loaded by any loop. After the loops
   conclude, only the chosen candidate is run on them, once, against the current design. It is
   **adopted** if its holdout mean gain is ≥ +0.005. The CI is reported, but not required: 8 people
   cannot resolve it. The holdout never chooses between options.

At most one option ships.

**Revision 2026-10-09: option D added** (Section 4b): a batch-level identity prior from neuralprint's
Transition Grammar Biometric Prior. It is evaluated like A, B and C, but **it is not shippable until
the competition organisers confirm** how sealed-phase `predict()` batches are composed (user
decision).

## Section 1: the shared evaluation (binding for A, B and C)

### Datasets (cached, kit format: 120 Hz, 4-s windows)

| Dataset | People (dev + holdout) | Classes | Channels | Sessions | Calibration → test |
|---|---|---|---|---|---|
| BNCI 2014-001 | 9 (6 + 3) | 4 | 22 | 2 | session 1 → session 2 |
| BNCI 2015-001 | 12 (8 + 4) | 2 | 13 | 2 (3 for some) | session 1 → session 2 (session 3: holdout people only, final check) |
| Zhou 2016 | 4 (3 + 1) | 3 | 14 | 3 | sessions 1–2 → session 3 |

**Holdout (locked, seed 20261005; `2026-10-05-identity-integration-holdout.json`):**
- BNCI 2014-001 people 2, 3, 9 (drawn);
- BNCI 2015-001 people 8, 9, 10, 11 (the people with a third session);
- Zhou 2016 person 1 (drawn).

The loops' banks are built **without** these people: they are absent from the fingerprint's
classes, the pooled model's data and every expert. The final confirmation rebuilds the shared
parts with everyone and scores only the holdout people. BNCI 2015-001 session 1 → session 3 for
people 8–11 is reported as a further, later-day check.

Caches:
- `data/experiments/tangermann_windows.npz` (BNCI 2014-001);
- `outputs/t2-fingerprint/cache/BNCI2015_001_windows.npz`;
- `outputs/t2-fingerprint/cache/Zhou2016_windows.npz`.

All are read-only. Within calibration, each person's **last calibration run** is the validation
run used for the pooled model's epoch choice.

### Pipeline per dataset

Everything is trained on calibration sessions only:
- the fingerprint (`fb_riemann`, K-way);
- the pooled EEGNet;
- per-person experts (whole-network fine-tune, lr 1e-4, 50 epochs, last epoch);
- Riemannian experts (ts_C0.1 + reliability);
- integration, as each option defines it.

### Reference rows (computed once, in Phase 0)

1. Pooled EEGNet alone (no identity).
2. **Current design:** E+T with C3 as shipped, soft-routed by the real fingerprint. *This is the
   row every option is compared against.*
3. Current design with oracle identity (the ceiling for routing-based designs).

### Metrics

- **Primary:** test-session balanced accuracy, as per-person paired differences against row 2,
  pooled over the 17 dev people (decisions) or the 8 holdout people (the one final
  confirmation); person-bootstrap 95% CI (5,000 resamples, rng seed 0); 3 seeds for trained parts
  (person score = mean over seeds).
- **Per dataset:** the same numbers, reported but not used for adoption (4–12 people each).
- **Routing diagnostic:** accuracy on test windows the fingerprint gets right vs wrong, per row
  and option.
- **Secondary (honest, within-session):** Dreyer sim2 (loop C's 14 fresh people), seed 0.
  Each option is implemented for the Dreyer pipeline too and run end to end on sim2 against the
  current design.
- Dreyer R4–R6 are **not** used (reused hidden runs; upper bounds only).

### No leakage

Any learned integration parameter is fitted either on calibration data only, or
**leave-one-dataset-out**: fitted on two datasets' dev test sessions, scored on the third. It is
never fitted on the data it is scored on, and never on holdout people. For the final
confirmation, parameters are fitted on all three datasets' dev people. Each loop commits its
protocols before their results.

## Section 2: Option A, pooled fallback with confidence gating

- **Loop:** `int-gate` · branch `exp/integration-gate` · worktree `.claude/worktrees/t2-int-gate` ·
  workspace `research/integration-gate/` · artifacts `outputs/t2-int-gate/`.
- **Integration:**
  p(c|x) = (1 − g(x)) · Σ_k p(k|x) · p_k(c|x) + g(x) · p_pooled(c|x), with
  g(x) = σ(α + β·H(x) + γ·D(x)).
  - H(x) is the fingerprint's entropy normalised by log K.
  - D(x) is the window's distance to the nearest enrolled person's centroid in the
    fingerprint's standardised tangent-space features (centroids stored at calibration).
  - p_pooled is the pooled EEGNet.
- **Fitting:** (α, β, γ) by NLL, leave-one-dataset-out. Pre-registered variants: A1 gate on H
  only; A2 gate on H and D.
- **Cost:** one extra EEGNet and three numbers; no retraining of existing parts.
- **Hypothesis:** the gain concentrates on windows the fingerprint gets wrong; within a session
  (sim2) the gate stays near 0 and changes nothing.
- **Order:** starts first; smallest change.

## Section 3: Option B, one shared network with per-person adapters

- **Loop:** `int-adapters` · branch `exp/integration-adapters` · worktree
  `.claude/worktrees/t2-int-adapters` · workspace `research/integration-adapters/` · artifacts
  `outputs/t2-int-adapters/`.
- **Integration:** one pooled trunk; each person k gets a small adapter; routing
  Σ_k p(k|x) · softmax(head_k(adapted trunk(x))), with **one trunk pass** per window.
- **Pre-registered variants:**
  - **B1:** per-person final layer (head) only.
  - **B2:** per-person head plus a per-person scale and shift of the trunk's final features
    (FiLM-like, applied after the trunk and before the head, so one trunk pass still serves every
    person).
  - Both are fitted on each person's calibration windows with an L2 pull toward the pooled
    values.
- **Trunks:** EEGNet first; ShallowFBCSPNet second. This also settles loop C's pending
  ShallowFBCSPNet question at the cost of one network instead of K. Riemannian experts are
  combined as in C3, with coefficients refitted.
- **Cost:** one trunk + K × (head [+ BN affine]); about K× smaller than per-person copies.
- **Hypothesis:** close to whole-network fine-tunes within a session (BNCI: head-only +0.036 vs
  whole +0.053 over pooled), and more robust across days because it overfits less.

## Section 4: Option C, identity as an input

- **Loop:** `int-conditioned` · branch `exp/integration-conditioned` · worktree
  `.claude/worktrees/t2-int-conditioned` · workspace `research/integration-conditioned/` ·
  artifacts `outputs/t2-int-conditioned/`.
- **Integration:** one network takes the window plus an identity code
  e(x) = Σ_k p(k|x) · E_k, where E_k is a learned per-person embedding. The code is injected by
  FiLM (per-feature scale and shift) after the spatial convolution.
- **Training:**
  - On the fingerprint's own **cross-validated** posteriors over the calibration windows
    (leave-one-calibration-run-out: each run's posteriors come from a fingerprint fitted on the
    other calibration runs), not true IDs, so the network sees realistic identity noise.
  - With **identity dropout:** with probability 0.2 the code is replaced by the uniform-posterior
    code, so the network learns to work without identity.
- **Cost:** one network + K embeddings.
- **Hypothesis:** the best of the three when identity is wrong or unsure.
- **Main risk:** overfitting with 4–12 people per dataset; embeddings are per dataset. Report the
  train–test gap.

## Section 4b: Option D, batch-level identity prior (Transition Grammar Biometric Prior)

- **Loop:** `int-sequence` · branch `exp/integration-sequence` · worktree
  `.claude/worktrees/t2-int-sequence` · workspace `research/integration-sequence/` · artifacts
  `outputs/t2-int-sequence/`.
- **Source:** the neuralprint repo (`/home/promit/Documents/neuralprint`, e.g.
  `experiments/h-bnci2015-transition-fingerprint/code/run_experiment.py`).
  - Each window's fingerprint tangent features are softly assigned to K covariance "states"
    (k-means on calibration; soft membership ∝ exp(−γ·distance), γ = 0.05).
  - A person's grammar is the Dirichlet-smoothed (ε = 1e-3) K × K Markov matrix of expected
    transitions between consecutive windows.
  - neuralprint matches a whole 200-window test stream to enrolled grammars by cosine similarity:
    100% person ID on BNCI 2015-001 and Zhou 2016 across days, 30-day PainLab retest 100% / 0% EER.
  - **Caveats:**
    1. Those are 12 (BNCI 2015) and 4 (Zhou) session-level decisions, not per-window accuracy.
    2. neuralprint's runs used *all* people, including our holdout people. Its K and γ were chosen
       on data that overlaps our holdout, so **D fixes K = 16 and γ = 0.05 a priori** (neuralprint's
       M3CV / PainLab settings) and never tunes them on our dev or holdout data. The code is copied
       into this repo with attribution, not imported.
- **Integration in Track 2:** `predict(X)` gets a batch of windows (the kit: 64, dataset order, no
  shuffling; **undocumented**, and the kit says to rely only on `meta` and the batch).
  - D turns the batch into a prior π(k | batch): a person's evidence is the log-likelihood of the
    batch's soft state sequence under their calibration grammar, plus (arm D3) the pooled
    per-window fingerprint log-posteriors.
  - The routing posterior is then p(k | x, batch) ∝ p(k | x) · π(k | batch)^λ.
  - **Homogeneity gate:** λ = 0 (pure per-window routing) unless the batch's per-window fingerprint
    argmax agrees on one person for ≥ 80% of windows. Mixed-person batches, recording boundaries and
    shuffled order therefore fall back to the current design.
- **Pre-registered arms:**
  - D1: pooled per-window posteriors only (no grammar), the baseline for "does the grammar add
    anything";
  - D2: grammar only;
  - D3: both.

  All three are gated; λ is fitted leave-one-dataset-out.
- **Required robustness tests**, scored like the primary metric:
  - the kit's batching (64, dataset order);
  - shuffled windows;
  - batches mixing two people;
  - batch sizes 8 and 16.

  **D must never score below row 2 on any of them** (gate check).
- **Relation to A/B/C:** D improves *who*; A/B/C improve *how identity is used*. The final
  comparison also scores D combined with the A/B/C winner.
- **Hypothesis:** across days, where the per-window fingerprint drops (0.79–0.94 with one
  calibration day), pooling evidence over a batch of one person's windows recovers near-perfect
  routing. The grammar adds to pooling only if transition structure is more stable across days
  than mean features.

## Section 5: running A, B, C and D in parallel

1. **Precondition:** loop C (`exp/expert-portfolio`) is concluded and merged into `main`. It
   holds the sim2 option in `train_mixture.py`, `track2/models.py`, the N-stream combiner and
   the stream banks that Phase 0 and the loops reuse.
2. **Phase 0, shared harness (once, before the loops):**
   - Branch `exp/integration-harness`, merged to `main`.
   - It provides the cross-day loaders for the three datasets, the shared pipeline (fingerprint,
     pooled EEGNet, experts, Riemannian experts per dataset, 3 seeds), reference rows 1–3, the
     metric and bootstrap code, and the adoption rule as code.
   - Its outputs (`outputs/t2-int-harness/`) are the cached per-window predictions every option
     reuses, so no loop retrains the shared parts.
3. **Each option is its own autoresearch loop,** set up like loops A and B:
   - its own branch and local worktree, created from the main checkout after Phase 0 merges
     (`git worktree add .claude/worktrees/<name> -b <branch> main`), with the shared folders
     symlinked per `CLAUDE.md` rule 6;
   - its own workspace under `research/<loop>/` (state, log, findings, protocols, reports) and
     artifacts under `outputs/<worktree name>/`;
   - its own 20-minute heartbeat in its own Claude Code session, protocols committed before
     results, and ≤ 2 CPU threads.
4. **Order:** A starts first. B, C and D start in parallel once the harness is on `main`.
5. **Finish:**
   - Each loop concludes with findings, a report in `research/<loop>/to_human/` and a PR.
   - A final comparison picks at most one candidate on dev (Goal, stage 1). It confirms that one
     candidate once on the holdout people (stage 2, protocol committed first). Only then does the
     change ship to `submission.py`, followed by the kit contract check (offline).
6. **`CLAUDE.md`:** add one row per worktree when it is created; remove it after its merge.

## Out of scope

- Changing the fingerprint model itself (loop A's domain), except for exposing the features
  option A needs.
- Open-set detection of people outside the enrolled set: the sealed phase tests the same
  evaluation participants it calibrates on.
- The sealed data itself (not released); its 47 channels and 500 Hz are handled when it
  arrives.
