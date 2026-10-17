# Findings: Fingerprint-Routed Covariance Re-Centering for Task Decoding (§2a)

- **Branch / Worktree:** `exp/recenter-pooled` (`.claude/worktrees/t2-recenter`)
- **Pre-registered protocol:** [`experiments/00-protocol/protocol.md`](experiments/00-protocol/protocol.md) (locked before evaluation in commit `6cd92a4`)
- **Generated results:** [`results/recenter_results.json`](results/recenter_results.json), [`results/recenter_results.md`](results/recenter_results.md)
- **Implementation & tests:** [`src/recenter.py`](src/recenter.py), [`src/evaluate_recenter.py`](src/evaluate_recenter.py), [`tests/test_recenter.py`](tests/test_recenter.py) (`5/5` unit tests passing)

---

## 1. Executive Summary

1. **H2a.1 (Does routing the reference covariance $\bar{C}_k$ improve a pooled tangent-space decoder?): Yes.**
   - On the 17 cross-day dev people, re-centering each participant's covariances by their calibration Fréchet mean $\bar{C}_k^{-1/2}$ before fitting a single shared `LogisticRegression(C=0.1)` lifts standalone pooled tangent-space balanced accuracy from **0.6905** (`ts_global_pooled`) to **0.7060** (`ts_recenter_routed`, **+0.0155** `[-0.0068, +0.0395]`, 9 better / 6 worse / 2 tied) and **0.7125** under oracle identity (`ts_recenter_oracle`, **+0.0221** `[+0.0015, +0.0442]`, 10 better / 5 worse / 2 tied).
   - The largest gain is on 22-channel 4-class `bnci2014` (**0.5347 $\to$ 0.5903**, **+0.0556**).
   - Single-pass whitener interpolation $\bar{W}(x) = \sum_k p(k \mid x)\bar{C}_k^{-1/2}$ (`ts_recenter_whitener`, **0.7066**) matches the $K$-pass tangent mixture (`ts_recenter_routed`, **0.7060**) within **+0.0006** `[-0.0026, +0.0045]` (12/17 exact ties) while requiring only **1 eigendecomposition per window** instead of $K$.
2. **H2a.2 (Does 1 pooled classifier on re-centered covariances beat $K$ per-person classifiers on the same calibration data?): Regime-dependent — yes when $F \gtrsim N_{\text{cal}}$, no when $N_{\text{cal}} \ge 200$.**
   - **Helps in high-dimension / smaller-calibration regimes:**
     - On `dreyer_sim2` (27 channels $\to 378$ tangent features, ~120 calibration windows/person), `ts_recenter_routed` scores **0.6911** (73-subject pool) and **0.7018** (14-subject calibration pool), beating `ts_per_person_routed` (**0.6762**) by **+0.0149** to **+0.0256**.
     - On `zhou2016` (3 people, ~100–119 calibration windows/person, 3 classes), `ts_recenter_routed` scores **0.7933** (`ts_recenter_oracle`: **0.8222**) vs `ts_per_person_routed` **0.7556** (**+0.0378** routed, **+0.0889** oracle).
   - **Underfits when per-person calibration is abundant ($N_{\text{cal}} = 200\text{--}288$):**
     - On `bnci2014` (288 calibration windows/person) and `bnci2015` (200 calibration windows/person), `ts_recenter_routed` trails `ts_per_person_routed` (`0.5903` vs `0.6672` on `bnci2014`; `0.7600` vs `0.7881` on `bnci2015`).
     - Pooled across all 17 cross-day dev people, `ts_recenter_routed − ts_per_person_routed` is **-0.0337** `[-0.0652, +0.0013]` (5 better / 12 worse), and even with **oracle identity** (`ts_recenter_oracle − ts_per_person_oracle`) it is **-0.0252** `[-0.0606, +0.0146]`.
3. **Level 2 (End-to-End Track 2 Integration vs `row 2`): Negative for replacing `row 2`; positive vs `row 1` (`pooled_eegnet`).**
   - A fully pooled system with **zero per-person classifiers** (`E1_fully_pooled_recenter` = 1 `pooled_eegnet` + 1 pooled `ts_recenter` where the fingerprint routes only $\bar{C}_k^{-1/2}$) lifts 17-person cross-day accuracy from **0.7652** (`row 1`) to **0.7885** (**+0.0232** `[-0.0010, +0.0460]`; **0.7937** / **+0.0285** `[+0.0062, +0.0499]` with oracle ID).
   - However, against `row 2` (`current` per-person E+T mixture, **0.8264**), all three Level 2 arms fall short of the pre-registered Track 2 adoption rule ($\ge +0.005$, 95% CI lower bound $> 0$):
     - `E1_fully_pooled_recenter`: **0.7885** (**-0.0379** `[-0.0556, -0.0215]` vs `row 2`, 3W / 14L)
     - `E2_eeg_experts_plus_recenter`: **0.8003** (**-0.0261** `[-0.0397, -0.0145]` vs `row 2`, 5W / 12L)
     - `E3_eeg_experts_plus_shrunk_riemann`: **0.8176** (**-0.0088** `[-0.0174, -0.0017]` vs `row 2`, 6W / 11L)

---

## 2. Geometric Diagnosis: Why Re-Centering Alone Cannot Replace Per-Person Classifiers

In Riemannian Procrustes Analysis (Rodrigues et al., 2019), inter-subject covariance alignment has three distinct steps on $\mathcal{S}_{++}^C$:
1. **Re-centering (parallel transport to $I_C$):** $\tilde{C}_i = \bar{C}_k^{-1/2} C_i \bar{C}_k^{-1/2}$, which aligns each person's **unlabeled session barycenter** to $I_C$ (`zero mean` in $T_I \mathcal{S}_{++}^C$).
2. **Dispersion scaling:** normalizing tangent dispersion across subjects.
3. **Tangent-space rotation (supervised Procrustes):** rotating each subject's class-conditional offsets so that class $c$ points in the same tangent direction across subjects.

`ts_recenter_*` performs Step 1 (which only requires unlabeled calibration covariances per person to compute $\bar{C}_k$). That successfully eliminates first-order resting/geometry shift (`+0.0556` on `bnci2014` over `ts_global_pooled`), **but** different participants still have different subject-specific spatial-frequency ERD dipole orientations $w_k \in T_I \mathcal{S}_{++}^C$. When a participant has 200–288 labeled calibration windows (`bnci2014`, `bnci2015`), fitting an individualized linear readout $w_k$ on $T_{\bar{C}_k}\mathcal{S}_{++}^C$ directly captures that subject's rotation, whereas a single pooled hyperplane $w_{\text{pool}}$ averages across incompatible subject topographies (e.g., `bnci2014` subject 4: `0.5000` pooled re-centered vs `0.6806` per-person LR).

Conversely, when the feature dimension $F = C(C+1)/2$ is large relative to $N_{\text{cal}}$ (`dreyer_sim2`: $F=378, N_{\text{cal}}\approx 120$) or per-person calibration is small (`zhou2016`: $N_{\text{cal}}\approx 100$), variance in estimating $w_k$ dominates inter-subject rotation bias, and the pooled re-centered hyperplane wins (`+0.0256` on `dreyer_sim2`, `+0.0378` routed / `+0.0889` oracle on `zhou2016`).
