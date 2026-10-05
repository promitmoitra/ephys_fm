# Track 2 comparison page (leakage-free)

Six solvers on the Dreyer 2023 warm-up test set, all trained **only on the warm-up train split**
(the kit's val split picks epochs). No evaluation person's labels are used, so these are
legitimate warm-up estimates. The calibrated mixture is not on this page: its 0.947 on the same
windows is leaky, because it trains on those people's calibration runs.

| Solver | Balanced accuracy | What it is |
|---|---|---|
| ShallowFBCSPNet-pooled | **0.844** | braindecode ShallowFBCSPNet, time constants scaled to 120 Hz (`track2/models.py`), 100 epochs |
| EEGNet | 0.795 | the kit's EEGNet baseline (kit training recipe) |
| Riemann-FB-pooled | 0.685 | 6-band covariance → tangent space → logistic regression (the fingerprint's model, trained on classes) |
| MeanLogReg | 0.681 | the kit's mean-over-time + logistic regression baseline |
| REVE-probe-pooled | 0.547 | frozen REVE (`brain-bzh/reve-base`), mean-pooled embedding + linear head; loads offline |
| Constant | 0.500 | chance floor |

One run, 2026-10-05: `external/2026-competition/tracks/bci_decoding/outputs/benchopt_run_2026-10-05_14h24m12.{parquet,html}`
(the kit clone is git-ignored).

## Reproduce

```bash
# our page solvers' weights → the kit's per-solver folders (train split only)
OMP_NUM_THREADS=4 .venv/bin/python -W ignore track2/bench/train_bench.py --threads 4
cd external/2026-competition
export BENCHOPT_DATA_HOME="$(realpath ../../data)" PYTHONWARNINGS=ignore::FutureWarning \
       MNE_LOGGING_LEVEL=ERROR TQDM_DISABLE=1 HF_HUB_OFFLINE=1
unset COMPET_SUBMISSION_DIR          # each solver reads tracks/bci_decoding/outputs/<name>/
# the kit's own baselines, trained by the kit
benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" -o "BCI-decoding[training=True]" -s MeanLogReg -s EEGNet
# the page (inference only); pass our solvers by absolute path from your worktree
W=<your worktree>
benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" -s Constant -s MeanLogReg -s EEGNet \
    -s $W/track2/bench/shallow_solver.py -s $W/track2/bench/riemann_solver.py -s $W/track2/bench/reve_solver.py
```

benchopt accepts several solver file paths in one run. Use absolute paths: `external/` is a
symlink to the main checkout, so `../../track2` from inside the kit resolves to the main
checkout, not your worktree.

## Notes

- ShallowFBCSPNet beats the kit's EEGNet by 0.05 cross-subject, without calibration. This bears
  on the identity-integration option B (a ShallowFBCSPNet trunk) and on any warm-up-legitimate
  submission. Inside the calibrated mixture it added nothing on fresh people (loop C).
- The REVE probe is weak with mean pooling over channels (loop C findings).
