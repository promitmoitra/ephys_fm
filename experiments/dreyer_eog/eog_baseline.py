"""Is Dreyer's EOG class signal a cue-triggered eye movement or a prior gaze?

run.py/polarity.py found EOG decoding the cued hand (inverted in part B).
With the pre-cue period (extract_pre.py), each trial's EOG can be referenced
to its own baseline (-2.5 to -1.0 s: fixation cross, before the beep):

  raw        EOG window means (as run.py's slow features)
  change     same minus the trial's baseline mean: a cue-triggered movement
  precue     the baseline window itself: the class cannot be known yet, so
             this must be at chance (negative control)

Logistic regression on per-channel 250 ms bin means, trained on the kit's
train subjects and scored on val (parts A/C) and test (part B) subjects,
over the cue window (0-1.25 s) and the full window (0-4 s). Also the sign of
the right-minus-left change per subject, by Dreyer part.

RESULT: INCONCLUSIVE. The pre-cue control is not at chance (0.565 val /
0.408 test). The pre-cue EOG difference has the opposite sign to the
post-cue one and grows toward the cue, the signature of the pipeline's
zero-phase (non-causal) 0.1 Hz high-pass smearing the large post-cue eye
deflection backwards; previous-trial carryover may add to it (consecutive
trials share a class 42.5% of the time). Re-extracting the EOG with a causal
high-pass is needed before the raw-vs-change comparison means anything.

Usage (repo root, venv active, after extract_pre.py):
    python experiments/dreyer_eog/eog_baseline.py
"""

CAVEAT = ("> **Inconclusive:** the pre-cue control is not at chance. The pipeline's "
          "zero-phase 0.1 Hz high-pass smears the post-cue eye deflection into the "
          "baseline (opposite sign, growing toward the cue). Needs a causal high-pass "
          "re-extraction.\n")

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_pre import BASELINE, CACHE_PRE, PRE_N  # noqa: E402
from run import CUE_END, N_TIMES, OUT, SFREQ  # noqa: E402

BIN = 30   # 250 ms


def bins(X):
    """(N, C, T) -> (N, C * T // BIN) bin means."""
    n, c, t = X.shape
    return X[:, :, :t - t % BIN].reshape(n, c, -1, BIN).mean(-1).reshape(n, -1)


def part(s):
    s = int(s)
    return "A" if s <= 60 else "B" if s <= 81 else "C"


def main():
    d = np.load(CACHE_PRE, allow_pickle=True)
    eog, y, subj, split = d["eog"], d["y"], d["subject"], d["split"]
    names = [str(c) for c in d["ch_eog"]]
    tr = split == "train"
    groups = {"val": split == "val", "test": split == "test"}
    base = eog[:, :, BASELINE[0]:BASELINE[1]].mean(-1, keepdims=True)

    def score(F):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(F[tr], y[tr])
        return {g: float(balanced_accuracy_score(y[m], clf.predict(F[m])))
                for g, m in groups.items()}

    res = {"precue_control": score(bins(eog[:, :, BASELINE[0]:BASELINE[1]]))}
    for wname, hi in (("cue_0-1.25s", CUE_END), ("full_0-4s", N_TIMES)):
        post = eog[:, :, PRE_N:PRE_N + hi]
        res[f"{wname}/raw"] = score(bins(post))
        res[f"{wname}/change"] = score(bins(post - base))

    change = eog[:, :, PRE_N:PRE_N + CUE_END].mean(-1) - base[..., 0]      # (N, 3)
    pol = {}
    for p in "ABC":
        subs = [s for s in np.unique(subj) if part(s) == p]
        diffs = np.array([change[(subj == s) & (y == 1)].mean(0)
                          - change[(subj == s) & (y == 0)].mean(0) for s in subs])
        pol[p] = {n: {"positive": int((diffs[:, i] > 0).sum()), "n": len(subs),
                      "median": float(np.median(diffs[:, i]))} for i, n in enumerate(names)}

    out = {"decoding": res, "polarity_of_change": pol,
           "baseline_s": [(BASELINE[0] - PRE_N) / SFREQ, (BASELINE[1] - PRE_N) / SFREQ]}
    (OUT / "eog_baseline.json").write_text(json.dumps(out, indent=2))
    lines = ["# Dreyer EOG: cue-triggered change vs prior gaze\n", CAVEAT,
             f"Baseline {out['baseline_s'][0]:+.1f} to {out['baseline_s'][1]:+.1f} s "
             "(fixation, before the beep). Logistic regression on 250 ms EOG bin means; "
             "train subjects → val (A/C), test (B). Chance 0.5.\n",
             "| Features | Val | Test |", "|---|---|---|"]
    for k, r in res.items():
        lines.append(f"| {k} | {r['val']:.3f} | {r['test']:.3f} |")
    lines += ["", "Right-minus-left **change from baseline** (0–1.25 s), subjects positive / n "
              "(median):\n", "| Part | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for p, r in pol.items():
        lines.append(f"| {p} | " + " | ".join(
            f"{r[n]['positive']}/{r[n]['n']} ({r[n]['median']:+.3f})" for n in names) + " |")
    (OUT / "eog_baseline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
