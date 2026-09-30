"""Follow-ups to run.py, from its window cache (no training):

1. EOG polarity per subject: the sign of the right-minus-left cue difference in
   each EOG channel (0-1.25 s mean), grouped by Dreyer part (A 1-60, B 61-81,
   C 82-87). run.py found EOG decoding inverted on the test subjects (part B).
2. Time-resolved decoding (run.py's sliding logistic regression) scored on the
   val subjects as well as the test subjects, to see whether part B is
   different.

Usage (repo root, venv active, after run.py):
    python experiments/dreyer_eog/polarity.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import CACHE, CUE_END, OUT, time_resolved  # noqa: E402


def part(s):
    s = int(s)
    return "A" if s <= 60 else "B" if s <= 81 else "C"


def main():
    d = np.load(CACHE, allow_pickle=True)
    y, subj, split = d["y"], d["subject"], d["split"]
    eog = d["eog"][:, :, :CUE_END].mean(-1)            # (N, 3) cue-period mean
    names = [str(c) for c in d["ch_eog"]]
    rows = []
    for s in np.unique(subj):
        m = subj == s
        diff = eog[m & (y == 1)].mean(0) - eog[m & (y == 0)].mean(0)
        rows.append({"subject": int(s), "part": part(s), "split": str(split[m][0]),
                     **{f"{n}_right_minus_left": float(v) for n, v in zip(names, diff)}})
    summary = {}
    for p in "ABC":
        pr = [r for r in rows if r["part"] == p]
        summary[p] = {"n": len(pr), **{
            n: {"positive": sum(r[f"{n}_right_minus_left"] > 0 for r in pr),
                "median": float(np.median([r[f"{n}_right_minus_left"] for r in pr]))}
            for n in names}}

    tr, va, te = split == "train", split == "val", split == "test"
    tres = {sig: {"val": time_resolved(d[sig], y, tr, va),
                  "test": time_resolved(d[sig], y, tr, te)}
            for sig in ("eeg", "eeg_reg", "eog")}

    out = {"eog_polarity_by_part": summary, "per_subject": rows,
           "time_resolved_val_vs_test": tres}
    (OUT / "polarity.json").write_text(json.dumps(out, indent=2))
    lines = ["# Dreyer EOG polarity by part, and time-resolved decoding on val vs test\n",
             "Sign of the right-minus-left cue difference (0–1.25 s mean of the scaled "
             "EOG) per subject, counted by Dreyer part.\n",
             "| Part | Subjects | " + " | ".join(f"{n} > 0 (median)" for n in names) + " |",
             "|---|---|" + "---|" * len(names)]
    for p, s in summary.items():
        lines.append(f"| {p} | {s['n']} | " + " | ".join(
            f"{s[n]['positive']}/{s['n']} ({s[n]['median']:+.3f})" for n in names) + " |")
    lines += ["", "Time-resolved logistic regression (250 ms windows), val subjects "
              "(parts A/C) vs test subjects (part B).\n",
              "| Centre (s) | " + " | ".join(f"{g} {h}" for g in tres for h in ("val", "test")) + " |",
              "|---|" + "---|" * (2 * len(tres))]
    for i, row in enumerate(tres["eeg"]["val"]):
        lines.append(f"| {row['t_center_s']:.3f} | " + " | ".join(
            f"{tres[g][h][i]['bal_acc']:.3f}" for g in tres for h in ("val", "test")) + " |")
    (OUT / "polarity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
