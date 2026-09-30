"""Is there motor-imagery band-power information in Dreyer, beyond eye movements?

run.py's time-resolved decoder used slow-waveform features (bin means), which
capture evoked responses and eye position but not oscillatory power, the
classic motor-imagery signature (contralateral mu/beta desynchronization,
building from ~0.5 s after the cue). This adds band-power features, from
run.py's window cache (no re-extraction):

1. Time-resolved decoding: log band power per channel (theta 4-8, mu 8-14,
   beta 14-30, gamma 30-46 Hz) in Hann-windowed 0.5 s segments every 125 ms;
   logistic regression trained on the train subjects, scored on the val
   (parts A/C) and test (part B) subjects. Signals: EEG, EEG with EOG
   regressed out, and the latter restricted to the motor strip (C/CP rows).
2. Sustained window: band power averaged over 1.25-4 s (after the arrow).
3. Classic lateralization, no classifier: per subject, the right-minus-left
   trial contrast of LI = log P(C3) - log P(C4) in mu and beta. Right-hand
   imagery desynchronizes the left hemisphere (C3), so MI predicts a negative
   contrast. Counted per Dreyer part, with a two-sided sign test.

Usage (repo root, venv active, after run.py has cached the windows):
    python experiments/dreyer_eog/bandpower.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import CACHE, N_TIMES, OUT, SFREQ  # noqa: E402

SEG = 60                     # 0.5 s
STEP = 15                    # 125 ms
BANDS = {"theta": (4, 8), "mu": (8, 14), "beta": (14, 30), "gamma": (30, 46)}
MOTOR = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
SUSTAINED = (int(1.25 * SFREQ), N_TIMES)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def part(s):
    s = int(s)
    return "A" if s <= 60 else "B" if s <= 81 else "C"


def band_power(x):
    """Log band power of (N, C, SEG) segments -> (N, C, n_bands)."""
    x = x - x.mean(-1, keepdims=True)
    p = np.abs(np.fft.rfft(x * np.hanning(x.shape[-1]), axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1 / SFREQ)
    return np.stack([np.log10(p[..., (f >= lo) & (f < hi)].mean(-1) + 1e-12)
                     for lo, hi in BANDS.values()], -1)


def sustained_power(x):
    """Band power averaged over 0.5 s segments tiling the sustained window."""
    lo, hi = SUSTAINED
    segs = [band_power(x[:, :, s:s + SEG]) for s in range(lo, hi - SEG + 1, SEG // 2)]
    return np.mean(segs, 0)


def score(feat, y, tr, groups):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    clf.fit(feat[tr], y[tr])
    return {g: float(balanced_accuracy_score(y[m], clf.predict(feat[m])))
            for g, m in groups.items()}


def main():
    d = np.load(CACHE, allow_pickle=True)
    y, subj, split = d["y"], d["subject"], d["split"]
    ch = [str(c) for c in d["ch_eeg"]]
    tr = split == "train"
    groups = {"val": split == "val", "test": split == "test"}
    motor = [ch.index(c) for c in MOTOR]
    signals = {"eeg": (d["eeg"], None), "eeg_reg": (d["eeg_reg"], None),
               "eeg_reg_motor": (d["eeg_reg"], motor)}

    log("time-resolved band-power decoding")
    starts = list(range(0, N_TIMES - SEG + 1, STEP))
    tres = {k: [] for k in signals}
    for s in starts:
        for k, (X, chs) in signals.items():
            seg = X[:, :, s:s + SEG] if chs is None else X[:, chs, s:s + SEG]
            bp = band_power(seg).reshape(len(seg), -1)
            tres[k].append({"t_center_s": round((s + SEG / 2) / SFREQ, 3),
                            **score(bp, y, tr, groups)})
    log("sustained window 1.25-4 s")
    sustained = {}
    for k, (X, chs) in signals.items():
        A = X if chs is None else X[:, chs]
        bp = sustained_power(A)
        sustained[k] = {"all_bands": score(bp.reshape(len(bp), -1), y, tr, groups)}
        for i, b in enumerate(BANDS):
            sustained[k][b] = score(bp[..., i], y, tr, groups)

    log("C3/C4 lateralization")
    c3, c4 = ch.index("C3"), ch.index("C4")
    lat = {}
    for sig in ("eeg", "eeg_reg"):
        lat[sig] = {}
        for band in ("mu", "beta"):
            bi = list(BANDS).index(band)
            rows = []
            for s in starts:
                bp = band_power(d[sig][:, [c3, c4], s:s + SEG])[..., bi]
                li = bp[:, 0] - bp[:, 1]
                contrast = {}
                for sub in np.unique(subj):
                    m = subj == sub
                    contrast[sub] = li[m & (y == 1)].mean() - li[m & (y == 0)].mean()
                row = {"t_center_s": round((s + SEG / 2) / SFREQ, 3)}
                for p in "ABC":
                    v = np.array([c for sub, c in contrast.items() if part(sub) == p])
                    neg = int((v < 0).sum())
                    row[p] = {"n": len(v), "negative": neg, "median": float(np.median(v)),
                              "p_sign": float(binomtest(neg, len(v)).pvalue)}
                rows.append(row)
            lat[sig][band] = rows

    out = {"bands": BANDS, "segment_s": SEG / SFREQ, "step_s": STEP / SFREQ,
           "time_resolved": tres, "sustained_1p25_4s": sustained,
           "lateralization": lat}
    (OUT / "bandpower.json").write_text(json.dumps(out, indent=2))
    (OUT / "bandpower.md").write_text(render(out))
    print(render(out), flush=True)


def render(o):
    t, s, lat = o["time_resolved"], o["sustained_1p25_4s"], o["lateralization"]
    sigs = list(t)
    lines = ["# Dreyer 2023: band-power decoding and C3/C4 lateralization\n",
             "Log band power (θ 4–8, μ 8–14, β 14–30, γ 30–46 Hz) per channel; "
             "logistic regression trained on the kit's train subjects, scored on val "
             "subjects (parts A/C) and test subjects (part B). Chance 0.5. `eeg_reg` = "
             "EEG with EOG regressed out; `_motor` = C and CP rows only.\n",
             "## Sustained window 1.25–4 s (after the arrow)\n",
             "| Signal | All bands val / test | " + " | ".join(
                 f"{b} val / test" for b in o["bands"]) + " |",
             "|---|---|" + "---|" * len(o["bands"])]
    for k, r in s.items():
        lines.append(f"| {k} | {r['all_bands']['val']:.3f} / {r['all_bands']['test']:.3f} | "
                     + " | ".join(f"{r[b]['val']:.3f} / {r[b]['test']:.3f}"
                                  for b in o["bands"]) + " |")
    lines += ["", "## Time-resolved (0.5 s segments, every 125 ms)\n",
              "| Centre (s) | " + " | ".join(f"{k} val | {k} test" for k in sigs) + " |",
              "|---|" + "---|---|" * len(sigs)]
    for i, row in enumerate(t[sigs[0]]):
        lines.append(f"| {row['t_center_s']:.3f} | " + " | ".join(
            f"{t[k][i]['val']:.3f} | {t[k][i]['test']:.3f}" for k in sigs) + " |")
    lines += ["", "## C3/C4 lateralization (no classifier)\n",
              "Per subject: right-minus-left trial contrast of log P(C3) − log P(C4). "
              "Motor imagery predicts a **negative** contrast (right-hand imagery "
              "desynchronizes C3). Cells: subjects negative / n (sign-test p).\n"]
    for sig in lat:
        for band, rows in lat[sig].items():
            lines += [f"**{sig}, {band}**\n",
                      "| Centre (s) | A (1–60) | B (61–81) | C (82–87) |", "|---|---|---|---|"]
            for r in rows[::2]:
                lines.append(f"| {r['t_center_s']:.3f} | " + " | ".join(
                    f"{r[p]['negative']}/{r[p]['n']} (p={r[p]['p_sign']:.2g})"
                    for p in "ABC") + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
