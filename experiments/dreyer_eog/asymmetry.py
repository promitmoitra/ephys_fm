"""Hemispheric-asymmetry and baseline-relative features for Dreyer 2023 MI.

bandpower.py found the classic MI signature (right-minus-left trial contrast
of log P(C3) - log P(C4) negative in mu/beta from ~0.5 s, in 70-80% of people,
surviving EOG removal) but a pooled per-channel band-power decoder reached
only ~0.55-0.57 across people. Two ways to encode that structure:

  asym      log P(left) - log P(right) per homologous pair and band; cancels
            power shared by both hemispheres. ((L-R)/(L+R) = tanh(asym/2)
            carries the same information.) 11 pairs x 4 bands = 44 features.
  *_rel     relative to the trial's own pre-cue baseline (-2.5 to -1.0 s,
            after the fixation cross, before the beep): bandpower_rel is the
            log-ratio form of ERD/ERS; asym_rel removes fixed left/right
            imbalances (anatomy, electrode placement).

Feature sets: bandpower (27 ch x 4 bands), asym, bandpower_rel, asym_rel,
asym_rel+bandpower_rel. Bands: theta 4-8, mu 8-14, beta 14-30, gamma 30-46.

A. Cross-subject (the kit's warm-up split: train subjects -> val (parts A/C)
   and test (part B)), on the cue window (0-1.25 s), the sustained window
   (1.25-4 s), and time-resolved 0.5 s segments.
B. Within-subject, the Track 2 sealed-phase setting: for each of the 87
   people, train on runs R1-R3, test on R4-R6; vs CSP (6) + LDA on 8-30 Hz.
   Paired across people: mean difference, people better / worse, Wilcoxon p.
C. Negative control: decoding from the pre-cue window itself (raw features),
   where the class cannot yet be known, must be at chance.

Signals: EEG with EOG regressed out (`eeg_reg`, primary) and plain `eeg`.
Uses the pre-cue cache from extract_pre.py (its post-cue part equals the
kit's windows).

Usage (repo root, venv active, after extract_pre.py):
    python experiments/dreyer_eog/asymmetry.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import BANDS, SEG, STEP, band_power  # noqa: E402
from extract_pre import BASELINE, CACHE_PRE, PRE_N  # noqa: E402
from run import N_TIMES, OUT, SFREQ  # noqa: E402

PAIRS = [("C1", "C2"), ("C3", "C4"), ("C5", "C6"), ("CP1", "CP2"), ("CP3", "CP4"),
         ("CP5", "CP6"), ("FC1", "FC2"), ("FC3", "FC4"), ("FC5", "FC6"),
         ("F3", "F4"), ("P3", "P4")]
# Cue-relative sample ranges (the cue is at PRE_N in the cached arrays)
WINDOWS = {"cue_0-1.25s": (0, int(1.25 * SFREQ)),
           "sustained_1.25-4s": (int(1.25 * SFREQ), N_TIMES)}
FEATURES = ["bandpower", "asym", "bandpower_rel", "asym_rel", "asym_rel+bandpower_rel"]
N_CALIB_RUNS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def part(s):
    s = int(s)
    return "A" if s <= 60 else "B" if s <= 81 else "C"


def bal(y, p):
    return float(balanced_accuracy_score(y, p))


def span_power(X, lo, hi):
    """Log band power (N, C, bands) averaged over 0.5 s segments, 50% overlap,
    tiling absolute samples [lo, hi)."""
    segs = [band_power(X[:, :, s:s + SEG]) for s in range(lo, hi - SEG + 1, SEG // 2)]
    return np.mean(segs, 0)


class Features:
    """Feature sets for one signal, with the baseline power computed once."""

    def __init__(self, X, li, ri):
        self.X, self.li, self.ri = X, li, ri
        self.base = span_power(X, *BASELINE)

    def asym(self, bp):
        return bp[:, self.li] - bp[:, self.ri]

    def sets(self, bp, rows=slice(None)):
        base = self.base[rows]
        flat = lambda a: a.reshape(len(a), -1)  # noqa: E731
        rel = bp - base
        asym_rel = self.asym(bp) - self.asym(base)
        return {"bandpower": flat(bp), "asym": flat(self.asym(bp)),
                "bandpower_rel": flat(rel), "asym_rel": flat(asym_rel),
                "asym_rel+bandpower_rel": np.hstack([flat(asym_rel), flat(rel)])}

    def window(self, lo, hi):
        return span_power(self.X, PRE_N + lo, PRE_N + hi)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def fit_score(F, y, tr, groups):
    clf = lr().fit(F[tr], y[tr])
    return {g: bal(y[m], clf.predict(F[m])) for g, m in groups.items()}


def cross_subject(d, feats):
    y, split = d["y"], d["split"]
    tr = split == "train"
    groups = {"val": split == "val", "test": split == "test"}
    out = {}
    for wname, (lo, hi) in WINDOWS.items():
        sets = feats.sets(feats.window(lo, hi))
        out[wname] = {f: fit_score(sets[f], y, tr, groups) for f in FEATURES}
    # negative control: the pre-cue window itself (raw features only)
    base_sets = {"bandpower": feats.base.reshape(len(y), -1),
                 "asym": feats.asym(feats.base).reshape(len(y), -1)}
    out["precue_control"] = {f: fit_score(F, y, tr, groups) for f, F in base_sets.items()}
    tres = {f: [] for f in ("bandpower", "asym", "bandpower_rel", "asym_rel")}
    for s in range(0, N_TIMES - SEG + 1, STEP):
        sets = feats.sets(band_power(feats.X[:, :, PRE_N + s:PRE_N + s + SEG]))
        for f in tres:
            tres[f].append({"t_center_s": round((s + SEG / 2) / SFREQ, 3),
                            **fit_score(sets[f], y, tr, groups)})
    out["time_resolved"] = tres
    return out


def csp_lda(X_tr, y_tr, X_te):
    from mne.decoding import CSP
    clf = make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True),
                        LinearDiscriminantAnalysis())
    return clf.fit(X_tr, y_tr).predict(X_te)


def within_subject(d, feats, sig):
    y, subj, run = d["y"], d["subject"], d["run"]
    sos = butter(4, [8, 30], btype="bandpass", fs=SFREQ, output="sos")
    per = []
    for s in sorted(np.unique(subj), key=int):
        m = np.where(subj == s)[0]
        cal, hid = run[m] < N_CALIB_RUNS, run[m] >= N_CALIB_RUNS
        ys = y[m]
        X8_30 = sosfiltfilt(sos, d[sig][m], axis=-1)
        row = {"subject": int(s), "part": part(s)}
        for wname, (lo, hi) in WINDOWS.items():
            bp = span_power(d[sig][m], PRE_N + lo, PRE_N + hi)
            sets = feats.sets(bp, rows=m)
            for f in FEATURES:
                clf = lr().fit(sets[f][cal], ys[cal])
                row[f"{wname}/{f}"] = bal(ys[hid], clf.predict(sets[f][hid]))
            row[f"{wname}/csp_lda"] = bal(ys[hid], csp_lda(
                X8_30[cal][:, :, PRE_N + lo:PRE_N + hi], ys[cal],
                X8_30[hid][:, :, PRE_N + lo:PRE_N + hi]))
        per.append(row)
    keys = [k for k in per[0] if "/" in k]
    summary = {}
    for p in ("all", "A", "B", "C"):
        rows = [r for r in per if p == "all" or r["part"] == p]
        summary[p] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                          "sem": float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(len(rows))),
                          "n": len(rows)} for k in keys}
    paired = {}
    comparisons = [("asym", "bandpower"), ("bandpower_rel", "bandpower"),
                   ("asym_rel", "asym"), ("asym_rel+bandpower_rel", "csp_lda"),
                   ("asym_rel", "csp_lda")]
    for wname in WINDOWS:
        for a, b in comparisons:
            va = np.array([r[f"{wname}/{a}"] for r in per])
            vb = np.array([r[f"{wname}/{b}"] for r in per])
            diff = va - vb
            paired[f"{wname}: {a} vs {b}"] = {
                "mean_diff": float(diff.mean()), "better": int((diff > 0).sum()),
                "worse": int((diff < 0).sum()), "n": len(diff),
                "wilcoxon_p": float(wilcoxon(va, vb).pvalue) if np.any(diff) else 1.0}
    return {"per_subject": per, "summary": summary, "paired": paired}


def main():
    d = np.load(CACHE_PRE, allow_pickle=True)
    d = {k: d[k] for k in d.files}
    ch = [str(c) for c in d["ch_eeg"]]
    li = [ch.index(a) for a, _ in PAIRS]
    ri = [ch.index(b) for _, b in PAIRS]
    out = {"pairs": PAIRS, "bands": BANDS,
           "baseline_s": [BASELINE[0] / SFREQ - PRE_N / SFREQ, BASELINE[1] / SFREQ - PRE_N / SFREQ],
           "windows_s": {k: [v[0] / SFREQ, v[1] / SFREQ] for k, v in WINDOWS.items()}}
    for sig in ("eeg_reg", "eeg"):
        feats = Features(d[sig], li, ri)
        log(f"{sig}: cross-subject")
        out[f"{sig}/cross_subject"] = cross_subject(d, feats)
        log(f"{sig}: within-subject (87 people)")
        out[f"{sig}/within_subject"] = within_subject(d, feats, sig)
    (OUT / "asymmetry.json").write_text(json.dumps(out, indent=2))
    (OUT / "asymmetry.md").write_text(render(out))
    print(render(out), flush=True)


def render(o):
    b0, b1 = o["baseline_s"]
    lines = ["# Dreyer 2023: hemispheric-asymmetry and baseline-relative features\n",
             f"`asym` = log P(left) − log P(right) for {len(o['pairs'])} homologous pairs × "
             f"{len(o['bands'])} bands (θ, μ, β, γ); `bandpower` = per-channel log power "
             f"(27 × 4); `_rel` = minus the same trial's pre-cue baseline ({b0:+.1f} to "
             f"{b1:+.1f} s). Chance 0.5.\n"]
    for sig in ("eeg_reg", "eeg"):
        cs = o[f"{sig}/cross_subject"]
        lines += [f"## `{sig}`: cross-subject (train subjects → val A/C, test B)\n",
                  "| Window | Features | Val | Test |", "|---|---|---|---|"]
        for w in list(o["windows_s"]) + ["precue_control"]:
            for f, r in cs[w].items():
                lines.append(f"| {w} | {f} | {r['val']:.3f} | {r['test']:.3f} |")
        tr = cs["time_resolved"]
        lines.append("\nTime-resolved peak (val): " + "; ".join(
            f"{f} {p['val']:.3f} @ {p['t_center_s']} s (test {p['test']:.3f})"
            for f, p in ((f, max(tr[f], key=lambda r: r["val"])) for f in tr)))
        ws = o[f"{sig}/within_subject"]
        lines += ["", f"## `{sig}`: within-subject (train R1–R3 → test R4–R6), mean ± SEM\n",
                  "| Window / features | All (87) | A (60) | B (21) | C (6) |",
                  "|---|---|---|---|---|"]
        for k in ws["summary"]["all"]:
            lines.append(f"| {k} | " + " | ".join(
                f"{ws['summary'][p][k]['mean']:.3f} ± {ws['summary'][p][k]['sem']:.3f}"
                for p in ("all", "A", "B", "C")) + " |")
        lines += ["", "| Paired comparison | Mean diff | Better / worse (of 87) | Wilcoxon p |",
                  "|---|---|---|---|"]
        for k, r in ws["paired"].items():
            lines.append(f"| {k} | {r['mean_diff']:+.3f} | {r['better']} / {r['worse']} | "
                         f"{r['wilcoxon_p']:.2g} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
