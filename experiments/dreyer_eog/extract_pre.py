"""Re-extract Dreyer windows with a pre-cue period (-3 to +4 s around the cue).

Same pipeline and per-recording scaling as run.py's extract_subject, so the
post-cue part (samples PRE_N onward) equals run.py's cue-locked windows
exactly. Dreyer's trial: fixation cross at -3 s, beep at -1 s, arrow at 0 s.
The baseline used downstream is -2.5 to -1.0 s: after the cross appears,
before the beep and its auditory response.

Cache: data/experiments/dreyer_eog_windows_pre.npz with eeg, eeg_reg, eog,
emg (N, C, 840), y, run, subject, split, ch_*.

Usage (repo root, venv active; run.py's cache must exist for the split):
    python experiments/dreyer_eog/extract_pre.py [--jobs 6]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R  # noqa: E402

PRE_S = 3.0
PRE_N = int(PRE_S * R.SFREQ)                  # cue at this sample
N_TOTAL = PRE_N + R.N_TIMES                   # 840
BASELINE = (int(0.5 * R.SFREQ), int(2.0 * R.SFREQ))   # -2.5 .. -1.0 s
CACHE_PRE = R.CACHE.with_name("dreyer_eog_windows_pre.npz")


def extract_subject_pre(subject):
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    import warnings
    warnings.filterwarnings("ignore")
    from neuralfetch.download import temp_mne_data
    with temp_mne_data(R.DL, clear_dataset_configs=True):
        from moabb.datasets import Dreyer2023
        sessions = Dreyer2023(return_all_modalities=True).get_data([subject])[subject]
    out = {k: [] for k in ("eeg", "eeg_reg", "eog", "emg", "y", "run")}
    for sess in sorted(sessions):
        for r_name in sorted(sessions[sess]):
            raw = sessions[sess][r_name].copy().load_data()
            sf = raw.info["sfreq"]
            onsets = [(a["onset"] - raw.first_time, R.LABELS[a["description"]])
                      for a in raw.annotations if a["description"] in R.LABELS]
            ex = raw.copy().pick(["eeg", "eog"])
            ex.notch_filter(R._notch_freqs(sf), picks="all", verbose=False)
            ex.filter(0.1, 75.0, picks="all", verbose=False)
            ex.resample(R.SFREQ, verbose=False)
            eeg = ex.copy().pick("eeg").get_data()
            eog = ex.copy().pick("eog").get_data()
            B, *_ = np.linalg.lstsq(eog.T, eeg.T, rcond=None)
            eeg_reg = eeg - (eog.T @ B).T
            em = raw.copy().pick("emg")
            em.notch_filter(R._notch_freqs(sf), picks="all", verbose=False)
            em.filter(20.0, None, picks="all", verbose=False)
            em.apply_function(np.abs, picks="all")
            em.filter(None, 8.0, picks="all", verbose=False)
            em.resample(R.SFREQ, verbose=False)
            sigs = {"eeg": R._scale(eeg), "eeg_reg": R._scale(eeg_reg),
                    "eog": R._scale(eog), "emg": R._scale(em.get_data())}
            for t, lab in onsets:
                s = int(round(t * R.SFREQ)) - PRE_N
                if s < 0 or s + N_TOTAL > sigs["eeg"].shape[1]:
                    continue
                for k, v in sigs.items():
                    out[k].append(v[:, s:s + N_TOTAL].astype(np.float32))
                out["y"].append(lab)
                out["run"].append(int(r_name[0]))
    return subject, {k: np.stack(v) if k not in ("y", "run") else np.asarray(v)
                     for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    cue = np.load(R.CACHE, allow_pickle=True)
    split_of = dict(zip(cue["subject"].astype(str), cue["split"].astype(str)))
    subjects = sorted({int(s) for s in split_of})
    R.log(f"extracting {len(subjects)} subjects, -{PRE_S} to +4 s")
    res = Parallel(n_jobs=args.jobs)(delayed(extract_subject_pre)(s) for s in subjects)
    parts = {k: [] for k in ("eeg", "eeg_reg", "eog", "emg", "y", "run", "subject", "split")}
    for s, arrs in res:
        n = len(arrs["y"])
        for k in ("eeg", "eeg_reg", "eog", "emg", "y", "run"):
            parts[k].append(arrs[k])
        parts["subject"].append(np.full(n, str(s)))
        parts["split"].append(np.full(n, split_of[str(s)]))
    d = {k: np.concatenate(v) for k, v in parts.items()}
    for k in ("ch_eeg", "ch_eog", "ch_emg"):
        d[k] = cue[k]
    # The post-cue part must equal run.py's cue-locked windows; trials whose
    # pre-cue period falls before the recording start are dropped here.
    R.log(f"windows: {len(d['y'])} (cue-locked cache: {len(cue['y'])})")
    keep = []
    for s in np.unique(d["subject"]):
        for r in np.unique(d["run"][d["subject"] == s]):
            a = np.where((d["subject"] == s) & (d["run"] == r))[0]
            b = np.where((cue["subject"] == s) & (cue["run"] == r))[0]
            if len(a) == len(b):
                keep.append(float(np.abs(d["eeg"][a][:, :, PRE_N:] - cue["eeg"][b]).max()))
    R.log(f"post-cue check on {len(keep)} runs with equal counts: max abs diff {max(keep):.2e}")
    np.savez(CACHE_PRE, **d)
    R.log(f"saved {CACHE_PRE}")


if __name__ == "__main__":
    main()
