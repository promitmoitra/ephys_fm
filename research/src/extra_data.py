"""Extra cross-day fingerprint benchmarks (MOABB), kit-equivalent preprocessing.

Downloads go under outputs/t2-fingerprint/mne_data (the shared data/ stays read-only
for this loop). Preprocessing copies experiments/fingerprint_tangermann/run.py:
EEG picks -> notch 50/60 Hz + harmonics -> band-pass 0.1-75 Hz -> resample 120 Hz
-> RobustScaler per channel per run -> clamp 20; windows 4 s from the cue.

    python research/src/extra_data.py BNCI2015_001 Zhou2016
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp  # noqa: E402

MNE_DIR = fp.OUT / "mne_data"
MNE_DIR.mkdir(parents=True, exist_ok=True)
for k in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH", "MNE_DATASETS_ZHOU2016_PATH"):
    os.environ[k] = str(MNE_DIR)
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

SFREQ, CLAMP, WIN = 120.0, 20.0, 4.0
N_TIMES = int(WIN * SFREQ)


def cache_path(name):
    return fp.OUT / "cache" / f"{name}_windows.npz"


def _notch(sfreq):
    return sorted({k * b for b in (50.0, 60.0) for k in range(1, 7) if k * b < sfreq / 2})


def preprocess_run(raw, t0):
    import mne
    from sklearn.preprocessing import RobustScaler
    raw = raw.copy().load_data()
    events = mne.find_events(raw, shortest_event=0, verbose=False)
    onsets = (events[:, 0] - raw.first_samp) / raw.info["sfreq"]
    raw.pick("eeg")
    raw.notch_filter(_notch(raw.info["sfreq"]), verbose=False)
    raw.filter(0.1, min(75.0, raw.info["sfreq"] / 2 - 1), verbose=False)
    raw.resample(SFREQ, verbose=False)
    data = np.clip(RobustScaler().fit_transform(raw.get_data().T).T, -CLAMP, CLAMP)
    X, y = [], []
    for t, lab in zip(onsets, events[:, 2]):
        s = int(round((t + t0) * SFREQ))
        if s >= 0 and s + N_TIMES <= data.shape[1]:
            X.append(data[:, s:s + N_TIMES])
            y.append(lab)
    return np.stack(X).astype(np.float32), np.asarray(y)


def build(name):
    import moabb.datasets as md
    ds = getattr(md, name)()
    t0 = ds.interval[0]                      # cue-relative window start
    parts = {k: [] for k in ("X", "task", "subject", "session", "run")}
    for subj in ds.subject_list:
        sessions = ds.get_data([subj])[subj]
        for s_idx, s_name in enumerate(sorted(sessions)):
            for r_idx, r_name in enumerate(sorted(sessions[s_name])):
                X, y = preprocess_run(sessions[s_name][r_name], t0)
                n = len(y)
                parts["X"].append(X)
                parts["task"].append(y)
                parts["subject"].append(np.full(n, subj))
                parts["session"].append(np.full(n, s_idx))
                parts["run"].append(np.full(n, r_idx))
        fp.log(f"{name} subject {subj}: sessions {sorted(sessions)}")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    np.savez(cache_path(name), **out)
    fp.log(f"{name}: X {out['X'].shape}, sessions {np.bincount(out['session']).tolist()}")


def load(name):
    d = np.load(cache_path(name))
    people = sorted(np.unique(d["subject"]))
    idx = {s: i for i, s in enumerate(people)}
    return {"X": d["X"], "person": np.array([idx[s] for s in d["subject"]]),
            "session": d["session"], "run": d["run"], "people": people}


if __name__ == "__main__":
    for name in sys.argv[1:]:
        if not cache_path(name).exists():
            build(name)
