"""Cross-day datasets for the identity-integration loops (kit format: 120 Hz, 4-s windows).

Calibrate on every session but the last, test on the last. Within calibration, each person's
last run is validation (pooled-model epoch choice). A person with a single calibration run has
it split into two contiguous halves that act as runs 0 and 1 (BNCI 2015-001).
"""

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PATHS = {
    "bnci2014": REPO / "data" / "experiments" / "tangermann_windows.npz",
    "bnci2015": REPO / "outputs" / "t2-fingerprint" / "cache" / "BNCI2015_001_windows.npz",
    "zhou2016": REPO / "outputs" / "t2-fingerprint" / "cache" / "Zhou2016_windows.npz",
}
SPLITS = {"bnci2014": ([0], 1), "bnci2015": ([0], 1), "zhou2016": ([0, 1], 2)}
DATASETS = tuple(PATHS)
SFREQ = 120.0
HOLDOUT = REPO / "docs" / "superpowers" / "specs" / "2026-10-05-identity-integration-holdout.json"


def holdout():
    import json
    return json.loads(HOLDOUT.read_text())["holdout"]


def load(name, part="dev"):
    full = np.load(PATHS[name], allow_pickle=True)
    keep = {"dev": ~np.isin(full["subject"], holdout()[name]),
            "holdout": np.isin(full["subject"], holdout()[name]),
            "all": np.ones(len(full["subject"]), bool)}[part]
    raw = {k: full[k][keep] for k in ("X", "task", "subject", "session", "run")}
    cal_sessions, test_session = SPLITS[name]
    subj, ses, run = raw["subject"], raw["session"], raw["run"]
    people = np.unique(subj)
    person = np.searchsorted(people, subj)
    y = np.searchsorted(np.unique(raw["task"]), raw["task"])
    calib = np.isin(ses, cal_sessions)
    test = ses == test_session
    unit = np.full(len(y), -1)
    val = np.zeros(len(y), bool)
    for k in range(len(people)):
        idx = np.where(calib & (person == k))[0]
        keys = sorted(set(zip(ses[idx].tolist(), run[idx].tolist())))
        if len(keys) == 1:                                  # single run: two contiguous halves
            half = len(idx) // 2
            unit[idx[:half]], unit[idx[half:]] = 0, 1
        else:
            pos = {key: i for i, key in enumerate(keys)}
            unit[idx] = [pos[(s, r)] for s, r in zip(ses[idx], run[idx])]
        val[idx] = unit[idx] == unit[idx].max()
    return {"X": raw["X"].astype(np.float32), "y": y, "person": person, "unit": unit,
            "calib": calib, "val": val, "test": test, "people": people,
            "n_classes": int(y.max()) + 1, "sfreq": SFREQ,
            "name": name if part == "dev" else f"{name}_{part}"}


def load_dreyer_sim2():
    """Dreyer sim2 (loop C): the 14 locked people are the evaluation people."""
    import json
    import sys
    sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2")]
    from train_mixture import load_windows, masks
    d = load_windows()
    spec = REPO / "research" / "expert-portfolio" / "experiments" / "00-protocol" / "sim2_people.json"
    train, calib, hidden, people = masks(d, json.loads(spec.read_text())["people"])
    person = np.array([people.index(s) if s in people else -1 for s in d["subject"]])
    return {"X": d["X"], "y": d["y"], "person": person, "unit": d["run"].copy(),
            "calib": calib, "val": d["split"] == "val", "test": hidden,
            "pool": train, "pool_val": d["split"] == "val",
            "people": np.asarray(people), "n_classes": int(d["y"].max()) + 1,
            "sfreq": float(d["sfreq"]), "name": "dreyer_sim2"}
