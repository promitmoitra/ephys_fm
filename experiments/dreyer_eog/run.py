"""What is the cue-locked Dreyer signal: eye movements, overt movement, or EEG?

experiments/dreyer_confound showed that most of the Dreyer warm-up decoding
(pooled EEGNet 0.80 cross-subject) lives in the first 1.25 s after the cue,
while the arrow is on screen. The kit keeps EEG only, but Dreyer also recorded
3 EOG channels (EOG1 below, EOG2 above, EOG3 beside one eye: horizontal) and
2 wrist EMG channels (EMGg left, EMGd right). This decodes left/right hand
from each, on the kit's warm-up split (train subjects -> val subjects ->
unseen test subjects 61-81):

  1. EEGNet (30 epochs, best on val) per signal and window;
  2. time-resolved logistic regression on sliding 250 ms windows (125 ms
     steps): when does each signal become decodable?

Signals, re-extracted from the raw recordings with the kit's EEG pipeline
(notch 50/60 Hz + harmonics, 0.1-75 Hz, 120 Hz, RobustScaler per recording,
clamp 20), cue-locked 0-4 s (480 samples):
  eeg       27 EEG channels (checked against the kit's own windows)
  eog       EOG1-3, same pipeline
  emg       wrist EMG envelope: 20 Hz high-pass, rectify, 8 Hz low-pass
  eeg_reg   EEG with the 3 EOG channels regressed out per recording (least
            squares on the continuous filtered data). Conservative: it also
            removes any brain signal the EOG electrodes pick up.

Usage (repo root, venv active; Dreyer prepared, and
track2/train_mixture.py's window cache present for the split):
    python experiments/dreyer_eog/run.py [--epochs 30] [--seed 0]
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[2]
DL = REPO / "data" / "neural_compet" / "moabb" / "Dreyer2023Large" / "download"
KIT_CACHE = REPO / "data" / "experiments" / "dreyer_windows.npz"
CACHE = REPO / "data" / "experiments" / "dreyer_eog_windows.npz"
OUT = Path(__file__).resolve().parent / "results"
SFREQ = 120.0
N_TIMES = 480
CUE_END = int(1.25 * SFREQ)
CLAMP = 20.0
LABELS = {"left_hand": 0, "right_hand": 1}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bal_acc(y, p):
    return float(balanced_accuracy_score(y, p))


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _notch_freqs(sfreq, bases=(50.0, 60.0), fmax=300.0):
    return sorted({k * b for b in bases for k in range(1, int(fmax // b) + 1)
                   if k * b < sfreq / 2})


def _scale(data):
    from sklearn.preprocessing import RobustScaler
    return np.clip(RobustScaler().fit_transform(data.T).T, -CLAMP, CLAMP)


def extract_subject(subject):
    """All cue-locked windows of one subject: eeg, eeg_reg, eog, emg."""
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    import warnings
    warnings.filterwarnings("ignore")
    from neuralfetch.download import temp_mne_data
    with temp_mne_data(DL, clear_dataset_configs=True):
        from moabb.datasets import Dreyer2023
        sessions = Dreyer2023(return_all_modalities=True).get_data([subject])[subject]
    out = {k: [] for k in ("eeg", "eeg_reg", "eog", "emg", "y", "run")}
    for sess in sorted(sessions):
        for r_name in sorted(sessions[sess]):
            raw = sessions[sess][r_name].copy().load_data()
            sf = raw.info["sfreq"]
            onsets = [(a["onset"] - raw.first_time, LABELS[a["description"]])
                      for a in raw.annotations if a["description"] in LABELS]
            # picks="all": MNE's filters otherwise skip EOG/EMG channels
            ex = raw.copy().pick(["eeg", "eog"])
            ex.notch_filter(_notch_freqs(sf), picks="all", verbose=False)
            ex.filter(0.1, 75.0, picks="all", verbose=False)
            ex.resample(SFREQ, verbose=False)
            eeg = ex.copy().pick("eeg").get_data()
            eog = ex.copy().pick("eog").get_data()
            B, *_ = np.linalg.lstsq(eog.T, eeg.T, rcond=None)
            eeg_reg = eeg - (eog.T @ B).T
            em = raw.copy().pick("emg")
            em.notch_filter(_notch_freqs(sf), picks="all", verbose=False)
            em.filter(20.0, None, picks="all", verbose=False)
            em.apply_function(np.abs, picks="all")
            em.filter(None, 8.0, picks="all", verbose=False)
            em.resample(SFREQ, verbose=False)
            sigs = {"eeg": _scale(eeg), "eeg_reg": _scale(eeg_reg),
                    "eog": _scale(eog), "emg": _scale(em.get_data())}
            for t, lab in onsets:
                s = int(round(t * SFREQ))
                if s + N_TIMES > sigs["eeg"].shape[1]:
                    continue
                for k, v in sigs.items():
                    out[k].append(v[:, s:s + N_TIMES].astype(np.float32))
                out["y"].append(lab)
                out["run"].append(int(r_name[0]))
    ch = {"eeg": raw.copy().pick("eeg").ch_names,
          "eog": raw.copy().pick("eog").ch_names,
          "emg": raw.copy().pick("emg").ch_names}
    return subject, {k: np.stack(v) if k not in ("y", "run") else np.asarray(v)
                     for k, v in out.items()}, ch


def load_windows(n_jobs):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    from joblib import Parallel, delayed
    kit = np.load(KIT_CACHE, allow_pickle=True)
    split_of = dict(zip(kit["subject"].astype(str), kit["split"].astype(str)))
    subjects = sorted({int(s) for s in split_of})
    log(f"extracting {len(subjects)} subjects with {n_jobs} workers")
    res = Parallel(n_jobs=n_jobs)(delayed(extract_subject)(s) for s in subjects)
    parts = {k: [] for k in ("eeg", "eeg_reg", "eog", "emg", "y", "run", "subject", "split")}
    for s, arrs, ch in res:
        n = len(arrs["y"])
        for k in ("eeg", "eeg_reg", "eog", "emg", "y", "run"):
            parts[k].append(arrs[k])
        parts["subject"].append(np.full(n, str(s)))
        parts["split"].append(np.full(n, split_of[str(s)]))
    d = {k: np.concatenate(v) for k, v in parts.items()}
    for k, v in ch.items():
        d[f"ch_{k}"] = np.asarray(v)
    np.savez(CACHE, **d)
    return d


def check_against_kit(d):
    """Correlate our EEG windows with the kit's for matching windows."""
    kit = np.load(KIT_CACHE, allow_pickle=True)
    kit_ch = [str(c) for c in kit["ch_names"]]
    ours_ch = [str(c) for c in d["ch_eeg"]]
    order = [ours_ch.index(c) for c in kit_ch]
    checks = []
    for s in ("1", "61", "85"):
        for r in (0, 5):
            km = np.where((kit["subject"] == s) & (kit["run"] == r))[0]
            om = np.where((d["subject"] == s) & (d["run"] == r))[0]
            n = min(len(km), len(om))
            a = kit["X"][km[:n]].ravel()
            b = d["eeg"][om[:n]][:, order].ravel()
            checks.append((s, r, len(km), len(om), float(np.corrcoef(a, b)[0, 1])))
    return checks


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

def logits(model, X, bs=512):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(torch.from_numpy(np.ascontiguousarray(X[i:i + bs])))
                          for i in range(0, len(X), bs)])


def eegnet(X_tr, y_tr, X_va, y_va, X_te, seed, epochs):
    from braindecode.models import EEGNet
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = EEGNet(n_chans=X_tr.shape[1], n_outputs=2, n_times=X_tr.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(np.ascontiguousarray(X_tr)), torch.from_numpy(y_tr).long()
    best, state, best_ep = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(64):
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        v = bal_acc(y_va, logits(model, X_va).argmax(1).numpy())
        if v > best:
            best, best_ep, state = v, ep + 1, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return logits(model, X_te).argmax(1).numpy(), best, best_ep


def time_resolved(sig, y, tr, te, win=30, step=15, bin_=5):
    """Logistic regression per sliding window (features: bin means)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = []
    for s in range(0, N_TIMES - win + 1, step):
        seg = sig[:, :, s:s + win]
        feat = seg.reshape(len(seg), seg.shape[1], win // bin_, bin_).mean(-1)
        feat = feat.reshape(len(seg), -1)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(feat[tr], y[tr])
        out.append({"t_center_s": round((s + win / 2) / SFREQ, 3),
                    "bal_acc": bal_acc(y[te], clf.predict(feat[te]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=6, help="extraction workers")
    ap.add_argument("--no-time-resolved", action="store_true",
                    help="skip the (seed-independent) time-resolved decoding")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = load_windows(args.jobs)
    y, split = d["y"], d["split"]
    tr, va, te = split == "train", split == "val", split == "test"
    log(f"windows: train {tr.sum()}, val {va.sum()}, test {te.sum()}; "
        f"eog {list(d['ch_eog'])}, emg {list(d['ch_emg'])}")
    checks = check_against_kit(d)
    for c in checks:
        log(f"  kit check subject {c[0]} run {c[1]}: kit {c[2]} / ours {c[3]} windows, r = {c[4]:.4f}")

    eog_names = [str(c) for c in d["ch_eog"]]
    heog = [eog_names.index("EOG3")]
    full, cue = slice(0, N_TIMES), slice(0, CUE_END)
    conds = {
        "eeg_full": ("eeg", None, full), "eeg_cue": ("eeg", None, cue),
        "eog_full": ("eog", None, full), "eog_cue": ("eog", None, cue),
        "heog_cue": ("eog", heog, cue),
        "emg_full": ("emg", None, full), "emg_cue": ("emg", None, cue),
        "eeg_reg_full": ("eeg_reg", None, full), "eeg_reg_cue": ("eeg_reg", None, cue),
    }
    results = {}
    for name, (sig, chs, t) in conds.items():
        A = d[sig] if chs is None else d[sig][:, chs]
        A = A[:, :, t]
        t0 = time.time()
        pred, val, ep = eegnet(A[tr], y[tr], A[va], y[va], A[te], args.seed, args.epochs)
        rt = d["run"][te]
        results[name] = {"test": bal_acc(y[te], pred), "val": val, "best_epoch": ep,
                         "acquisition_R1_R2": bal_acc(y[te][rt < 2], pred[rt < 2]),
                         "online_R3_R6": bal_acc(y[te][rt >= 2], pred[rt >= 2]),
                         "shape": list(A.shape[1:])}
        log(f"RESULT {name}: test {results[name]['test']:.3f} (val {val:.3f} @ {ep}, "
            f"{time.time() - t0:.0f} s)")

    tres = {}
    if not args.no_time_resolved:
        log("time-resolved decoding")
        tres = {sig: time_resolved(d[sig], y, tr, te)
                for sig in ("eeg", "eog", "emg", "eeg_reg")}
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"args": vars(args), "kit_check": checks, "results": results,
           "time_resolved": tres}
    (OUT / f"seed{args.seed}.json").write_text(json.dumps(out, indent=2))
    (OUT / f"seed{args.seed}.md").write_text(render(out))
    print(render(out), flush=True)


def render(o):
    lines = ["# Dreyer 2023: EOG, EMG and EOG-regressed EEG decoding\n",
             "Left vs right hand motor imagery; EEGNet trained on the kit's train "
             f"subjects, best of {o['args']['epochs']} epochs on val subjects, tested on "
             "unseen test subjects 61–81. Chance 0.5. Window from the cue: arrow "
             "0–1.25 s (`cue`), full 0–4 s (`full`). Seed "
             f"{o['args']['seed']}.\n",
             "| Signal / window | Channels × samples | Val bal. acc | Test bal. acc | "
             "Acquisition R1–R2 | Online R3–R6 |",
             "|---|---|---|---|---|---|"]
    for k, r in o["results"].items():
        lines.append(f"| {k} | {r['shape'][0]} × {r['shape'][1]} | {r['val']:.3f} | "
                     f"{r['test']:.3f} | {r['acquisition_R1_R2']:.3f} | "
                     f"{r['online_R3_R6']:.3f} |")
    tr = o["time_resolved"]
    if tr:
        sigs = list(tr)
        lines += ["", "## Time-resolved decoding (logistic regression, 250 ms windows)\n",
                  "| Window centre (s) | " + " | ".join(sigs) + " |",
                  "|---|" + "---|" * len(sigs)]
        for i, row in enumerate(tr[sigs[0]]):
            lines.append(f"| {row['t_center_s']:.3f} | " + " | ".join(
                f"{tr[s][i]['bal_acc']:.3f}" for s in sigs) + " |")
    lines += ["", "Kit check (our EEG windows vs the kit's, correlation): " + ", ".join(
        f"s{c[0]} r{c[1]}: {c[4]:.4f}" for c in o["kit_check"])]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
