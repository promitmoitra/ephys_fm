"""Cross-session EEG fingerprinting on BNCI2014_001 (Tangermann 2012, BCI IV-2a).

Question: from a single 4-s motor-imagery window, can we tell *who* it came
from on a different day? If yes, a Track 2 model could infer the subject from
the signal and route to a subject-specific decoder (predict(X) gets no IDs).

Data: 9 subjects x 2 sessions (different days) x 6 runs x 48 trials, 22 EEG,
4 classes (left hand / right hand / feet / tongue).

Preprocessing replicates the competition pipeline (neuralbench EegExtractor
defaults): EEG picks -> notch 50/60 Hz + harmonics -> band-pass 0.1-75 Hz ->
resample 120 Hz -> RobustScaler per channel *per recording (run)* -> clamp 20.
Windows: 4 s from the cue (MOABB interval [2, 6] s after trial start), i.e.
480 samples, the kit's 4-s motor-imagery window.

Experiments (all per-window, no aggregation across windows, as at inference):
  A. Subject ID within session 1, leave-one-run-out  (no day-to-day drift)
  B. Subject ID train session 1 -> test session 2    (the Track 2 shift)
  C. Control: B with shuffled training subject labels (should be ~chance)
  D. Does routing help? 4-class MI decoding, session 1 -> session 2:
     pooled model vs per-subject models chosen by oracle ID, predicted ID
     (hard routing) and fingerprint probabilities (soft routing).

Usage (from repo root, venv active):
    python experiments/fingerprint_tangermann/run.py [--no-cnn]
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = Path(__file__).resolve().parent / "results"
CACHE = DATA / "experiments" / "tangermann_windows.npz"

# Keep MOABB's downloads inside our data/ folder. Env vars only: with them
# unset, MOABB writes MNE_DATA=~/mne_data into ~/.mne/mne-python.json. MOABB
# >= 1.7 tries the NEMAR mirror (under $MNE_DATA/NEMAR) before BNCI's server.
(DATA / "mne_data").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MNE_DATA", str(DATA / "mne_data"))
os.environ.setdefault("MNE_DATASETS_BNCI_PATH", str(DATA / "mne_data"))
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

SFREQ = 120.0
WIN_S = (2.0, 6.0)  # relative to trial start: cue -> cue + 4 s
N_TIMES = int(round((WIN_S[1] - WIN_S[0]) * SFREQ))
CLAMP = 20.0
SEED = 42


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _notch_freqs(sfreq, bases=(50.0, 60.0), fmax=300.0):
    nyq = sfreq / 2
    return sorted({k * b for b in bases for k in range(1, int(fmax // b) + 1)
                   if k * b < nyq})


def preprocess_run(raw):
    """One MOABB run -> (windows (n, C, T), labels (n,)) on the kit's pipeline."""
    import mne
    from sklearn.preprocessing import RobustScaler

    raw = raw.copy().load_data()
    events = mne.find_events(raw, shortest_event=0, verbose=False)
    onsets_s = (events[:, 0] - raw.first_samp) / raw.info["sfreq"]
    labels = events[:, 2]

    raw.pick("eeg")
    raw.notch_filter(_notch_freqs(raw.info["sfreq"]), verbose=False)
    raw.filter(0.1, min(75.0, raw.info["sfreq"] / 2 - 1), verbose=False)
    raw.resample(SFREQ, verbose=False)
    data = RobustScaler().fit_transform(raw.get_data().T).T
    data = np.clip(data, -CLAMP, CLAMP).astype(np.float32)

    X, y = [], []
    for t, lab in zip(onsets_s, labels):
        start = int(round((t + WIN_S[0]) * SFREQ))
        if start + N_TIMES <= data.shape[1]:
            X.append(data[:, start:start + N_TIMES])
            y.append(lab)
    return np.stack(X), np.asarray(y), raw.ch_names


def load_windows():
    """All windows with subject / session / run metadata (cached)."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    from moabb.datasets import BNCI2014_001

    ds = BNCI2014_001()
    parts = {k: [] for k in ("X", "task", "subject", "session", "run")}
    for subj in ds.subject_list:
        sessions = ds.get_data([subj])[subj]
        for s_idx, s_name in enumerate(sorted(sessions)):
            for r_idx, r_name in enumerate(sorted(sessions[s_name])):
                X, y, ch_names = preprocess_run(sessions[s_name][r_name])
                n = len(y)
                parts["X"].append(X)
                parts["task"].append(y)
                parts["subject"].append(np.full(n, subj))
                parts["session"].append(np.full(n, s_idx))
                parts["run"].append(np.full(n, r_idx))
        print(f"  subject {subj}: sessions {sorted(sessions)}", flush=True)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["ch_names"] = np.asarray(ch_names)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def make_model(kind):
    """Fingerprint / MI classifiers on fixed hyper-parameters (no tuning on
    session 2: it plays the hidden test set)."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    lr = dict(max_iter=3000, random_state=SEED)
    if kind == "riemann":
        # Spatial covariance -> tangent space at the training mean.
        return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                             StandardScaler(), LogisticRegression(C=1.0, **lr))
    if kind == "psd":
        # Log Welch spectrum 1-45 Hz per channel (0.5 Hz bins).
        return make_pipeline(FunctionTransformer(log_psd), StandardScaler(),
                             LogisticRegression(C=0.05, **lr))
    raise ValueError(kind)


def log_psd(X):
    from scipy.signal import welch
    f, P = welch(X, fs=SFREQ, nperseg=int(2 * SFREQ), axis=-1)
    keep = (f >= 1) & (f <= 45)
    return np.log10(P[..., keep] + 1e-12).reshape(len(X), -1)


def train_eegnet(X_tr, y_tr, X_va, y_va, n_classes, epochs=60):
    """EEGNet (braindecode) for subject ID; best epoch picked on validation."""
    import torch
    from braindecode.models import EEGNet

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = EEGNet(n_chans=X_tr.shape[1], n_outputs=n_classes,
                   n_times=X_tr.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt)).split(64):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        acc = (predict_torch(model, X_va).argmax(1) == y_va).mean()
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def predict_torch(model, X):
    import torch
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(torch.from_numpy(X)), 1).numpy()


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def bal_acc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p))


def per_class_recall(y, p):
    return {int(c): float((p[y == c] == c).mean()) for c in np.unique(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cnn", action="store_true", help="skip EEGNet")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading windows ...", flush=True)
    d = load_windows()
    X, subj, sess, run, task = (d["X"], d["subject"], d["session"], d["run"],
                                d["task"])
    subjects = np.unique(subj)
    print(f"  X {X.shape}, subjects {subjects.tolist()}, "
          f"windows per session {np.bincount(sess).tolist()}", flush=True)
    assert X.shape[1:] == (22, N_TIMES), X.shape
    s1, s2 = sess == 0, sess == 1
    results = {"n_subjects": int(len(subjects)),
               "chance": 1 / len(subjects),
               "n_windows": {"session1": int(s1.sum()), "session2": int(s2.sum())}}

    # A. within session 1, leave-one-run-out
    print("A. within-session subject ID (leave-one-run-out) ...", flush=True)
    results["A_within_session"] = {}
    for kind in ("riemann", "psd"):
        accs = []
        for r in np.unique(run[s1]):
            tr, te = s1 & (run != r), s1 & (run == r)
            m = make_model(kind).fit(X[tr], subj[tr])
            accs.append(bal_acc(subj[te], m.predict(X[te])))
        results["A_within_session"][kind] = float(np.mean(accs))
        print(f"   {kind}: {np.mean(accs):.3f}", flush=True)

    # B. cross-session: train session 1 -> test session 2
    print("B. cross-session subject ID (session 1 -> 2) ...", flush=True)
    results["B_cross_session"] = {}
    fp_proba = {}
    for kind in ("riemann", "psd"):
        m = make_model(kind).fit(X[s1], subj[s1])
        pred = m.predict(X[s2])
        fp_proba[kind] = m.predict_proba(X[s2])
        results["B_cross_session"][kind] = {
            "bal_acc": bal_acc(subj[s2], pred),
            "per_subject_recall": per_class_recall(subj[s2], pred)}
        print(f"   {kind}: {results['B_cross_session'][kind]['bal_acc']:.3f}",
              flush=True)
    if not args.no_cnn:
        # labels 0..K-1 for torch; last run of session 1 is validation
        lab = np.searchsorted(subjects, subj)
        tr, va = s1 & (run != 5), s1 & (run == 5)
        model = train_eegnet(X[tr], lab[tr], X[va], lab[va], len(subjects))
        proba = predict_torch(model, X[s2])
        pred = subjects[proba.argmax(1)]
        fp_proba["eegnet"] = proba
        results["B_cross_session"]["eegnet"] = {
            "bal_acc": bal_acc(subj[s2], pred),
            "per_subject_recall": per_class_recall(subj[s2], pred)}
        print(f"   eegnet: {results['B_cross_session']['eegnet']['bal_acc']:.3f}",
              flush=True)

    # C. shuffled-label control
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(subj[s1])
    m = make_model("riemann").fit(X[s1], shuffled)
    results["C_shuffled_control"] = bal_acc(subj[s2], m.predict(X[s2]))
    print(f"C. shuffled-label control (riemann): "
          f"{results['C_shuffled_control']:.3f}", flush=True)

    # D. does routing by inferred identity help MI decoding?
    print("D. 4-class MI decoding, session 1 -> 2 ...", flush=True)
    best_fp = max(results["B_cross_session"],
                  key=lambda k: results["B_cross_session"][k]["bal_acc"])
    P = fp_proba[best_fp]  # (n_s2, K), columns ordered as `subjects`
    pooled = make_model("riemann").fit(X[s1], task[s1])
    per_subj = {s: make_model("riemann").fit(X[s1 & (subj == s)],
                                            task[s1 & (subj == s)])
                for s in subjects}
    Xte, yte, ste = X[s2], task[s2], subj[s2]
    classes = pooled.classes_
    per_proba = np.stack([per_subj[s].predict_proba(Xte) for s in subjects], 1)
    oracle = per_proba[np.arange(len(Xte)), np.searchsorted(subjects, ste)]
    hard = per_proba[np.arange(len(Xte)), P.argmax(1)]
    soft = (P[:, :, None] * per_proba).sum(1)
    D = {"fingerprint_model": best_fp,
         "pooled": pooled.predict(Xte),
         "oracle_subject": classes[oracle.argmax(1)],
         "routed_hard": classes[hard.argmax(1)],
         "routed_soft": classes[soft.argmax(1)]}
    results["D_routing"] = {"fingerprint_model": best_fp}
    for name in ("pooled", "oracle_subject", "routed_hard", "routed_soft"):
        overall = bal_acc(yte, D[name])
        per_s = {int(s): bal_acc(yte[ste == s], D[name][ste == s])
                 for s in subjects}
        results["D_routing"][name] = {"bal_acc": overall,
                                      "mean_over_subjects": float(np.mean(list(per_s.values()))),
                                      "per_subject": per_s}
        print(f"   {name}: {overall:.3f} (mean over subjects "
              f"{np.mean(list(per_s.values())):.3f})", flush=True)

    results["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "results.md").write_text(render_markdown(results))
    print(f"Done in {results['runtime_s']} s -> {OUT}/results.md", flush=True)


def render_markdown(r):
    lines = [f"# Cross-session fingerprinting: BNCI2014_001\n",
             f"{r['n_subjects']} subjects, chance = {r['chance']:.3f}. "
             f"Windows: {r['n_windows']['session1']} (session 1), "
             f"{r['n_windows']['session2']} (session 2). "
             "Metric: balanced accuracy, per 4-s window.\n",
             "## Subject identification\n",
             "| Model | A. within session 1 (LORO) | B. session 1 → 2 |",
             "|---|---|---|"]
    for k, v in r["B_cross_session"].items():
        a = r["A_within_session"].get(k)
        lines.append(f"| {k} | {'—' if a is None else f'{a:.3f}'} | {v['bal_acc']:.3f} |")
    lines.append(f"| riemann, shuffled labels (control) | | "
                 f"{r['C_shuffled_control']:.3f} |\n")
    best = r["D_routing"]["fingerprint_model"]
    lines += ["### Per-subject recall, session 1 → 2\n",
              "| Subject | " + " | ".join(r["B_cross_session"]) + " |",
              "|---|" + "---|" * len(r["B_cross_session"])]
    for s in r["B_cross_session"][best]["per_subject_recall"]:
        lines.append(f"| {s} | " + " | ".join(
            f"{v['per_subject_recall'][s]:.2f}"
            for v in r["B_cross_session"].values()) + " |")
    lines += ["\n## Does routing by inferred identity help MI decoding?\n",
              f"4-class MI, session 1 → 2, Riemannian decoders; routing uses "
              f"the `{best}` fingerprint. Chance = 0.25.\n",
              "| Strategy | Balanced acc (pooled windows) | Mean over subjects |",
              "|---|---|---|"]
    for name in ("pooled", "oracle_subject", "routed_hard", "routed_soft"):
        v = r["D_routing"][name]
        lines.append(f"| {name} | {v['bal_acc']:.3f} | {v['mean_over_subjects']:.3f} |")
    lines.append(f"\nRuntime: {r['runtime_s']} s.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
