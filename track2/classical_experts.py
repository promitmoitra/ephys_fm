"""Within-person MI filters (CSP, Riemannian) as additional Track 2 experts.

The Dreyer analyses showed a real motor-imagery signature (contralateral
mu/beta desynchronization in 70-80% of people) that transfers poorly across
people, which is where per-person models should help. This fits two standard
within-person MI decoders on each evaluation person's calibration runs and
evaluates them as experts in the fingerprint mixture:

  csp  CSP (6 components, Ledoit-Wolf) + shrinkage LDA
  ts   covariance (OAS) -> Riemannian tangent space -> logistic regression

Both on 8-30 Hz band-passed signals, window 0.5-4 s after the cue (the
imagery period, skipping the cue-evoked response).

Setting: the sealed-phase simulation of train_mixture.py (Dreyer test people
61-81; calibration R1-R3, hidden R4-R6; the kit's 27-channel input). The
EEGNet experts and the fingerprint are loaded from the packaged seed-0
mixture (outputs/track2_dreyer_sim/submission), so the results are directly
comparable with it. Per person, the combined expert averages the class
probabilities of its members.

Usage (repo root, venv active, after train_mixture.py):
    python track2/classical_experts.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(HERE)]

from submission import build_model  # noqa: E402

CACHE = REPO / "data" / "experiments" / "dreyer_windows.npz"
SUB = REPO / "outputs" / "track2_dreyer_sim" / "submission"
RESULTS = HERE / "results"
SFREQ = 120.0
WIN = (int(0.5 * SFREQ), 480)
N_CALIB_RUNS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bal(y, p):
    return float(balanced_accuracy_score(y, p))


def classical_models():
    from mne.decoding import CSP
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    return {
        "csp": lambda: make_pipeline(
            CSP(n_components=6, reg="ledoit_wolf", log=True),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "ts": lambda: make_pipeline(
            Covariances("oas"), TangentSpace(metric="riemann"),
            LogisticRegression(C=1.0, max_iter=3000)),
    }


def main():
    t0 = time.time()
    d = np.load(CACHE, allow_pickle=True)
    X, y, subj, run, split = d["X"], d["y"], d["subject"], d["run"], d["split"]
    ch_names = [str(c) for c in d["ch_names"]]
    is_eval = split == "test"
    people = sorted(np.unique(subj[is_eval]), key=int)
    K = len(people)
    cal = is_eval & (run < N_CALIB_RUNS)
    hid = is_eval & (run >= N_CALIB_RUNS)
    Xh, yh, sh = X[hid], y[hid], subj[hid]
    true_idx = np.array([people.index(s) for s in sh])
    n = np.arange(len(Xh))

    # EEGNet experts + fingerprint from the packaged seed-0 mixture
    config = json.loads((SUB / "config.json").read_text())
    assert config["experts"] == [f"Dreyer2023Large/{s}" for s in people], "expert order"
    state = torch.load(SUB / "mixture.pt", map_location="cpu", weights_only=True)
    meta = {"ch_names": ch_names, "n_times": X.shape[2],
            "n_classes": int(y.max()) + 1, "device": "cpu"}
    mix = build_model(meta, config, state)
    with torch.inference_mode():
        Xt = torch.from_numpy(Xh)
        p_subj = torch.softmax(mix.fingerprint(Xt), 1).numpy()                   # (n, K)
        probs = {"eegnet": np.stack([torch.softmax(e(Xt), 1).numpy()
                                     for e in mix.experts], 1)}                  # (n, K, C)
    log(f"loaded mixture; fingerprint on hidden runs {bal(true_idx, p_subj.argmax(1)):.3f}")

    # Per-person classical experts on 8-30 Hz, 0.5-4 s
    sos = butter(4, [8, 30], btype="bandpass", fs=SFREQ, output="sos")
    Xf = sosfiltfilt(sos, X[is_eval], axis=-1)[:, :, WIN[0]:WIN[1]].astype(np.float64)
    f_idx = np.where(is_eval)[0]
    pos = {g: i for i, g in enumerate(f_idx)}
    Xf_h = Xf[[pos[i] for i in np.where(hid)[0]]]
    for name, make in classical_models().items():
        P = np.empty((len(Xh), K, 2))
        for j, s in enumerate(people):
            m = cal & (subj == s)
            clf = make().fit(Xf[[pos[i] for i in np.where(m)[0]]], y[m])
            P[:, j] = clf.predict_proba(Xf_h)
        probs[name] = P
        log(f"fitted {name} for {K} people")

    combos = {"eegnet": ["eegnet"], "csp": ["csp"], "ts": ["ts"],
              "eegnet+csp": ["eegnet", "csp"], "eegnet+ts": ["eegnet", "ts"],
              "csp+ts": ["csp", "ts"], "eegnet+csp+ts": ["eegnet", "csp", "ts"]}
    scores = {}
    for cname, members in combos.items():
        P = np.mean([probs[m] for m in members], 0)                             # (n, K, C)
        preds = {"oracle": P[n, true_idx].argmax(1),
                 "soft": (p_subj[:, :, None] * P).sum(1).argmax(1),
                 "hard": P[n, p_subj.argmax(1)].argmax(1)}
        scores[cname] = {}
        for rname, p in preds.items():
            per = {s: bal(yh[sh == s], p[sh == s]) for s in people}
            scores[cname][rname] = {"bal_acc": bal(yh, p),
                                    "mean_over_people": float(np.mean(list(per.values()))),
                                    "per_person": per}
    # paired per-person comparison vs the EEGNet-only mixture (soft)
    base = np.array(list(scores["eegnet"]["soft"]["per_person"].values()))
    paired = {}
    for cname in combos:
        if cname == "eegnet":
            continue
        v = np.array(list(scores[cname]["soft"]["per_person"].values()))
        paired[cname] = {"mean_diff": float((v - base).mean()),
                         "better": int((v > base).sum()), "worse": int((v < base).sum())}
    out = {"window_s": [WIN[0] / SFREQ, WIN[1] / SFREQ], "band_hz": [8, 30],
           "scores": scores, "paired_soft_vs_eegnet": paired,
           "reference": {"pooled": 0.873, "control": 0.890},
           "runtime_s": round(time.time() - t0, 1)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "classical_experts_seed0.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "classical_experts_seed0.md").write_text(render(out, K))
    print(render(out, K), flush=True)


def render(o, K):
    lines = ["# Track 2 prototype: per-person CSP / Riemannian experts (Dreyer simulation)\n",
             f"{K} evaluation people; experts fitted on calibration runs R1–R3, scored on "
             "hidden runs R4–R6. `csp` = CSP(6) + shrinkage LDA; `ts` = OAS covariance → "
             f"tangent space → logistic regression; both {o['band_hz'][0]}–{o['band_hz'][1]} Hz, "
             f"{o['window_s'][0]}–{o['window_s'][1]} s. `eegnet` = the fine-tuned EEGNet experts of "
             "the packaged seed-0 mixture; combinations average class probabilities per person. "
             "Routing by the same fingerprint. Chance 0.5. Reference (seed 0): pooled EEGNet "
             f"{o['reference']['pooled']}, epoch-matched control {o['reference']['control']}.\n",
             "| Experts | Oracle ID | Soft routing | Hard routing | Soft vs `eegnet`: Δ, people better / worse |",
             "|---|---|---|---|---|"]
    for c, r in o["scores"].items():
        pr = o["paired_soft_vs_eegnet"].get(c)
        cmp_ = "—" if pr is None else f"{pr['mean_diff']:+.3f}, {pr['better']} / {pr['worse']}"
        lines.append(f"| {c} | {r['oracle']['bal_acc']:.3f} | {r['soft']['bal_acc']:.3f} | "
                     f"{r['hard']['bal_acc']:.3f} | {cmp_} |")
    lines.append(f"\nRuntime {o['runtime_s']} s.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
