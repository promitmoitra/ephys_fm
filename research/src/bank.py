"""Prediction banks for loop B (learned expert weights), Dreyer sealed-phase simulation.

A bank holds every expert's prediction for every target window, for every one of the
21 evaluation people's experts, so combiners can be fitted and scored in seconds.

    dev bank   experts trained on calibration R1-R2, predicting R3 (decisions)
    test bank  experts trained on calibration R1-R3, predicting R4-R6 (confirmation only)

Dev bank (one-time, ~1 h on 4 threads):
    pooled EEGNet on training pool + calibration R1-R2, epoch picked on the kit's val split
    control: pooled fine-tuned on all 21 people's R1-R2 (lr 1e-4, 50 epochs, last)
    experts: pooled fine-tuned per person on their R1-R2 (lr 1e-4, 50 epochs, last)
    classical (csp, ts): per person, 8-30 Hz, 0.5-4 s, fitted on R1-R2
    fp_biased: the packaged fingerprint's p(person) on R3 (it picked its epoch on R3, so
               this is optimistic; exploratory use only)

Test bank: the EEGNet experts and fingerprint are the packaged seed-0 mixture
(outputs/track2_dreyer_sim/submission, trained on R1-R3 with 8 threads), so its numbers
match the reference 0.901; classical experts fitted on R1-R3.

Every expert's output is stored as log-probabilities, shape (n_windows, K, C) for
per-person experts and (n_windows, C) for pooled / control.

Usage (worktree root, venv active):
    python research/src/bank.py dev --threads 4 --seed 0
    python research/src/bank.py test
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2")]

from train_mixture import fit, load_windows, logits, make_eegnet  # noqa: E402

OUT = REPO / "outputs" / "t2-expert-weights"
SUB = REPO / "outputs" / "track2_dreyer_sim" / "submission"
SFREQ = 120.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setting(d, n_calib_runs):
    """Masks for the evaluation people: calibration runs < n_calib_runs, target the rest
    (dev: target is run == n_calib_runs only)."""
    subj, run, split = d["subject"], d["run"], d["split"]
    is_eval = split == "test"
    people = sorted(np.unique(subj[is_eval]), key=int)
    calib = is_eval & (run < n_calib_runs)
    target = is_eval & (run == n_calib_runs) if n_calib_runs < 3 else is_eval & (run >= 3)
    return people, calib, target


def log_softmax_np(z):
    return torch.log_softmax(torch.as_tensor(z), -1).numpy()


def classical_logp(d, people, calib, target, kinds=("csp", "ts"), band=(8, 30),
                   win=(0.5, 4.0)):
    """Per-person classical experts fitted on calib, log-probabilities on target windows."""
    from scipy.signal import butter, sosfiltfilt
    import mne
    from classical_experts import classical_models
    mne.set_log_level("WARNING")
    X, y, subj = d["X"], d["y"], d["subject"]
    sos = butter(4, list(band), btype="bandpass", fs=SFREQ, output="sos")
    a, b = int(win[0] * SFREQ), int(win[1] * SFREQ)

    def filt(m):
        return sosfiltfilt(sos, X[m], axis=-1)[:, :, a:b].astype(np.float64)

    Xt = filt(target)
    makers = classical_models()
    out = {}
    for k in kinds:
        P = np.empty((len(Xt), len(people), int(y.max()) + 1))
        for j, s in enumerate(people):
            m = calib & (subj == s)
            P[:, j] = makers[k]().fit(filt(m), y[m]).predict_proba(Xt)
        out[k] = np.log(np.clip(P, 1e-7, 1.0))
    return out


def expert_logp(models, X):
    return np.stack([log_softmax_np(logits(m, X).numpy()) for m in models], 1)


def build_dev(args):
    """Dev bank for one held-out calibration run r (default R3): experts trained on the
    other two calibration runs predict run r. r = 2 is the original dev bank."""
    r = args.holdout
    tag = "" if r == 2 else f"_hold{r}"
    d = load_windows()
    X, y, subj, run, split = d["X"], d["y"], d["subject"], d["run"], d["split"]
    people, _, _ = setting(d, 2)
    is_eval = split == "test"
    calib = is_eval & (run < 3) & (run != r)
    target = is_eval & (run == r)
    n_chans, n_times = X.shape[1:]
    C = int(y.max()) + 1
    log(f"dev bank, held-out run R{r + 1}: {len(people)} people, calib {calib.sum()} "
        f"windows, target {target.sum()} windows")

    pooled_path = OUT / f"pooled_dev{tag}_seed{args.seed}.pt"
    pool = (split == "train") | calib
    val = split == "val"
    if pooled_path.exists():
        pooled = make_eegnet(n_chans, C, n_times)
        ck = torch.load(pooled_path, weights_only=True)
        pooled.load_state_dict(ck["state"])
        info = ck["info"]
        log(f"loaded {pooled_path.name} ({info})")
    else:
        log(f"pooled EEGNet: {pool.sum()} windows, {args.epochs} epochs")
        pooled, info = fit(make_eegnet(n_chans, C, n_times), X[pool], y[pool], lr=1e-3,
                           epochs=args.epochs, seed=args.seed, bs=64, X_val=X[val],
                           y_val=y[val], name="pooled", log_every=5)
        torch.save({"state": pooled.state_dict(), "info": info}, pooled_path)
        log(f"saved {pooled_path.name} ({info})")

    Xt = X[target]
    bank = {"people": np.asarray(people), "y": y[target], "subject": subj[target],
            "true_idx": np.array([people.index(s) for s in subj[target]])}
    bank["pooled"] = log_softmax_np(logits(pooled, Xt).numpy())

    log("control fine-tune (all 21 people's calibration runs)")
    control, _ = fit(copy.deepcopy(pooled), X[calib], y[calib], lr=1e-4,
                     epochs=args.ft_epochs, seed=args.seed, bs=32)
    bank["control"] = log_softmax_np(logits(control, Xt).numpy())

    experts = []
    for s in people:
        m = calib & (subj == s)
        e, _ = fit(copy.deepcopy(pooled), X[m], y[m], lr=1e-4, epochs=args.ft_epochs,
                   seed=args.seed, bs=32)
        experts.append(e)
    log(f"  {len(experts)} experts done")
    torch.save({"experts": [e.state_dict() for e in experts],
                "control": control.state_dict()},
               OUT / f"experts_dev{tag}_seed{args.seed}.pt")
    bank["eegnet"] = expert_logp(experts, Xt)

    log("classical experts")
    bank.update(classical_logp(d, people, calib, target))

    if r == 2:
        log("packaged fingerprint on R3 (biased: its epoch was picked on R3)")
        bank["fp_biased"] = fingerprint_logp(d, people, Xt)

    path = OUT / f"bank_dev{tag}_seed{args.seed}.npz"
    np.savez(path, **bank, pooled_info=json.dumps(info))
    log(f"saved {path}")
    summarize(bank)


def build_xfit(args):
    """Concatenate the three held-out-run dev banks into one cross-fitted bank."""
    parts = []
    for r in range(3):
        tag = "" if r == 2 else f"_hold{r}"
        d = np.load(OUT / f"bank_dev{tag}_seed{args.seed}.npz", allow_pickle=True)
        parts.append({k: d[k] for k in ("y", "subject", "true_idx", "pooled", "control",
                                        "eegnet", "csp", "ts")} | {"people": d["people"]})
    bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "people"}
    bank["people"] = parts[0]["people"]
    bank["fold"] = np.concatenate([np.full(len(p["y"]), r) for r, p in enumerate(parts)])
    path = OUT / f"bank_xfit_seed{args.seed}.npz"
    np.savez(path, **bank)
    log(f"saved {path} ({len(bank['y'])} windows)")
    summarize(bank)


def packaged_mixture(d, people):
    sys.path.insert(0, str(REPO / "track2"))
    from submission import build_model
    config = json.loads((SUB / "config.json").read_text())
    assert config["experts"] == [f"Dreyer2023Large/{s}" for s in people], "expert order"
    state = torch.load(SUB / "mixture.pt", map_location="cpu", weights_only=True)
    meta = {"ch_names": [str(c) for c in d["ch_names"]], "n_times": d["X"].shape[2],
            "n_classes": int(d["y"].max()) + 1, "device": "cpu"}
    return build_model(meta, config, state)


def fingerprint_logp(d, people, Xt):
    mix = packaged_mixture(d, people)
    return log_softmax_np(logits(mix.fingerprint, Xt).numpy())


def build_test(args):
    d = load_windows()
    X, y, subj = d["X"], d["y"], d["subject"]
    people, calib, target = setting(d, 3)
    Xt = X[target]
    log(f"test bank: calib {calib.sum()} (R1-R3), target {target.sum()} (R4-R6)")
    mix = packaged_mixture(d, people)
    bank = {"people": np.asarray(people), "y": y[target], "subject": subj[target],
            "true_idx": np.array([people.index(s) for s in subj[target]]),
            "eegnet": expert_logp(list(mix.experts), Xt),
            "fp": log_softmax_np(logits(mix.fingerprint, Xt).numpy())}
    bank.update(classical_logp(d, people, calib, target))
    path = OUT / "bank_test_packaged.npz"
    np.savez(path, **bank)
    log(f"saved {path} (confirmation only; do not use for decisions)")


def summarize(bank):
    """Oracle-ID balanced accuracy of each stored expert (sanity check)."""
    from sklearn.metrics import balanced_accuracy_score
    n = np.arange(len(bank["y"]))
    for k, v in bank.items():
        if not isinstance(v, np.ndarray) or v.dtype.kind != "f" or k.startswith("fp"):
            continue
        p = v[n, bank["true_idx"]] if v.ndim == 3 else v
        log(f"  {k:8s} oracle bal-acc {balanced_accuracy_score(bank['y'], p.argmax(1)):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank", choices=["dev", "test", "xfit"])
    ap.add_argument("--holdout", type=int, default=2, help="dev: held-out calib run (0-2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--ft-epochs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="1 epoch each, writes to smoke/")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.smoke:
        global OUT
        OUT = OUT / "smoke"
        args.epochs = args.ft_epochs = 1
    OUT.mkdir(parents=True, exist_ok=True)
    {"dev": build_dev, "test": build_test, "xfit": build_xfit}[args.bank](args)


if __name__ == "__main__":
    main()
