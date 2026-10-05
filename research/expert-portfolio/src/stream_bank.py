"""Loop C prediction banks for the new streams (REVE probe, ShallowFBCSPNet experts).

Window order and masks match loop B's banks exactly (asserted against
outputs/t2-expert-weights/bank_xfit_seed0.npz and bank_test_packaged.npz).

    python research/expert-portfolio/src/stream_bank.py embed --dataset dreyer
    python research/expert-portfolio/src/stream_bank.py dreyer --stream reve --bank xfit
    python research/expert-portfolio/src/stream_bank.py dreyer --stream shallow --bank xfit --seed 0
    python research/expert-portfolio/src/stream_bank.py dreyer --stream shallow --bank test --seed 1
    python research/expert-portfolio/src/stream_bank.py bnci --stream shallow --seed 3
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[2]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2"),
                str(REPO / "experiments" / "fingerprint_tangermann")]
OUT = REPO / "outputs" / "t2-portfolio"
LOOPB = REPO / "outputs" / "t2-expert-weights"
POS_FILE = OUT / "reve_positions" / "reve_positions.json"
EARLY_S = 1.25
LAM_POOL = 1e-3
LAM_PERSON_GRID = (0.01, 0.1, 1.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def masked(X, sfreq, part):
    """part='early': zero samples ≥ 1.25 s; 'late': zero samples < 1.25 s."""
    cut = int(round(EARLY_S * sfreq))
    Y = X.copy()
    if part == "early":
        Y[..., cut:] = 0
    else:
        Y[..., :cut] = 0
    return Y


def dreyer():
    from train_mixture import load_windows
    d = load_windows()
    people = sorted(np.unique(d["subject"][d["split"] == "test"]), key=int)
    lab = np.array([people.index(s) if s in people else -1 for s in d["subject"]])
    return d, people, lab


def folds(d, bank):
    """[(calib_mask, target_mask)] in loop B's order."""
    ev = d["split"] == "test"
    if bank == "xfit":
        return [(ev & (d["run"] < 3) & (d["run"] != r), ev & (d["run"] == r)) for r in range(3)]
    return [(ev & (d["run"] < 3), ev & (d["run"] >= 3))]


def check_alignment(y, subject, bank):
    ref = np.load(LOOPB / ("bank_xfit_seed0.npz" if bank == "xfit" else "bank_test_packaged.npz"),
                  allow_pickle=True)
    assert (ref["y"] == y).all() and (ref["subject"] == subject).all(), "window order differs"


# ------------------------------------------------------------------ REVE

def cmd_embed(args):
    from reve_parts import embed, positions, resample_matrix
    from submission import load_reve_encoder
    enc = load_reve_encoder(POS_FILE.parent)
    if args.dataset == "dreyer":
        d, _, _ = dreyer()
        X, chs, sf = d["X"], [str(c) for c in d["ch_names"]], float(d["sfreq"])
        sub = np.where(d["split"] == "test")[0]
    else:
        from run import load_windows
        d = load_windows()
        X, chs, sf = d["X"], [str(c) for c in d["ch_names"]], 120.0
        sub = np.arange(len(X))
    R, pos = resample_matrix(X.shape[-1], sf), positions(chs, POS_FILE)
    out = {"eval_idx": sub}
    log(f"embedding {len(X)} windows")
    out["full"] = embed(enc, X, R, pos)
    for part in ("early", "late"):
        out[part] = embed(enc, masked(X[sub], sf, part), R, pos)
    np.savez(OUT / f"reve_emb_{args.dataset}.npz", **out)
    log("saved")


def reve_predict(Z_tr, y_tr, Z_cal, y_cal, p_cal, K, C, lam_person, Z_tgt_list):
    from reve_parts import fit_head, fit_person_heads
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-6
    W0, b0 = fit_head((Z_tr - mu) / sd, y_tr, C, LAM_POOL)
    W, b = fit_person_heads((Z_cal - mu) / sd, y_cal, p_cal, K, C, W0, b0, lam_person)
    outs = []
    for Z in Z_tgt_list:
        logits = np.einsum("nd,kcd->nkc", (Z - mu) / sd, W) + b
        outs.append(torch.log_softmax(torch.as_tensor(logits), -1).numpy())
    return outs, (mu, sd, W0, b0, W, b)


def cmd_dreyer_reve(args):
    d, people, lab = dreyer()
    e = np.load(OUT / "reve_emb_dreyer.npz")
    pos_in_eval = {g: i for i, g in enumerate(e["eval_idx"])}
    K, C = len(people), int(d["y"].max()) + 1
    for lam in LAM_PERSON_GRID:
        parts = {k: [] for k in ("reve", "reve_early", "reve_late")}
        ys, ss, fs = [], [], []
        for f, (cal, tgt) in enumerate(folds(d, args.bank)):
            pool = (d["split"] == "train") | cal
            ti = np.where(tgt)[0]
            ei = [pos_in_eval[g] for g in ti]
            (lp, lpe, lpl), _ = reve_predict(
                e["full"][pool], d["y"][pool], e["full"][cal], d["y"][cal], lab[cal], K, C, lam,
                [e["full"][ti], e["early"][ei], e["late"][ei]])
            parts["reve"].append(lp); parts["reve_early"].append(lpe); parts["reve_late"].append(lpl)
            ys.append(d["y"][tgt]); ss.append(d["subject"][tgt]); fs.append(np.full(tgt.sum(), f))
        bank = {k: np.concatenate(v) for k, v in parts.items()}
        bank |= {"y": np.concatenate(ys), "subject": np.concatenate(ss), "fold": np.concatenate(fs)}
        check_alignment(bank["y"], bank["subject"], args.bank)
        np.savez(OUT / f"bank_{args.bank}_reve_lam{lam}.npz", **bank)
        log(f"saved REVE {args.bank} bank, lam_person={lam}")


# ------------------------------------------------------------------ ShallowFBCSPNet

def train_stream(X, y, pool, val, cal_sets, sfreq, seed, threads, epochs=100, ft_epochs=50):
    from models import make_model
    from train_mixture import fit
    torch.set_num_threads(threads)
    n_chans, n_times = X.shape[1:]
    C = int(y.max()) + 1
    pooled, info = fit(make_model("shallow", n_chans, C, n_times, sfreq), X[pool], y[pool],
                       lr=1e-3, epochs=epochs, seed=seed, bs=64, X_val=X[val], y_val=y[val],
                       name="shallow pooled", log_every=5)
    experts = [fit(copy.deepcopy(pooled), X[m], y[m], lr=1e-4, epochs=ft_epochs, seed=seed,
                   bs=32)[0] for m in cal_sets]
    return pooled, experts, info


def predict_all(experts, X):
    from train_mixture import logits
    return np.stack([torch.log_softmax(logits(e, X), 1).numpy() for e in experts], 1)


def cmd_dreyer_shallow(args):
    d, people, lab = dreyer()
    X, y, sf = d["X"], d["y"], float(d["sfreq"])
    parts = {k: [] for k in ("shallow", "shallow_early", "shallow_late")}
    ys, ss, fs = [], [], []
    for f, (cal, tgt) in enumerate(folds(d, args.bank)):
        pool = (d["split"] == "train") | cal
        _, experts, info = train_stream(X, y, pool, d["split"] == "val",
                                        [cal & (d["subject"] == s) for s in people], sf,
                                        args.seed, args.threads, args.epochs)
        log(f"fold {f}: {info}")
        Xt = X[tgt]
        parts["shallow"].append(predict_all(experts, Xt))
        parts["shallow_early"].append(predict_all(experts, masked(Xt, sf, "early")))
        parts["shallow_late"].append(predict_all(experts, masked(Xt, sf, "late")))
        ys.append(y[tgt]); ss.append(d["subject"][tgt]); fs.append(np.full(tgt.sum(), f))
    bank = {k: np.concatenate(v) for k, v in parts.items()}
    bank |= {"y": np.concatenate(ys), "subject": np.concatenate(ss), "fold": np.concatenate(fs)}
    check_alignment(bank["y"], bank["subject"], args.bank)
    np.savez(OUT / f"bank_{args.bank}_shallow_seed{args.seed}.npz", **bank)
    log("saved")


# ------------------------------------------------------------------ EEGNet shortcut diagnostic

def cmd_eegnet_shortcut(args):
    """Early/late-masked predictions of loop B's saved dev EEGNet experts (no retraining)."""
    from train_mixture import make_eegnet
    d, people, _ = dreyer()
    X, sf = d["X"], float(d["sfreq"])
    parts = {"eegnet_early": [], "eegnet_late": []}
    ys, ss = [], []
    for r, (_, tgt) in enumerate(folds(d, "xfit")):
        tag = "" if r == 2 else f"_hold{r}"
        sds = torch.load(LOOPB / f"experts_dev{tag}_seed0.pt", weights_only=True)["experts"]
        experts = []
        for sd in sds:
            e = make_eegnet(X.shape[1], int(d["y"].max()) + 1, X.shape[2])
            e.load_state_dict(sd)
            experts.append(e)
        for part in ("early", "late"):
            parts[f"eegnet_{part}"].append(predict_all(experts, masked(X[tgt], sf, part)))
        ys.append(d["y"][tgt]); ss.append(d["subject"][tgt])
    bank = {k: np.concatenate(v) for k, v in parts.items()}
    bank |= {"y": np.concatenate(ys), "subject": np.concatenate(ss)}
    check_alignment(bank["y"], bank["subject"], "xfit")
    np.savez(OUT / "bank_xfit_eegnet_shortcut.npz", **bank)
    log("saved")


# ------------------------------------------------------------------ BNCI

def cmd_bnci(args):
    from run import load_windows
    d = load_windows()
    subjects = np.unique(d["subject"])
    y = np.searchsorted(np.unique(d["task"]), d["task"])
    tr = (d["session"] == 0) & (d["run"] != 5)
    va = (d["session"] == 0) & (d["run"] == 5)
    te = d["session"] == 1
    C, ste = len(np.unique(y)), d["subject"][te]
    out = {}
    if args.stream == "shallow":
        _, experts, _ = train_stream(d["X"], y, tr, va, [tr & (d["subject"] == s) for s in subjects],
                                     120.0, args.seed, args.threads, epochs=150)
        idx = np.searchsorted(subjects, ste)
        n = np.arange(te.sum())
        for k, X in [("logp", d["X"][te]), ("early", masked(d["X"][te], 120.0, "early")),
                     ("late", masked(d["X"][te], 120.0, "late"))]:
            out[k] = predict_all(experts, X)[n, idx]
    else:
        e = np.load(OUT / "reve_emb_bnci.npz")
        K, p_tr = len(subjects), np.searchsorted(subjects, d["subject"][tr])
        lps, _ = reve_predict(e["full"][tr], y[tr], e["full"][tr], y[tr], p_tr, K, C,
                              args.lam_person, [e["full"][te], e["early"][te], e["late"][te]])
        idx, n = np.searchsorted(subjects, ste), np.arange(te.sum())
        out = {k: lp[n, idx] for k, lp in zip(("logp", "early", "late"), lps)}
    np.savez(OUT / f"bnci_{args.stream}_seed{args.seed}.npz", **out)
    log("saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "dreyer", "bnci", "eegnet-shortcut"])
    ap.add_argument("--dataset", choices=["dreyer", "bnci"], default="dreyer")
    ap.add_argument("--stream", choices=["reve", "shallow"], default="reve")
    ap.add_argument("--bank", choices=["xfit", "test"], default="xfit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lam-person", type=float, default=0.1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "embed":
        cmd_embed(args)
    elif args.cmd == "eegnet-shortcut":
        cmd_eegnet_shortcut(args)
    elif args.cmd == "dreyer":
        (cmd_dreyer_reve if args.stream == "reve" else cmd_dreyer_shallow)(args)
    else:
        cmd_bnci(args)


if __name__ == "__main__":
    main()
