"""Shared harness for loop A: the per-window fingerprint (person ID) model.

Setting (docs/handoff/track2-research-loops.md): Dreyer 2023, the kit's 21
test people (subjects 61-81) play the sealed phase's evaluation participants.
Runs R1-R3 (run 0-2) are labeled calibration, R4-R6 (run 3-5) hidden test.

Locked inner-loop protocol: train on R1-R2, score on R3 (21-way balanced
accuracy and NLL), fixed epoch budget, last epoch. R4-R6 are touched only by
the confirmation script (confirm.py), never for decisions.
"""

import json
import time
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]              # the worktree root
REPO_DATA = WT / "data"
CACHE = REPO_DATA / "experiments" / "dreyer_windows.npz"
OUT = WT / "outputs" / "t2-fingerprint"
EVAL_CACHE = OUT / "cache" / "eval_windows.npz"
POOL_CACHE = OUT / "cache" / "pool_windows.npz"
SFREQ = 120.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _split_caches():
    """Split the 1 GB kit cache once into eval-people and pool files."""
    d = np.load(CACHE, allow_pickle=True)
    split = d["split"]
    X = d["X"]
    common = {"ch_names": d["ch_names"], "sfreq": d["sfreq"]}
    for path, m in ((EVAL_CACHE, split == "test"), (POOL_CACHE, split != "test")):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X[m], y=d["y"][m], subject=d["subject"][m],
                 run=d["run"][m], split=split[m], **common)
    del X


def load_eval():
    """The 21 evaluation people: X (5040, 27, 480), y, person index 0-20, run."""
    if not EVAL_CACHE.exists():
        _split_caches()
    d = np.load(EVAL_CACHE, allow_pickle=True)
    subj = d["subject"].astype(str)
    people = sorted(np.unique(subj), key=int)
    idx = {s: i for i, s in enumerate(people)}
    return {"X": d["X"], "y": d["y"], "person": np.array([idx[s] for s in subj]),
            "run": d["run"], "people": people,
            "ch_names": [str(c) for c in d["ch_names"]]}


def load_pool(splits=("train", "val")):
    """The kit's training people (train: 52, val: 14) with their run labels."""
    if not POOL_CACHE.exists():
        _split_caches()
    d = np.load(POOL_CACHE, allow_pickle=True)
    m = np.isin(d["split"], splits)
    subj = d["subject"][m].astype(str)
    people = sorted(np.unique(subj), key=int)
    idx = {s: i for i, s in enumerate(people)}
    return {"X": d["X"][m], "y": d["y"][m],
            "person": np.array([idx[s] for s in subj]), "run": d["run"][m],
            "people": people}


def load_bnci():
    """BNCI2014_001 (9 people x 2 days), the kit's preprocessing at 120 Hz, 22 ch x 480."""
    d = np.load(REPO_DATA / "experiments" / "tangermann_windows.npz", allow_pickle=True)
    subj = d["subject"]
    people = sorted(np.unique(subj))
    idx = {s: i for i, s in enumerate(people)}
    return {"X": d["X"].astype(np.float32), "y": d["task"],
            "person": np.array([idx[s] for s in subj]), "session": d["session"],
            "run": d["run"], "people": people}


# --------------------------------------------------------------------------
# Metrics: soft routing uses the probabilities, so score them, not just argmax
# --------------------------------------------------------------------------

def metrics(P, t, n_bins=15):
    """P (n, K) probabilities, t (n,) true person index."""
    from sklearn.metrics import balanced_accuracy_score
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-12, 1)
    P = P / P.sum(1, keepdims=True)
    n = np.arange(len(t))
    pred = P.argmax(1)
    conf = P.max(1)
    correct = pred == t
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean())
              * (bins == b).mean() for b in range(n_bins) if (bins == b).any())
    top3 = (np.argsort(-P, 1)[:, :3] == t[:, None]).any(1).mean()
    K = P.shape[1]
    recall = [float(correct[t == k].mean()) for k in range(K) if (t == k).any()]
    return {"bal_acc": float(balanced_accuracy_score(t, pred)),
            "nll": float(-np.log(P[n, t]).mean()),
            "p_true": float(P[n, t].mean()),          # mass soft routing puts on the right expert
            "ece": float(ece), "top3": float(top3),
            "worst_person_recall": float(min(recall)),
            "per_person_recall": recall}


def fmt(m):
    return (f"bal_acc {m['bal_acc']:.3f}  nll {m['nll']:.3f}  p_true {m['p_true']:.3f}  "
            f"ece {m['ece']:.3f}  top3 {m['top3']:.3f}  worst {m['worst_person_recall']:.2f}")


def save_run(exp_dir, name, P, t, info):
    """Per-window probabilities to outputs/, metrics JSON next to the protocol."""
    exp_dir = Path(exp_dir)
    art = OUT / exp_dir.name
    art.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art / f"{name}_probs.npz", P=np.asarray(P, np.float32), t=t)
    m = metrics(P, t)
    res = exp_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{name}.json").write_text(json.dumps({"metrics": m, **info}, indent=2))
    log(f"{name}: {fmt(m)}")
    return m


# --------------------------------------------------------------------------
# EEGNet (the recipe of track2/train_mixture.py, with a fixed budget)
# --------------------------------------------------------------------------

def make_eegnet(n_chans, n_outputs, n_times):
    from braindecode.models import EEGNet
    return EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)


def predict_proba(model, X, bs=512):
    import torch
    model.eval()
    with torch.inference_mode():
        return torch.cat([torch.softmax(model(torch.as_tensor(X[i:i + bs])), 1)
                          for i in range(0, len(X), bs)]).numpy()


def fit_fixed(model, X, y, *, lr=1e-3, epochs=150, seed=0, bs=64, wd=1e-3,
              augment=None, monitor=None, monitor_every=25, name=""):
    """AdamW + cross-entropy for a fixed number of epochs; keeps the LAST epoch.

    `monitor(model, epoch)` is for logging curves only; it never selects.
    `augment(xb, generator)` transforms a training batch tensor.
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    t0 = time.time()
    curve = []
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        for idx in torch.randperm(len(Xt), generator=g).split(bs):
            xb = Xt[idx]
            if augment is not None:
                xb = augment(xb, g)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if monitor is not None and ((epoch + 1) % monitor_every == 0 or epoch + 1 == epochs):
            m = monitor(model, epoch + 1)
            curve.append({"epoch": epoch + 1, "train_loss": tot / len(Xt), **m})
            log(f"  {name} epoch {epoch + 1}/{epochs}: loss {tot / len(Xt):.3f} "
                f"{' '.join(f'{k} {v:.3f}' for k, v in m.items())} "
                f"({(time.time() - t0) / (epoch + 1):.1f} s/epoch)")
    return model, curve
