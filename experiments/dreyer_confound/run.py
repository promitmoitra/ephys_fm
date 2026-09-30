"""Is Dreyer 2023 decoding motor imagery, or the screen?

The Track 2 warm-up window runs 0-4 s from the cue. In Dreyer's protocol a
left/right arrow is shown for 0-1.25 s, then a feedback bar extends left or
right for 1.25-4 s: lateralized visual stimuli that can drive eye movements
and visual responses. A pooled EEGNet reaching ~0.87-0.90 cross-subject is far
above typical motor-imagery performance, so this screens where the decodable
information lives.

Every condition trains the same pooled EEGNet on the kit's warm-up split
(train subjects, epoch picked on val subjects) and tests on the unseen test
subjects 61-81 (all runs), i.e. the warm-up evaluation:

  full           27 channels, 0-4 s                  reference
  cue_only       0-1.25 s: arrow, before feedback     cue period alone
  feedback_only  1.25-4 s: feedback bar period        imagery + bar
  no_frontal     drop Fz, F3, F4                      eye-movement-prone channels
  motor_strip    C and CP rows only (14 channels)     motor cortex only
  shuffled       full input, labels shuffled within   chance control
                 each training subject

Scores are also split by run type: acquisition runs R1-R2 (described as sham
feedback) vs online runs R3-R6 (feedback driven by the online classifier).

Uses the window cache written by track2/train_mixture.py (the kit's exact
Dreyer pipeline: 27 EEG channels, 120 Hz, 480 samples from the cue).

Usage (repo root, venv active):
    python experiments/dreyer_confound/run.py [--epochs 30] [--seed 0]
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "experiments" / "dreyer_windows.npz"
OUT = Path(__file__).resolve().parent / "results"
SFREQ = 120
CUE_END = int(1.25 * SFREQ)                       # arrow disappears
FRONTAL = ["Fz", "F3", "F4"]
MOTOR = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bal_acc(y, p):
    return float(balanced_accuracy_score(y, p))


def conditions(ch_names):
    all_ch = list(range(len(ch_names)))
    idx = {c: i for i, c in enumerate(ch_names)}
    full_t = slice(0, None)
    return {
        "full": (all_ch, full_t, False),
        "cue_only": (all_ch, slice(0, CUE_END), False),
        "feedback_only": (all_ch, slice(CUE_END, None), False),
        "no_frontal": ([i for c, i in idx.items() if c not in FRONTAL], full_t, False),
        "motor_strip": ([idx[c] for c in MOTOR], full_t, False),
        "shuffled": (all_ch, full_t, True),
    }


def logits(model, X, bs=512):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(torch.from_numpy(np.ascontiguousarray(X[i:i + bs])))
                          for i in range(0, len(X), bs)])


def train_eval(X_tr, y_tr, X_va, y_va, X_te, seed, epochs, name):
    from braindecode.models import EEGNet
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = EEGNet(n_chans=X_tr.shape[1], n_outputs=2, n_times=X_tr.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(np.ascontiguousarray(X_tr)), torch.from_numpy(y_tr).long()
    best, best_state, best_epoch = -1.0, None, 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(64):
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        score = bal_acc(y_va, logits(model, X_va).argmax(1).numpy())
        if score > best:
            best, best_epoch = score, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    log(f"  {name}: best val {best:.3f} @ epoch {best_epoch}, "
        f"{(time.time() - t0) / epochs:.1f} s/epoch")
    return logits(model, X_te).argmax(1).numpy(), best, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated condition subset")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = np.load(CACHE, allow_pickle=True)
    X, y, subj, run, split = d["X"], d["y"], d["subject"], d["run"], d["split"]
    ch_names = [str(c) for c in d["ch_names"]]
    tr, va, te = split == "train", split == "val", split == "test"
    rng = np.random.default_rng(args.seed)
    y_shuf = y.copy()
    for s in np.unique(subj[tr]):              # keep each subject's class balance
        m = np.where(tr & (subj == s))[0]
        y_shuf[m] = rng.permutation(y[m])
    log(f"windows: train {tr.sum()}, val {va.sum()}, test {te.sum()}; "
        f"channels {len(ch_names)}")

    conds = conditions(ch_names)
    if args.only:
        conds = {k: conds[k] for k in args.only.split(",")}
    results = {}
    for name, (chs, t, shuffle) in conds.items():
        ytr = y_shuf[tr] if shuffle else y[tr]
        pred, val, best_epoch = train_eval(
            X[tr][:, chs, t], ytr, X[va][:, chs, t], y[va], X[te][:, chs, t],
            args.seed, args.epochs, name)
        yt, rt, st = y[te], run[te], subj[te]
        acq, onl = rt < 2, rt >= 2
        per_run = {int(r): bal_acc(yt[rt == r], pred[rt == r]) for r in np.unique(rt)}
        per_subj = [bal_acc(yt[st == s], pred[st == s]) for s in np.unique(st)]
        results[name] = {
            "test": bal_acc(yt, pred), "val": val, "best_epoch": best_epoch,
            "acquisition_R1_R2": bal_acc(yt[acq], pred[acq]),
            "online_R3_R6": bal_acc(yt[onl], pred[onl]),
            "per_run": per_run,
            "per_subject_min_max": [float(min(per_subj)), float(max(per_subj))],
            "n_channels": len(chs), "n_times": int(X[:, :, t].shape[2])}
        r = results[name]
        log(f"RESULT {name}: test {r['test']:.3f} | acquisition {r['acquisition_R1_R2']:.3f} "
            f"| online {r['online_R3_R6']:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}" + (f"_{args.only.replace(',', '-')}" if args.only else "")
    out = {"args": vars(args), "results": results}
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=2))
    (OUT / f"{tag}.md").write_text(render(out))
    print(render(out), flush=True)


def render(o):
    a = o["args"]
    lines = ["# Dreyer 2023 confound screen: motor imagery or the screen?\n",
             f"Pooled EEGNet per condition: train on the kit's train subjects, best of "
             f"{a['epochs']} epochs on val subjects, tested on unseen test subjects "
             f"61–81 (warm-up split). 2 classes, chance 0.5. Seed {a['seed']}. Window "
             "0–4 s from the cue: arrow 0–1.25 s, feedback bar 1.25–4 s.\n",
             "| Condition | Channels × samples | Test bal. acc | Acquisition R1–R2 | "
             "Online R3–R6 | Per-subject range |",
             "|---|---|---|---|---|---|"]
    for k, r in o["results"].items():
        lo, hi = r["per_subject_min_max"]
        lines.append(f"| {k} | {r['n_channels']} × {r['n_times']} | {r['test']:.3f} | "
                     f"{r['acquisition_R1_R2']:.3f} | {r['online_R3_R6']:.3f} | "
                     f"{lo:.2f}–{hi:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
