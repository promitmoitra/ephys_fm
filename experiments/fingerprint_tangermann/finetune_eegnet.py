"""Most favourable subject specialization: fine-tune the pooled EEGNet per subject.

Follow-up to routing_eegnet.py, where per-subject heads lost to a pooled
EEGNet even with the true subject ID. Here each subject gets its own copy of
the trained pooled model, fine-tuned on that subject's session-1 data and
tested on its session 2 with the *true* ID (oracle). If this cannot beat the
pooled model, subject specialization at inference is not worth pursuing.

Arms (4-class MI, session 1 -> 2, BNCI2014_001, balanced accuracy per window):
  pooled         the pooled EEGNet, as in routing_eegnet.py
  ft_head_*      final layer only (lr 1e-3), BatchNorm/dropout frozen:
                 a per-subject linear probe on the pooled features
  ft_all_*       all weights (lr 1e-4), BatchNorm statistics adapt
  *_sel          epoch picked on the subject's session-1 validation run, epoch
                 0 (= the pooled model) included, so it can fall back to pooled
  *_fixed        weights after the last fine-tuning epoch (no selection noise)

Usage (from repo root, venv active; run.py must have cached the windows):
    python experiments/fingerprint_tangermann/finetune_eegnet.py [--seeds 5]
"""

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from routing_eegnet import VAL_RUN, logits, train
from run import OUT, bal_acc, load_windows

VARIANTS = {"ft_head": dict(lr=1e-3, head_only=True),
            "ft_all": dict(lr=1e-4, head_only=False)}


def finetune(pooled, X_tr, y_tr, X_va, y_va, lr, head_only, epochs, seed):
    """Fine-tune a copy of `pooled`; return (selected, last) weight copies."""
    torch.manual_seed(seed)
    model = copy.deepcopy(pooled)
    params = (model.final_layer.parameters() if head_only
              else model.parameters())
    if head_only:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.final_layer.parameters():
            p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()
    g = torch.Generator().manual_seed(seed)

    def val():
        return bal_acc(y_va, logits(model, X_va).argmax(1).numpy())

    best, best_state, best_epoch = val(), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        # head-only keeps BN statistics and dropout frozen (eval mode)
        model.train(not head_only)
        for idx in torch.randperm(len(Xt), generator=g).split(32):
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        score = val()
        if score > best:
            best, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
    last = copy.deepcopy(model)
    model.load_state_dict(best_state)
    return model, last, best_epoch


def run_seed(d, seed, epochs, ft_epochs, threads):
    X, subj, sess, run, task = (d["X"], d["subject"], d["session"], d["run"],
                                d["task"])
    subjects = np.unique(subj)
    C = len(np.unique(task))
    y = np.searchsorted(np.unique(task), task)
    tr = (sess == 0) & (run != VAL_RUN)
    va = (sess == 0) & (run == VAL_RUN)
    te = sess == 1
    y_tr = torch.from_numpy(y[tr]).long()

    pooled, v, e = train(
        C, X[tr], lambda z, i: F.cross_entropy(z, y_tr[i]), X[va],
        lambda z: bal_acc(y[va], z.argmax(1).numpy()), seed, epochs, threads)
    preds = {"pooled": logits(pooled, X[te]).argmax(1).numpy()}
    info = {"pooled": {"val": v, "best_epoch": e}}

    arms = [f"{k}_{m}" for k in VARIANTS for m in ("sel", "fixed")]
    for a in arms:
        preds[a] = np.empty(te.sum(), dtype=int)
        info[a] = {"best_epoch_per_subject": {}}
    te_subj = subj[te]
    for s in subjects:
        str_, sva, ste = tr & (subj == s), va & (subj == s), te_subj == s
        for k, kw in VARIANTS.items():
            sel, last, be = finetune(pooled, X[str_], y[str_], X[sva], y[sva],
                                     epochs=ft_epochs, seed=seed, **kw)
            X_te_s = X[te][ste]
            preds[f"{k}_sel"][ste] = logits(sel, X_te_s).argmax(1).numpy()
            preds[f"{k}_fixed"][ste] = logits(last, X_te_s).argmax(1).numpy()
            info[f"{k}_sel"]["best_epoch_per_subject"][int(s)] = be

    scores = {}
    for a, p in preds.items():
        per_s = [bal_acc(y[te][te_subj == s], p[te_subj == s]) for s in subjects]
        scores[a] = {"bal_acc": bal_acc(y[te], p),
                     "mean_over_subjects": float(np.mean(per_s)),
                     "per_subject": [float(x) for x in per_s]}
    return scores, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150, help="pooled training")
    ap.add_argument("--ft-epochs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default="finetune_eegnet")
    args = ap.parse_args()

    d = load_windows()
    t0 = time.time()
    per_seed = []
    for seed in range(args.seeds):
        ts = time.time()
        scores, info = run_seed(d, seed, args.epochs, args.ft_epochs, args.threads)
        per_seed.append({"seed": seed, "scores": scores, "info": info})
        print(f"seed {seed} ({time.time() - ts:.0f} s): " + ", ".join(
            f"{a} {v['bal_acc']:.3f}" for a, v in scores.items()), flush=True)

    arms = list(per_seed[0]["scores"])
    base = np.array([s["scores"]["pooled"]["bal_acc"] for s in per_seed])
    summary = {}
    for a in arms:
        vals = np.array([s["scores"][a]["bal_acc"] for s in per_seed])
        diff = vals - base
        sd = (lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0)
        summary[a] = {"mean": float(vals.mean()), "sd": sd(vals),
                      "diff_vs_pooled_mean": float(diff.mean()),
                      "diff_vs_pooled_sd": sd(diff),
                      "seeds_better_than_pooled": int((diff > 0).sum())}
    result = {"args": vars(args), "summary": summary, "per_seed": per_seed,
              "runtime_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(result, indent=2))
    (OUT / f"{args.tag}.md").write_text(render(result))
    print(render(result))


def render(r):
    n = r["args"]["seeds"]
    lines = ["# EEGNet: pooled vs per-subject fine-tuning with oracle ID (BNCI2014_001)\n",
             f"4-class MI, session 1 → session 2, balanced accuracy per 4-s window "
             f"(chance 0.25). {n} seeds; pooled model {r['args']['epochs']} epochs, "
             f"then {r['args']['ft_epochs']} fine-tuning epochs per subject on its "
             f"session-1 runs 0–4; `_sel` picks the epoch on the subject's run "
             f"{VAL_RUN} (epoch 0 = pooled included), `_fixed` keeps the last.\n",
             "| Arm | Balanced acc (mean ± SD) | Δ vs pooled, paired (mean ± SD) | Seeds > pooled |",
             "|---|---|---|---|"]
    for a, s in r["summary"].items():
        delta = "—" if a == "pooled" else (f"{s['diff_vs_pooled_mean']:+.3f} ± "
                                           f"{s['diff_vs_pooled_sd']:.3f}")
        better = "—" if a == "pooled" else f"{s['seeds_better_than_pooled']}/{n}"
        lines.append(f"| {a} | {s['mean']:.3f} ± {s['sd']:.3f} | {delta} | {better} |")
    lines.append(f"\nRuntime: {r['runtime_s']} s.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
