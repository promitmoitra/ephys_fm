"""Fair test of subject routing: EEGNet pooled vs EEGNet with per-subject heads.

Follow-up to run.py, whose routing result (0.490 -> 0.553) was against a weak
pooled Riemannian baseline. Here every arm is the same EEGNet trunk with the
same training budget, 4-class MI, trained on session 1 and tested on session 2
(different day) of BNCI2014_001, over several seeds.

Arms (balanced accuracy per 4-s window on session 2):
  pooled             EEGNet -> 4 classes
  multihead_oracle   EEGNet -> 9 subject heads x 4 classes (one final layer of
                     36 logits); routed by the true subject (upper bound)
  multihead_hard     same model, routed by argmax of a fingerprint EEGNet
  multihead_soft     same model, heads weighted by fingerprint probabilities
  multihead_uniform  same model, all heads averaged equally: the control that
                     separates identity routing from plain head ensembling

The fingerprint EEGNet (9-way subject ID) is trained per seed on session 1.
Epochs are selected on session 1's last run (validation); session 2 is never
seen before testing.

Usage (from repo root, venv active; run.py must have cached the windows):
    python experiments/fingerprint_tangermann/routing_eegnet.py [--seeds 5] [--epochs 150]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run import OUT, bal_acc, load_windows

VAL_RUN = 5


def train(n_outputs, X_tr, loss_fn, X_va, val_score, seed, epochs, threads):
    """Train an EEGNet; keep the weights of the best validation epoch."""
    from braindecode.models import EEGNet

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    model = EEGNet(n_chans=X_tr.shape[1], n_outputs=n_outputs,
                   n_times=X_tr.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt = torch.from_numpy(X_tr)
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch = -1.0, None, -1
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(64):
            opt.zero_grad()
            loss_fn(model(Xt[idx]), idx).backward()
            opt.step()
        score = val_score(logits(model, X_va))
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_epoch


def logits(model, X):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(torch.from_numpy(X[i:i + 512]))
                          for i in range(0, len(X), 512)])


def run_seed(d, seed, epochs, threads):
    X, subj, sess, run, task = (d["X"], d["subject"], d["session"], d["run"],
                                d["task"])
    subjects = np.unique(subj)
    K, C = len(subjects), len(np.unique(task))
    s_idx = np.searchsorted(subjects, subj)          # subject -> 0..K-1
    y = np.searchsorted(np.unique(task), task)       # class -> 0..C-1
    tr = (sess == 0) & (run != VAL_RUN)
    va = (sess == 0) & (run == VAL_RUN)
    te = sess == 1
    y_tr = torch.from_numpy(y[tr]).long()
    s_tr = torch.from_numpy(s_idx[tr]).long()
    out, info = {}, {}

    # pooled: one 4-class head
    m, v, e = train(
        C, X[tr], lambda z, i: F.cross_entropy(z, y_tr[i]), X[va],
        lambda z: bal_acc(y[va], z.argmax(1).numpy()), seed, epochs, threads)
    out["pooled"] = logits(m, X[te]).argmax(1).numpy()
    info["pooled"] = {"val": v, "best_epoch": e}

    # multi-head: 36 logits viewed as (K subjects, C classes); train each
    # window on its own subject's head
    def head_logits(z, s):
        return z.view(-1, K, C)[torch.arange(len(z)), s]

    s_va = torch.from_numpy(s_idx[va]).long()
    m, v, e = train(
        K * C, X[tr], lambda z, i: F.cross_entropy(head_logits(z, s_tr[i]), y_tr[i]),
        X[va], lambda z: bal_acc(y[va], head_logits(z, s_va).argmax(1).numpy()),
        seed, epochs, threads)
    info["multihead"] = {"val": v, "best_epoch": e}
    probs = torch.softmax(logits(m, X[te]).view(-1, K, C), -1)  # (n, K, C)

    # fingerprint: 9-way subject ID
    f, v, e = train(
        K, X[tr], lambda z, i: F.cross_entropy(z, s_tr[i]), X[va],
        lambda z: bal_acc(s_idx[va], z.argmax(1).numpy()), seed, epochs, threads)
    p_subj = torch.softmax(logits(f, X[te]), 1)                 # (n, K)
    info["fingerprint"] = {"val": v, "best_epoch": e,
                           "test_bal_acc": bal_acc(s_idx[te], p_subj.argmax(1).numpy())}

    n = torch.arange(len(probs))
    out["multihead_oracle"] = probs[n, torch.from_numpy(s_idx[te])].argmax(1).numpy()
    out["multihead_hard"] = probs[n, p_subj.argmax(1)].argmax(1).numpy()
    out["multihead_soft"] = (p_subj[:, :, None] * probs).sum(1).argmax(1).numpy()
    out["multihead_uniform"] = probs.mean(1).argmax(1).numpy()

    scores = {}
    for arm, pred in out.items():
        per_s = [bal_acc(y[te][subj[te] == s], pred[subj[te] == s]) for s in subjects]
        scores[arm] = {"bal_acc": bal_acc(y[te], pred),
                       "mean_over_subjects": float(np.mean(per_s)),
                       "per_subject": [float(a) for a in per_s]}
    return scores, info


ARMS = ["pooled", "multihead_oracle", "multihead_hard", "multihead_soft",
        "multihead_uniform"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default="routing_eegnet")
    args = ap.parse_args()

    d = load_windows()
    t0 = time.time()
    per_seed = []
    for seed in range(args.seeds):
        ts = time.time()
        scores, info = run_seed(d, seed, args.epochs, args.threads)
        per_seed.append({"seed": seed, "scores": scores, "info": info})
        print(f"seed {seed} ({time.time() - ts:.0f} s): " + ", ".join(
            f"{a} {scores[a]['bal_acc']:.3f}" for a in ARMS)
            + f" | fingerprint {info['fingerprint']['test_bal_acc']:.3f}", flush=True)

    summary = {}
    for a in ARMS:
        vals = np.array([s["scores"][a]["bal_acc"] for s in per_seed])
        diff = vals - np.array([s["scores"]["pooled"]["bal_acc"] for s in per_seed])
        summary[a] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                      "diff_vs_pooled_mean": float(diff.mean()),
                      "diff_vs_pooled_sd": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
                      "seeds_better_than_pooled": int((diff > 0).sum())}
    fp = np.array([s["info"]["fingerprint"]["test_bal_acc"] for s in per_seed])
    result = {"args": vars(args), "summary": summary,
              "fingerprint_test_bal_acc": {"mean": float(fp.mean()),
                                           "sd": float(fp.std(ddof=1)) if len(fp) > 1 else 0.0},
              "per_seed": per_seed, "runtime_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(result, indent=2))
    (OUT / f"{args.tag}.md").write_text(render(result))
    print(render(result))


def render(r):
    n = r["args"]["seeds"]
    lines = ["# EEGNet: pooled vs subject-routed heads (BNCI2014_001)\n",
             f"4-class MI, session 1 → session 2, balanced accuracy per 4-s window "
             f"(chance 0.25). {n} seeds, {r['args']['epochs']} epochs, best epoch on "
             f"session-1 run {VAL_RUN}. Fingerprint EEGNet subject-ID accuracy on "
             f"session 2: {r['fingerprint_test_bal_acc']['mean']:.3f} ± "
             f"{r['fingerprint_test_bal_acc']['sd']:.3f} (chance 0.111).\n",
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
