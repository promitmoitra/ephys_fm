"""Is the per-subject fine-tuning gain real, and does it survive without IDs?

Follow-up to finetune_eegnet.py, where fine-tuning the pooled EEGNet per
subject (oracle ID) beat it by +0.053. Two open issues:

1. Confound: the pooled model was still improving at its last epoch, and
   fine-tuning adds 50 epochs. Control: continue training the pooled model on
   *all* subjects' session-1 data with the same recipe (50 epochs, same lr and
   frozen parts). Epoch-matched, so the control takes ~9x more gradient steps
   than each per-subject fine-tune: conservative, it favours the control.
2. Track 2 gives no subject IDs at inference. Route the per-subject
   fine-tuned models with a fingerprint EEGNet (9-way subject ID trained on
   session 1): soft = mixture weighted by p(subject | window), hard = argmax.

3. BN-only adaptation (`ft_bn`, AdaBN): keep every weight of the pooled model
   and only re-estimate the BatchNorm running statistics on the subject's
   session-1 windows. No gradients, no labels. Isolates how much of `ft_all`
   comes from BatchNorm statistics adapting to the subject.

All arms use last-epoch weights (epoch selection on 48 validation windows per
subject was noisy in finetune_eegnet.py). 4-class MI, session 1 -> 2,
BNCI2014_001, balanced accuracy per 4-s window. Pooled and fingerprint
training are deterministic given the seed, epochs *and* --threads (the thread
count changes float reduction order), so arms from separate runs with all
three equal are directly comparable.

Usage (from repo root, venv active; run.py must have cached the windows):
    python experiments/fingerprint_tangermann/finetune_routing_eegnet.py \
        [--seeds 5] [--variants ft_head,ft_all,ft_bn]
"""

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from finetune_eegnet import VARIANTS, finetune
from routing_eegnet import VAL_RUN, logits, train
from run import OUT, bal_acc, load_windows

ALL_VARIANTS = [*VARIANTS, "ft_bn"]


def adapt_bn(pooled, X, batch_size=256):
    """AdaBN: copy `pooled`, re-estimate BatchNorm running stats on X.

    Cumulative averages over the batches (momentum=None), dropout off:
    EEGNet's bnorm_2 sits after drop_1, so train-mode dropout would bias its
    statistics. No weights change.
    """
    model = copy.deepcopy(pooled).eval()
    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    with torch.inference_mode():
        for i in range(0, len(X), batch_size):
            model(torch.from_numpy(X[i:i + batch_size]))
    return model.eval()


def run_seed(d, seed, epochs, ft_epochs, threads, variants):
    X, subj, sess, run, task = (d["X"], d["subject"], d["session"], d["run"],
                                d["task"])
    subjects = np.unique(subj)
    K, C = len(subjects), len(np.unique(task))
    s_idx = np.searchsorted(subjects, subj)
    y = np.searchsorted(np.unique(task), task)
    tr = (sess == 0) & (run != VAL_RUN)
    va = (sess == 0) & (run == VAL_RUN)
    te = sess == 1
    y_tr = torch.from_numpy(y[tr]).long()
    s_tr = torch.from_numpy(s_idx[tr]).long()
    X_te, te_s = X[te], s_idx[te]
    n = torch.arange(len(X_te))
    preds, info = {}, {}

    pooled, v, e = train(
        C, X[tr], lambda z, i: F.cross_entropy(z, y_tr[i]), X[va],
        lambda z: bal_acc(y[va], z.argmax(1).numpy()), seed, epochs, threads)
    preds["pooled"] = logits(pooled, X_te).argmax(1).numpy()
    info["pooled"] = {"val": v, "best_epoch": e}

    fp, v, e = train(
        K, X[tr], lambda z, i: F.cross_entropy(z, s_tr[i]), X[va],
        lambda z: bal_acc(s_idx[va], z.argmax(1).numpy()), seed, epochs, threads)
    p_subj = torch.softmax(logits(fp, X_te), 1)                   # (n, K)
    info["fingerprint"] = {"val": v, "best_epoch": e,
                           "test_bal_acc": bal_acc(te_s, p_subj.argmax(1).numpy())}

    def adapt(k, m, mv):
        if k == "ft_bn":
            return adapt_bn(pooled, X[m])
        _, last, _ = finetune(pooled, X[m], y[m], X[mv], y[mv],
                              epochs=ft_epochs, seed=seed, **VARIANTS[k])
        return last

    for k in variants:
        # control: same recipe, all subjects' data
        cont = adapt(k, tr, va)
        preds[f"pooled_cont_{k[3:]}"] = logits(cont, X_te).argmax(1).numpy()

        # per-subject adaptations, each applied to every test window
        probs = torch.empty(len(X_te), K, C)
        for j, s in enumerate(subjects):
            last = adapt(k, tr & (subj == s), va & (subj == s))
            probs[:, j] = torch.softmax(logits(last, X_te), 1)
        preds[f"{k}_oracle"] = probs[n, torch.from_numpy(te_s)].argmax(1).numpy()
        preds[f"{k}_soft"] = (p_subj[:, :, None] * probs).sum(1).argmax(1).numpy()
        preds[f"{k}_hard"] = probs[n, p_subj.argmax(1)].argmax(1).numpy()

    scores = {}
    for a, p in preds.items():
        per_s = [bal_acc(y[te][te_s == j], p[te_s == j]) for j in range(K)]
        scores[a] = {"bal_acc": bal_acc(y[te], p),
                     "mean_over_subjects": float(np.mean(per_s)),
                     "per_subject": [float(x) for x in per_s]}
    return scores, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150, help="pooled/fingerprint")
    ap.add_argument("--ft-epochs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS),
                    help=f"comma-separated subset of {ALL_VARIANTS}")
    ap.add_argument("--tag", default="finetune_routing_eegnet")
    args = ap.parse_args()
    variants = args.variants.split(",")
    unknown = set(variants) - set(ALL_VARIANTS)
    if unknown:
        ap.error(f"unknown variants {sorted(unknown)}")
    order = ["pooled"] + [a for k in variants for a in
                          (f"pooled_cont_{k[3:]}", f"{k}_oracle", f"{k}_soft",
                           f"{k}_hard")]
    baseline = {k: f"pooled_cont_{k[3:]}" for k in variants}

    d = load_windows()
    t0 = time.time()
    per_seed = []
    for seed in range(args.seeds):
        ts = time.time()
        scores, info = run_seed(d, seed, args.epochs, args.ft_epochs, args.threads,
                                variants)
        per_seed.append({"seed": seed, "scores": scores, "info": info})
        print(f"seed {seed} ({time.time() - ts:.0f} s): " + ", ".join(
            f"{a} {scores[a]['bal_acc']:.3f}" for a in order)
            + f" | fingerprint {info['fingerprint']['test_bal_acc']:.3f}", flush=True)

    def arr(a):
        return np.array([s["scores"][a]["bal_acc"] for s in per_seed])

    def sd(x):
        return float(x.std(ddof=1)) if len(x) > 1 else 0.0

    summary = {}
    for a in order:
        vals = arr(a)
        row = {"mean": float(vals.mean()), "sd": sd(vals)}
        refs = ["pooled"] + [b for k, b in baseline.items() if a.startswith(k)]
        for ref in refs:
            if ref != a:
                diff = vals - arr(ref)
                row[f"vs_{ref}"] = {"mean": float(diff.mean()), "sd": sd(diff),
                                    "seeds_better": int((diff > 0).sum())}
        summary[a] = row
    fp = np.array([s["info"]["fingerprint"]["test_bal_acc"] for s in per_seed])
    result = {"args": vars(args), "summary": summary,
              "fingerprint_test_bal_acc": {"mean": float(fp.mean()), "sd": sd(fp)},
              "per_seed": per_seed, "runtime_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(result, indent=2))
    (OUT / f"{args.tag}.md").write_text(render(result))
    print(render(result))


def render(r):
    n = r["args"]["seeds"]

    def delta(row, ref):
        x = row.get(f"vs_{ref}")
        return "—" if x is None else (f"{x['mean']:+.3f} ± {x['sd']:.3f} "
                                      f"({x['seeds_better']}/{n})")

    lines = ["# EEGNet: is per-subject fine-tuning real, and does it survive routing? "
             "(BNCI2014_001)\n",
             f"4-class MI, session 1 → session 2, balanced accuracy per 4-s window "
             f"(chance 0.25), last-epoch weights. {n} seeds; pooled and fingerprint "
             f"EEGNets {r['args']['epochs']} epochs; then {r['args']['ft_epochs']} "
             "epochs of either continued pooled training on all subjects "
             "(`pooled_cont_*`, the control) or per-subject fine-tuning "
             "(`ft_*`). `head` = final layer only (lr 1e-3, BatchNorm frozen); "
             "`all` = all weights (lr 1e-4); `bn` = BatchNorm running statistics "
             "re-estimated only (AdaBN: no gradients, no labels, no epochs). "
             "Routing uses the fingerprint EEGNet: "
             f"subject-ID accuracy on session 2 {r['fingerprint_test_bal_acc']['mean']:.3f} "
             f"± {r['fingerprint_test_bal_acc']['sd']:.3f} (chance 0.111).\n",
             "Δ columns are paired over seeds: mean ± SD (seeds better).\n",
             "| Arm | Balanced acc | Δ vs pooled | Δ vs control (`pooled_cont_*`) |",
             "|---|---|---|---|"]
    for a, row in r["summary"].items():
        ref = f"pooled_cont_{a.split('_')[1]}" if a.startswith("ft_") else None
        lines.append(f"| {a} | {row['mean']:.3f} ± {row['sd']:.3f} | "
                     f"{delta(row, 'pooled')} | {delta(row, ref) if ref else '—'} |")
    lines.append(f"\nRuntime: {r['runtime_s']} s.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
