"""Protocol 06: zero-shot transfer of the Dreyer-fitted combiners to BNCI 2014-001.

Stage 1 (--stage experts): per seed, pooled EEGNet + per-person fine-tunes on session 1
(runs 0–4, val run 5) → own-person log-probs on session 2; cached per seed.
Stage 2 (--stage classical): per-person ts / ts_C0.1 experts and reliability; cached.
Stage 3 (--stage score): combiners with transferred coefficients.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "experiments" / "fingerprint_tangermann"), str(REPO / "research" / "src")]
from classical import FBTangent, VARIANTS, _banded  # noqa: E402
from combine import OUT as BOUT, LogLinearPool, bal, eval_global, nll  # noqa: E402

OUT = BOUT / "bnci"
COEF = {"C1": (0.826, 0.420), "C3": (0.810, 0.960, 1.701)}


def data():
    from run import load_windows
    d = load_windows()
    subjects = np.unique(d["subject"])
    y = np.searchsorted(np.unique(d["task"]), d["task"])
    tr = (d["session"] == 0) & (d["run"] != 5)
    va = (d["session"] == 0) & (d["run"] == 5)
    te = d["session"] == 1
    return d, subjects, y, tr, va, te


def stage_experts(seeds, threads):
    from finetune_eegnet import finetune
    from routing_eegnet import logits, train
    d, subjects, y, tr, va, te = data()
    X, subj = d["X"], d["subject"]
    C = len(np.unique(y))
    yt = torch.from_numpy(y[tr]).long()
    for seed in seeds:
        path = OUT / f"eegnet_seed{seed}.npz"
        if path.exists():
            continue
        t0 = time.time()
        pooled, v, e = train(C, X[tr], lambda z, i: F.cross_entropy(z, yt[i]), X[va],
                             lambda z: bal(y[va], z.numpy()), seed, 150, threads)
        lp = np.zeros((int(te.sum()), C))
        ste = subj[te]
        for s in subjects:
            m, mv = tr & (subj == s), va & (subj == s)
            _, last, _ = finetune(pooled, X[m], y[m], X[mv], y[mv], lr=1e-4, head_only=False,
                                  epochs=50, seed=seed)
            lp[ste == s] = torch.log_softmax(logits(last, X[te & (subj == s)]), 1).numpy()
        pooled_lp = torch.log_softmax(logits(pooled, X[te]), 1).numpy()
        np.savez(path, eegnet=lp, pooled=pooled_lp, best_epoch=e)
        print(f"seed {seed}: pooled best epoch {e}, expert oracle {bal(y[te], lp):.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)


def stage_classical():
    d, subjects, y, tr, va, te = data()
    X, subj, run = d["X"], d["subject"], d["run"]
    C = len(np.unique(y))
    out = {}
    for v in ["ts", "ts_C0.1"]:
        bands, win, Cr, scale = VARIANTS[v]
        Xb = _banded(X, bands, win)
        lp = np.zeros((int(te.sum()), C))
        rel = np.zeros(len(subjects))
        for j, s in enumerate(subjects):
            m = tr & (subj == s)
            model = FBTangent(bands, Cr, scale).fit([x[m] for x in Xb], y[m])
            lp[subj[te] == s] = model.predict_logp([x[te & (subj == s)] for x in Xb])
            accs = []
            for a in range(5):
                ma = m & (run == a)
                ma_model = FBTangent(bands, Cr, scale).fit([x[ma] for x in Xb], y[ma])
                for b in range(5):
                    if b != a:
                        mb = m & (run == b)
                        accs.append(bal(y[mb], ma_model.predict_logp([x[mb] for x in Xb])))
            rel[j] = np.mean(accs)
        out[v], out[f"rel_{v}"] = lp, rel
        print(f"{v}: oracle {bal(y[te], lp):.3f}; reliability {np.round(rel, 2)}", flush=True)
    np.savez(OUT / "classical.npz", **out)


def stage_score(seeds):
    d, subjects, y, tr, va, te = data()
    yte, ste = y[te], d["subject"][te]
    K, C = len(subjects), len(np.unique(y))
    cl = np.load(OUT / "classical.npz")
    rel = cl["rel_ts_C0.1"]
    rel_n = 0.5 + 0.5 * (rel - 1 / C) / (1 - 1 / C)
    rel_w = rel_n[np.searchsorted(subjects, ste)][:, None]

    def lsm(z):
        return torch.log_softmax(torch.as_tensor(z), -1).numpy()

    res = {}
    for seed in seeds:
        e = np.load(OUT / f"eegnet_seed{seed}.npz")["eegnet"]
        rules = {
            "R0": e,
            "R1": np.log(0.5 * np.exp(e) + 0.5 * np.exp(cl["ts"])),
            "C1": lsm(COEF["C1"][0] * e + COEF["C1"][1] * cl["ts"]),
            "C3": lsm(COEF["C3"][0] * e + (COEF["C3"][1] + COEF["C3"][2] * (rel_w - 0.5))
                      * cl["ts_C0.1"]),
        }
        # exploratory: H4 fitted leave-one-person-out on BNCI's own session 2
        bank = {"y": yte, "true_idx": np.searchsorted(subjects, ste), "people": subjects,
                "eegnet": e, "ts": cl["ts"]}
        rules["H4_LOPO_on_test (exploratory)"] = eval_global(bank, LogLinearPool,
                                                             ["eegnet", "ts"])["oof"]
        res[seed] = {}
        for k, lp in rules.items():
            pp = [bal(yte[ste == s], lp[ste == s]) for s in subjects]
            res[seed][k] = {"bal_acc": bal(yte, lp), "nll": nll(lp, yte), "per_person": pp}
    names = list(res[seeds[0]])
    lines = ["# Protocol 06: BNCI 2014-001, session 1 → 2, oracle ID; Dreyer coefficients "
             "transferred unchanged\n",
             f"ts alone {bal(yte, cl['ts']):.3f}, ts_C0.1 alone {bal(yte, cl['ts_C0.1']):.3f} "
             f"(chance 0.25). Reliability (ts_C0.1, 4-class) per person: "
             f"{', '.join(f'{r:.2f}' for r in rel)}.\n",
             "| Rule | Bal acc (mean ± SD over seeds) | NLL | Δ vs R0 (mean ± SD) | seeds better |",
             "|---|---|---|---|---|"]
    for k in names:
        a = np.array([res[s][k]["bal_acc"] for s in seeds])
        n_ = np.array([res[s][k]["nll"] for s in seeds])
        dlt = a - np.array([res[s]["R0"]["bal_acc"] for s in seeds])
        lines.append(f"| {k} | {a.mean():.3f} ± {a.std():.3f} | {n_.mean():.3f} | "
                     f"{dlt.mean():+.3f} ± {dlt.std():.3f} | {(dlt > 0).sum()}/{len(seeds)} |")
    (HERE.parent / "results" / "bnci.json").write_text(json.dumps(
        {str(k): v for k, v in res.items()}, indent=2))
    (HERE.parent / "results" / "bnci.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["experts", "classical", "score"], required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    {"experts": lambda: stage_experts(seeds, a.threads), "classical": stage_classical,
     "score": lambda: stage_score(seeds)}[a.stage]()


if __name__ == "__main__":
    main()
