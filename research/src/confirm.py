"""Confirmation on the hidden runs R4-R6 (outer-loop checkpoints only).

Run ONLY for candidates pre-registered in a committed
research/experiments/confirm-<n>/protocol.md. Never use its output to pick
among variants after the fact.

Each candidate fingerprint is refit on R1-R3 of the 21 evaluation people and
scored on R4-R6: 21-way metrics, and the soft-routed mixture with the FIXED
EEGNet experts of outputs/track2_dreyer_sim/submission/mixture.pt
(baseline: fingerprint 0.744, soft mixture 0.901, oracle 0.908, seed 0).

    python research/src/confirm.py --exp confirm-1 --candidates eegnet,ts_fb_C1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp  # noqa: E402
import models  # noqa: E402

SUB = fp.WT / "outputs" / "track2_dreyer_sim" / "submission"
EXPERT_CACHE = fp.OUT / "cache" / "experts_R4R6.npz"


def expert_probs(X):
    """(n, K, C) class probabilities of the 21 fixed experts on the hidden windows."""
    if EXPERT_CACHE.exists():
        return np.load(EXPERT_CACHE)["P"]
    sys.path[:0] = [str(fp.WT / "external" / "2026-competition"), str(fp.WT / "track2")]
    from submission import build_model
    config = json.loads((SUB / "config.json").read_text())
    state = torch.load(SUB / "mixture.pt", map_location="cpu", weights_only=True)
    meta = {"ch_names": config["ch_names"], "n_times": config["n_times"],
            "n_classes": config["n_classes"], "device": "cpu"}
    mix = build_model(meta, config, state)
    P = np.stack([fp.predict_proba(e, X) for e in mix.experts], 1)
    np.savez_compressed(EXPERT_CACHE, P=P.astype(np.float32))
    return P


def mixture_scores(P_fp, P_exp, y, t):
    n = np.arange(len(y))
    K = P_exp.shape[1]
    out = {}
    for name, w in (("soft", P_fp),
                    ("hard", np.eye(K)[P_fp.argmax(1)]),
                    ("oracle", np.eye(K)[t]),
                    ("uniform", np.full_like(P_fp, 1 / K))):
        p = (w[:, :, None] * P_exp).sum(1).argmax(1)
        out[name] = float(balanced_accuracy_score(y, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    exp_dir = fp.WT / "research" / "experiments" / args.exp
    assert (exp_dir / "protocol.md").exists(), "pre-register the candidates first"

    d = fp.load_eval()
    X, y, t, run = d["X"], d["y"], d["person"], d["run"]
    tr, te = run <= 2, run >= 3
    P_exp = expert_probs(X[te])
    summary = {}
    for cand in args.candidates.split(","):
        seeds = [int(s) for s in args.seeds.split(",")] if models.stochastic(cand) else [0]
        for seed in seeds:
            t0 = time.time()
            P = models.fit_predict(cand, X[tr], t[tr], X[te], seed=seed)
            name = f"{cand}_seed{seed}"
            m = fp.save_run(exp_dir, name, P, t[te], {"candidate": cand, "seed": seed})
            mix = mixture_scores(P, P_exp, y[te], t[te])
            summary[name] = {"fp_bal_acc": m["bal_acc"], "fp_nll": m["nll"],
                             "p_true": m["p_true"], **{f"mix_{k}": v for k, v in mix.items()},
                             "wall_s": round(time.time() - t0, 1)}
            fp.log(f"{name}: mixture {mix}")
    (exp_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
