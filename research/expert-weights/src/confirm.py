"""Confirmation on the test bank (R4–R6): fit combiners on the full dev bank, apply unchanged.

Only for candidates pre-registered in a committed checkpoint protocol.

    python research/expert-weights/src/confirm.py --checkpoint checkpoint-1 --candidates R0,R1,H5a,H3 \
        --registry research/expert-weights/experiments/01-combiners/code/run.py
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from combine import OUT, bal, load_bank, nll, own, soft_route

REPO = Path(__file__).resolve().parents[3]
LOOP_A_FP = REPO / "outputs" / "t2-fingerprint" / "confirm-1" / "ts_fb_C1_seed0_probs.npz"


def per_person(y, t, logp, K):
    return np.array([bal(y[t == k], logp[t == k]) for k in range(K)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--registry", required=True, help="module defining RUNS")
    args = ap.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("reg", args.registry)
    reg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reg)

    dev, test = load_bank(), load_bank("bank_test_packaged.npz")
    y, t, K = test["y"], test["true_idx"], len(test["people"])
    routes = {"oracle": None, "soft_fp_packaged": "fp"}
    if LOOP_A_FP.exists():
        a = np.load(LOOP_A_FP)
        assert (a["t"] == t).all(), "loop A window order differs"
        test["fp_loopA"] = np.log(np.clip(a["P"].astype(np.float64), 1e-7, 1))
        routes["soft_fp_loopA_tsfb"] = "fp_loopA"

    res = {}
    for cid in args.candidates.split(","):
        desc, make, ex, scope = reg.RUNS[cid]
        assert scope != "person", "per-person combiners are not transferable as fitted"
        comb = make(ex).fit(own(dev, ex), dev["y"])
        res[cid] = {"desc": desc, "params": comb.describe()}
        for rname, key in routes.items():
            lp = comb.predict(own(test, ex)) if key is None else soft_route(test, comb, ex, key)
            res[cid][rname] = {"bal_acc": bal(y, lp), "nll": nll(lp, y),
                               "per_person": per_person(y, t, lp, K).tolist()}
    lines = [f"# {args.checkpoint}: test bank (R4–R6), combiners fitted on the dev bank\n",
             "| ID | Combiner | " + " | ".join(f"{r} bal acc / NLL" for r in routes)
             + " | oracle vs R0: Δ, better / worse |",
             "|---|---|" + "---|" * len(routes) + "---|"]
    base = np.array(res["R0"]["oracle"]["per_person"]) if "R0" in res else None
    for cid, r in res.items():
        cells = " | ".join(f"{r[k]['bal_acc']:.3f} / {r[k]['nll']:.3f}" for k in routes)
        cmp_ = "—"
        if base is not None and cid != "R0":
            d = np.array(r["oracle"]["per_person"]) - base
            cmp_ = f"{d.mean():+.3f}, {(d > 0).sum()} / {(d < 0).sum()}"
        lines.append(f"| {cid} | {r['desc']} | {cells} | {cmp_} |")
    lines.append(f"\n2,520 windows (21 people × 120). Params: "
                 + "; ".join(f"{c} `{json.dumps(r['params'])}`" for c, r in res.items()))
    out = REPO / "research" / "expert-weights" / "experiments" / args.checkpoint / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirm.json").write_text(json.dumps(res, indent=2))
    (out / "confirm.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
