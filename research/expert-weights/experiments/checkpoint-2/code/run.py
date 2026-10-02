"""Checkpoint 2: fit C1–C4 (and references) on the cross-fitted dev bank, apply to R4–R6."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE.parents[2] / "src"))
from classical import test_fit  # noqa: E402
from combine import (OUT, EqualLinear, LogLinearPool, RelLogLinear, Single, bal,  # noqa: E402
                     load_bank, nll, own, soft_route)

torch.set_num_threads(1)
LOOP_A_FP = REPO / "outputs" / "t2-fingerprint" / "confirm-1" / "ts_fb_C1_seed0_probs.npz"

CANDIDATES = {  # id: (description, make, experts)
    "R0": ("EEGNet only", lambda e: Single(e), ["eegnet"]),
    "R1": ("equal average, ts", EqualLinear, ["eegnet", "ts"]),
    "H3": ("log-linear + bias, ts", lambda e: LogLinearPool(e, bias=True), ["eegnet", "ts"]),
    "C1": ("H4 log-linear, ts", LogLinearPool, ["eegnet", "ts"]),
    "C2": ("H3 log-linear + bias, ts_C0.1", lambda e: LogLinearPool(e, bias=True),
           ["eegnet", "ts_C0.1"]),
    "C3": ("H8 reliability-weighted, ts_C0.1", RelLogLinear,
           ["eegnet", "ts_C0.1", "rel_ts_C0.1"]),
    "C4": ("H8 reliability-weighted, ts", RelLogLinear, ["eegnet", "ts", "rel_ts"]),
}


def main():
    dev, test = load_bank("bank_xfit_seed0.npz"), load_bank("bank_test_packaged.npz")
    y, t, K = test["y"], test["true_idx"], len(test["people"])
    for v in ["ts", "ts_C0.1"]:
        d = np.load(OUT / f"classical_xfit_{v}.npz")
        dev[v] = d["logp"]
        dev[f"rel_{v}"] = d["rel_dev"][dev["fold"]][:, :, None]
        test[f"rel_{v}"] = np.broadcast_to(d["rel_test"][None, :, None], (len(y), K, 1)).copy()
    tpath = OUT / "classical_test_ts_C0.1.npy"
    if not tpath.exists():
        np.save(tpath, test_fit("ts_C0.1"))
    test["ts_C0.1"] = np.load(tpath)
    routes = {"oracle": None, "soft_fp_packaged": "fp"}
    if LOOP_A_FP.exists():
        a = np.load(LOOP_A_FP)
        assert (a["t"] == t).all()
        test["fp_loopA"] = np.log(np.clip(a["P"].astype(np.float64), 1e-7, 1))
        routes["soft_fp_loopA"] = "fp_loopA"

    rng = np.random.default_rng(0)
    res = {}
    for cid, (desc, make, ex) in CANDIDATES.items():
        comb = make(ex).fit(own(dev, ex), dev["y"])
        res[cid] = {"desc": desc, "params": comb.describe()}
        for rname, key in routes.items():
            lp = comb.predict(own(test, ex)) if key is None else soft_route(test, comb, ex, key)
            pp = np.array([bal(y[t == k], lp[t == k]) for k in range(K)])
            res[cid][rname] = {"bal_acc": bal(y, lp), "nll": nll(lp, y), "per_person": pp.tolist()}
    base = np.array(res["R0"]["oracle"]["per_person"])
    for cid in res:
        d = np.array(res[cid]["oracle"]["per_person"]) - base
        bs = [d[rng.integers(0, K, K)].mean() for _ in range(5000)]
        res[cid]["vs_R0_oracle"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                    float(np.percentile(bs, 97.5))],
                                    "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
    lines = ["# Checkpoint 2: R4–R6, combiners fitted on the cross-fitted dev bank\n",
             "| ID | Combiner | " + " | ".join(f"{r} acc / NLL" for r in routes)
             + " | oracle vs R0: Δ [95% CI], better / worse |",
             "|---|---|" + "---|" * len(routes) + "---|"]
    for cid, r in res.items():
        v = r["vs_R0_oracle"]
        lines.append(f"| {cid} | {r['desc']} | "
                     + " | ".join(f"{r[k]['bal_acc']:.3f} / {r[k]['nll']:.3f}" for k in routes)
                     + f" | {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}], "
                       f"{v['better']} / {v['worse']} |")
    lines.append("\n2,520 windows. Params: " + "; ".join(
        f"{c} `{json.dumps(r['params'])}`" for c, r in res.items()))
    (HERE.parent / "results" / "confirm.json").write_text(json.dumps(res, indent=2))
    (HERE.parent / "results" / "confirm.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
