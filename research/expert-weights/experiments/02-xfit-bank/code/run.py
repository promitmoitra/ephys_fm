"""Protocol 02: protocol 01's combiners on the cross-fitted dev bank (2,520 windows)."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from combine import (bal, eval_global, eval_person_loro, load_bank, nll,  # noqa: E402
                     own)

torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location(
    "p01", HERE.parents[1] / "01-combiners" / "code" / "run.py")
p01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p01)
RUNS = p01.RUNS


def main():
    t0 = time.time()
    bank = load_bank("bank_xfit_seed0.npz")
    y, t, fold = bank["y"], bank["true_idx"], bank["fold"]
    K = len(bank["people"])
    res = {}
    for k in ["pooled", "control", "ts", "csp"]:
        F = own(bank, [k])[k]
        res[k] = {"desc": k, "bal_acc": bal(y, F), "nll": nll(F, y),
                  "per_fold": [bal(y[fold == f], F[fold == f]) for f in range(3)]}
    for rid, (desc, make, ex, scope) in RUNS.items():
        r = (eval_person_loro(bank, make, ex) if scope == "person"
             else eval_global(bank, make, ex))
        oof = r["oof"]
        res[rid] = {"desc": desc, "experts": ex, "scope": scope, "bal_acc": r["bal_acc"],
                    "nll": r["nll"], "per_person": r["per_person"].tolist(),
                    "per_fold": [bal(y[fold == f], oof[fold == f]) for f in range(3)],
                    "per_fold_person": [[bal(y[(fold == f) & (t == k)],
                                             oof[(fold == f) & (t == k)]) for k in range(K)]
                                        for f in range(3)],
                    "params_full_fit": make(ex).fit(own(bank, ex), y).describe()}
    base = res["R0"]
    lines = ["# Protocol 02 results: cross-fitted dev bank (each calibration run held out; "
             "true ID)\n",
             "| ID | Combiner | Scope | Bal acc | NLL | vs R0: Δ, better / worse | "
             "Fold R1 | Fold R2 | Fold R3 | Δ vs R0 per fold | Params (full fit) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ["pooled", "control", "ts", "csp"]:
        r = res[k]
        lines.append(f"| — | {k} alone | — | {r['bal_acc']:.3f} | {r['nll']:.3f} | | "
                     + " | ".join(f"{v:.3f}" for v in r["per_fold"]) + " | | |")
    for rid in RUNS:
        r = res[rid]
        d = np.array(r["per_person"]) - np.array(base["per_person"])
        dpf = [np.mean(np.array(r["per_fold_person"][f]) - np.array(base["per_fold_person"][f]))
               for f in range(3)]
        r["vs_R0"] = {"mean_diff": float(d.mean()), "better": int((d > 0).sum()),
                      "worse": int((d < 0).sum()), "per_fold": dpf}
        lines.append(
            f"| {rid} | {r['desc']} | {r['scope']} | {r['bal_acc']:.3f} | {r['nll']:.3f} | "
            f"{d.mean():+.3f}, {(d > 0).sum()} / {(d < 0).sum()} | "
            + " | ".join(f"{v:.3f}" for v in r["per_fold"])
            + f" | {', '.join(f'{v:+.3f}' for v in dpf)} | `{json.dumps(r['params_full_fit'])}` |")
    lines.append(f"\n2,520 windows (21 people × 120; 840 per fold). Per-person rows (H2): "
                 f"leave-one-run-out within person; their 'full fit' params are global. "
                 f"Runtime {time.time() - t0:.0f} s.")
    out = HERE.parent / "results"
    (out / "sweep.json").write_text(json.dumps(res, indent=2))
    (out / "sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
