"""Protocol 01: first combiner sweep on the dev bank (true ID). Writes results/ md + json."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from combine import (ConfidenceGate, EqualLinear, LinearPool, LogLinearPool,  # noqa: E402
                     Single, TempEqualLinear, TempLinearPool, bal, eval_global,
                     eval_person, load_bank, nll, own, paired, soft_route)

torch.set_num_threads(1)
ET, ETC = ["eegnet", "ts"], ["eegnet", "ts", "csp"]

RUNS = {  # id: (description, make, experts, scope)
    "R0": ("EEGNet only", lambda e: Single(e), ["eegnet"], "fixed"),
    "R1": ("equal average eegnet+ts", EqualLinear, ET, "fixed"),
    "R2": ("equal average eegnet+ts+csp", EqualLinear, ETC, "fixed"),
    "H1": ("linear pool, learned", LinearPool, ET, "global"),
    "H2": ("linear pool, learned per person", LinearPool, ET, "person"),
    "H3": ("log-linear + bias (stacking)", lambda e: LogLinearPool(e, bias=True), ET, "global"),
    "H4": ("log-linear", LogLinearPool, ET, "global"),
    "H5a": ("temperature, then equal average", TempEqualLinear, ET, "global"),
    "H5b": ("temperature, then learned linear pool", TempLinearPool, ET, "global"),
    "H6": ("confidence gate", ConfidenceGate, ET, "global"),
    "H7": ("linear pool, learned, +csp", LinearPool, ETC, "global"),
    "H7b": ("log-linear, +csp", LogLinearPool, ETC, "global"),
}
CONTEXT = {"pooled": "pooled EEGNet", "control": "control (fine-tuned on all calib)",
           "ts": "ts only", "csp": "csp only"}


def main():
    t0 = time.time()
    bank = load_bank()
    y = bank["y"]
    res, table = {}, []
    for k, name in CONTEXT.items():
        F = own(bank, [k])[k]
        res[k] = {"desc": name, "bal_acc": bal(y, F), "nll": nll(F, y)}
    for rid, (desc, make, ex, scope) in RUNS.items():
        r = (eval_person(bank, make, ex) if scope == "person" else eval_global(bank, make, ex))
        full = make(ex).fit(own(bank, ex), y)                 # fitted on all of R3
        r_soft = bal(y, soft_route(bank, full, ex, "fp_biased"))
        res[rid] = {"desc": desc, "experts": ex, "scope": scope, "bal_acc": r["bal_acc"],
                    "bal_acc_sd": r.get("bal_acc_sd"), "nll": r["nll"],
                    "per_person": r["per_person"].tolist(),
                    "params_full_fit": full.describe(), "soft_fp_biased": r_soft}
    for rid in RUNS:
        res[rid]["vs_R1"] = paired({"per_person": np.array(res[rid]["per_person"])},
                                   {"per_person": np.array(res["R1"]["per_person"])})
    out = HERE.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(res, indent=2))
    lines = ["# Protocol 01 results: dev bank (experts on R1–R2 → R3, true ID)\n",
             "| ID | Combiner | Experts | Scope | Bal acc | NLL | vs R1: Δ, better / worse "
             "| Soft (fp_biased) | Params (full fit) |", "|---|---|---|---|---|---|---|---|---|"]
    for k in CONTEXT:
        lines.append(f"| — | {res[k]['desc']} | {k} | — | {res[k]['bal_acc']:.3f} | "
                     f"{res[k]['nll']:.3f} | | | |")
    for rid in RUNS:
        r = res[rid]
        sd = f" ± {r['bal_acc_sd']:.3f}" if r["bal_acc_sd"] else ""
        v = r["vs_R1"]
        lines.append(f"| {rid} | {r['desc']} | {'+'.join(r['experts'])} | {r['scope']} | "
                     f"{r['bal_acc']:.3f}{sd} | {r['nll']:.3f} | {v['mean_diff']:+.3f}, "
                     f"{v['better']} / {v['worse']} | {r['soft_fp_biased']:.3f} | "
                     f"`{json.dumps(r['params_full_fit'])}` |")
    lines.append(f"\n840 windows (21 people × 40); one window ≈ 0.0012. Runtime "
                 f"{time.time() - t0:.0f} s.")
    (out / "sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
