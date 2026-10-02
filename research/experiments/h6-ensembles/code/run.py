"""H6: fixed equal-weight ensembles of saved H3.1 probabilities (see ../protocol.md)."""

import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import fp  # noqa: E402

SRC = fp.OUT / "h3.1-drift-benchmarks"


def load(name):
    z = np.load(SRC / f"{name}_probs.npz")
    return z["P"].astype(np.float64), z["t"]


def geo(*Ps):
    L = np.mean([np.log(np.clip(P, 1e-12, 1)) for P in Ps], 0)
    E = np.exp(L - L.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def main():
    rows = {}
    for bench in ("D1_R1toR2", "D1_R1toR3", "D2_bnci_s1tos2"):
        P_fb, t = load(f"{bench}_ts_fb_C1")
        P_br, _ = load(f"{bench}_ts_broad_C1")
        for seed in (0, 1, 2):
            P_nn, t2 = load(f"{bench}_eegnet_seed{seed}")
            assert (t == t2).all()
            arms = {"eegnet": P_nn, "ts_fb": P_fb, "ts_broad": P_br,
                    "lin": (P_nn + P_fb) / 2, "geo": geo(P_nn, P_fb),
                    "lin3": (P_nn + P_fb + P_br) / 3, "geo3": geo(P_nn, P_fb, P_br)}
            for arm, P in arms.items():
                m = fp.metrics(P, t)
                m.pop("per_person_recall")
                rows.setdefault(bench, {}).setdefault(arm, []).append(m)
    out = {}
    lines = ["| bench | arm | bal_acc | nll | p_true | ece |", "|---|---|---|---|---|---|"]
    for bench, arms in rows.items():
        for arm, ms in arms.items():
            agg = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
                   for k in ms[0]}
            out.setdefault(bench, {})[arm] = agg
            lines.append(f"| {bench} | {arm} | {agg['bal_acc'][0]:.3f} ± {agg['bal_acc'][1]:.3f} "
                         f"| {agg['nll'][0]:.3f} ± {agg['nll'][1]:.3f} | {agg['p_true'][0]:.3f} "
                         f"| {agg['ece'][0]:.3f} |")
    res = EXP / "results"
    res.mkdir(exist_ok=True)
    (res / "summary.json").write_text(json.dumps(out, indent=2))
    (res / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
