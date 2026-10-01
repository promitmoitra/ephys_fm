"""Protocol 04: checkpoint-2 combiners (fitted on the seed-0 cross-fitted dev bank) applied
unchanged to test banks from EEGNet training seeds 0, 1, 2 (oracle ID)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from combine import (OUT, EqualLinear, LogLinearPool, RelLogLinear, Single, bal,  # noqa: E402
                     load_bank, nll, own)

torch.set_num_threads(1)
CANDS = {
    "R0": ("EEGNet only", lambda e: Single(e), ["eegnet"]),
    "R1": ("equal average, ts", EqualLinear, ["eegnet", "ts"]),
    "C1": ("log-linear, ts", LogLinearPool, ["eegnet", "ts"]),
    "C3": ("reliability-weighted, ts_C0.1", RelLogLinear, ["eegnet", "ts_C0.1", "rel_ts_C0.1"]),
}
BANKS = {0: "bank_test_packaged.npz", 1: "bank_test_seed1.npz", 2: "bank_test_seed2.npz"}


def main():
    dev = load_bank("bank_xfit_seed0.npz")
    rel = {}
    for v in ["ts", "ts_C0.1"]:
        d = np.load(OUT / f"classical_xfit_{v}.npz")
        dev[v] = d["logp"]
        dev[f"rel_{v}"] = d["rel_dev"][dev["fold"]][:, :, None]
        rel[v] = d["rel_test"]
    combs = {c: m(ex).fit(own(dev, ex), dev["y"]) for c, (_, m, ex) in CANDS.items()}
    ts_c01 = np.load(OUT / "classical_test_ts_C0.1.npy")
    rng = np.random.default_rng(0)
    res = {}
    for seed, name in BANKS.items():
        b = load_bank(name)
        y, t, K = b["y"], b["true_idx"], len(b["people"])
        b["ts_C0.1"] = ts_c01
        for v in rel:
            b[f"rel_{v}"] = np.broadcast_to(rel[v][None, :, None], (len(y), K, 1)).copy()
        res[seed] = {}
        pp = {}
        for c, (_, _, ex) in CANDS.items():
            lp = combs[c].predict(own(b, ex))
            pp[c] = np.array([bal(y[t == k], lp[t == k]) for k in range(K)])
            res[seed][c] = {"bal_acc": bal(y, lp), "nll": nll(lp, y)}
        for c in CANDS:
            d = pp[c] - pp["R0"]
            bs = [d[rng.integers(0, K, K)].mean() for _ in range(5000)]
            res[seed][c]["vs_R0"] = [float(d.mean()), float(np.percentile(bs, 2.5)),
                                     float(np.percentile(bs, 97.5)), int((d > 0).sum()),
                                     int((d < 0).sum())]
    lines = ["# Protocol 04: seed robustness on R4–R6 (oracle ID; combiners fitted on the "
             "seed-0 cross-fitted dev bank)\n",
             "| Seed | " + " | ".join(f"{c} acc / NLL" for c in CANDS)
             + " | C1 − R0 [95% CI], b/w | C3 − R0 [95% CI], b/w |",
             "|---|" + "---|" * (len(CANDS) + 2)]
    for seed, r in res.items():
        def g(c):
            v = r[c]["vs_R0"]
            return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}], {v[3]}/{v[4]}"
        lines.append(f"| {seed} | " + " | ".join(f"{r[c]['bal_acc']:.3f} / {r[c]['nll']:.3f}"
                                                 for c in CANDS) + f" | {g('C1')} | {g('C3')} |")
    mean = {c: (np.mean([res[s][c]["bal_acc"] for s in res]),
                np.mean([res[s][c]["nll"] for s in res])) for c in CANDS}
    lines.append("| mean | " + " | ".join(f"{a:.3f} / {n:.3f}" for a, n in mean.values())
                 + " | " + f"{mean['C1'][0] - mean['R0'][0]:+.3f} | "
                 + f"{mean['C3'][0] - mean['R0'][0]:+.3f} |")
    (HERE.parent / "results" / "seeds.json").write_text(json.dumps(res, indent=2))
    (HERE.parent / "results" / "seeds.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
