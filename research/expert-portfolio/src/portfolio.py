"""Loop C: N-stream reliability-weighted log-linear combiner, and its locked evaluation."""

import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[2]
sys.path.insert(0, str(REPO / "research" / "expert-weights" / "src"))
from combine import Combiner  # noqa: E402

OUT = REPO / "outputs" / "t2-portfolio"


class PortfolioLogLinear(Combiner):
    """z = Σ_s w_s·log p_s + (c0 + c1·(rel − 0.5))·log p_riemann + [0, b];
    experts = [*neural_streams, riemann_key, rel_key]. With one neural stream this is
    loop B's RelLogLinear (same initialisation and penalty)."""

    def init(self):
        S = len(self.experts) - 2
        th = {"w": torch.full((S,), 0.8, dtype=torch.float64),
              "c": torch.tensor([0.4, 0.0], dtype=torch.float64),
              "b": torch.zeros(1, dtype=torch.float64)}
        return th

    def penalty(self, theta, theta0):
        return sum(((theta[k] - theta0[k]) ** 2).sum() for k in theta)

    def combine(self, F, theta):
        *neural, cls, rel = self.experts
        z = sum(w * F[e] for w, e in zip(theta["w"], neural))
        z = z + (theta["c"][0] + theta["c"][1] * (F[rel] - 0.5)) * F[cls]
        z = z + torch.cat([torch.zeros_like(theta["b"]), theta["b"]])
        return torch.log_softmax(z, -1)


import json  # noqa: E402

from combine import bal, eval_global, nll, own  # noqa: E402

LOOPB = REPO / "outputs" / "t2-expert-weights"
EXP = REPO / "research" / "expert-portfolio" / "experiments" / "01-dev-eval" / "results"


def boot_ci(d, seed=0, n=5000):
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def dev_bank(lam):
    b = dict(np.load(LOOPB / "bank_xfit_seed0.npz", allow_pickle=True))
    c = np.load(LOOPB / "classical_xfit_ts_C0.1.npz")
    b["ts"], b["rel"] = c["logp"], c["rel_dev"][b["fold"]][:, :, None]
    for name, f in [("reve", OUT / f"bank_xfit_reve_lam{lam}.npz"),
                    ("shallow", OUT / "bank_xfit_shallow_seed0.npz"),
                    ("eegnet", OUT / "bank_xfit_eegnet_shortcut.npz")]:   # eegnet_early/_late only
        if f.exists():
            s = np.load(f)
            assert (s["y"] == b["y"]).all()
            for k in s.files:
                if k.startswith(name):
                    b[k] = s[k]
    return b


def person_nll(y, t, lp, K):
    return np.array([nll(lp[t == k], y[t == k]) for k in range(K)])


def evaluate_dev(lam):
    b = dev_bank(lam)
    y, t, K = b["y"], b["true_idx"], len(b["people"])
    streams = [s for s in ("eegnet", "reve", "shallow") if s in b]
    res = {"alone": {}, "combo": {}, "drop": {}, "shortcut": {}}
    for s in streams + ["ts"]:
        F = own(b, [s])[s]
        res["alone"][s] = {"bal_acc": bal(y, F), "nll": nll(F, y)}
        for part in ("early", "late"):
            if f"{s}_{part}" in b:
                res["shortcut"].setdefault(s, {})[part] = bal(y, own(b, [f"{s}_{part}"])[f"{s}_{part}"])
    combos = {"E+T": ["eegnet"], "E+T+R": ["eegnet", "reve"], "E+T+S": ["eegnet", "shallow"],
              "E+T+R+S": ["eegnet", "reve", "shallow"]}
    oofs = {}
    for name, neural in combos.items():
        if not all(s in b for s in neural):
            continue
        r = eval_global(b, PortfolioLogLinear, [*neural, "ts", "rel"])
        oofs[name] = r
        res["combo"][name] = {"bal_acc": r["bal_acc"], "nll": r["nll"]}
    full = "E+T+R+S" if "E+T+R+S" in oofs else max(oofs, key=lambda k: len(k))
    for drop, name in [("reve", full.replace("+R", "")), ("shallow", full.replace("+S", ""))]:
        if name == full or name not in oofs:
            continue
        a, z = oofs[full], oofs[name]
        d_acc = a["per_person"] - z["per_person"]
        d_nll = person_nll(y, t, z["oof"], K) - person_nll(y, t, a["oof"], K)
        res["drop"][drop] = {"acc_cost": a["bal_acc"] - z["bal_acc"], "acc_ci": boot_ci(d_acc),
                             "nll_cost": z["nll"] - a["nll"], "nll_ci": boot_ci(d_nll)}
        # amendment 1: the accuracy route also needs its CI to exclude zero
        crit_a = res["drop"][drop]["acc_cost"] >= 0.005 and res["drop"][drop]["acc_ci"][1] > 0
        crit_n = res["drop"][drop]["nll_cost"] >= 0.005 and res["drop"][drop]["nll_ci"][1] > 0
        res["drop"][drop]["passes_a"] = bool(crit_a or crit_n)
    return res, b


def evaluate_bnci(lam_unused, dev):
    """Condition (b): Dreyer-fitted weights, applied to BNCI's own-person streams."""
    cl = np.load(LOOPB / "bnci" / "classical.npz")
    from run import load_windows  # experiments/fingerprint_tangermann
    d = load_windows()
    y = np.searchsorted(np.unique(d["task"]), d["task"])[d["session"] == 1]
    subj = d["subject"][d["session"] == 1]
    C = len(np.unique(y))
    rel = cl["rel_ts_C0.1"]
    rel_n = 0.5 + 0.5 * (rel - 1 / C) / (1 - 1 / C)
    rel_w = rel_n[np.searchsorted(np.unique(subj), subj)][:, None]
    out = {}
    for seed in range(5):
        F = {"eegnet": np.load(LOOPB / "bnci" / f"eegnet_seed{seed}.npz")["eegnet"],
             "ts": cl["ts_C0.1"], "rel": rel_w}
        for s in ("reve", "shallow"):
            f = OUT / f"bnci_{s}_seed{seed if s == 'shallow' else 0}.npz"
            if f.exists():
                F[s] = np.load(f)["logp"]
        row = {}
        for name, neural in {"E+T": ["eegnet"], "E+T+R": ["eegnet", "reve"],
                             "E+T+S": ["eegnet", "shallow"]}.items():
            if not all(s in F for s in neural):
                continue
            comb = PortfolioLogLinear([*neural, "ts", "rel"]).fit(own(dev, [*neural, "ts", "rel"]),
                                                                  dev["y"])
            if C != 2:
                comb.theta["b"] = torch.zeros(C - 1, dtype=torch.float64)
            row[name] = bal(y, comb.predict({k: F[k] for k in [*neural, "ts", "rel"]}))
        out[seed] = row
    return out


def main():
    import argparse
    sys.path.insert(0, str(REPO / "experiments" / "fingerprint_tangermann"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    EXP.mkdir(parents=True, exist_ok=True)
    res, dev = evaluate_dev(a.lam)
    (EXP / "dev.json").write_text(json.dumps(res, indent=2))
    bn = evaluate_bnci(a.lam, dev)
    (EXP / "bnci.json").write_text(json.dumps({str(k): v for k, v in bn.items()}, indent=2))
    print(json.dumps(res, indent=2)); print(json.dumps(bn, indent=2))


if __name__ == "__main__":
    main()
