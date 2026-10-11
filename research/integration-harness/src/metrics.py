"""Reference rows, per-person scores, the pooled comparison and the adoption rule."""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2")]


def c3_logp(bank):
    """E+T per person with C3 as shipped (Dreyer coefficients; class bias for 2 classes only)."""
    import riemann_parts
    from submission import LogLinearCombiner
    K, C = bank["eeg_logp"].shape[1:]
    comb = LogLinearCombiner(K, C)
    comb.coef.copy_(torch.tensor([riemann_parts.C3[k] for k in ("a", "c0", "c1")], dtype=torch.float64))
    if C == 2:
        comb.class_bias[1] = riemann_parts.C3["b"]
    comb.rel.copy_(torch.as_tensor(bank["rel"], dtype=torch.float64))
    with torch.inference_mode():
        return comb(torch.as_tensor(bank["eeg_logp"]), torch.as_tensor(bank["riemann_logp"])).numpy()


def route(fp_logp, logp):
    p = (np.exp(fp_logp)[:, :, None] * np.exp(logp)).sum(1)
    return np.log(np.clip(p, 1e-12, 1))


def rows(bank):
    lp = c3_logp(bank)
    n = np.arange(len(bank["person"]))
    return {"pooled": bank["pooled_logp"], "current": route(bank["fp_logp"], lp),
            "oracle": lp[n, bank["person"]]}


def person_scores(y, person, logp, K):
    pred = np.asarray(logp).argmax(-1)
    return np.array([balanced_accuracy_score(y[person == k], pred[person == k]) for k in range(K)])


def pooled_comparison(per_dataset, seed=0, n_boot=5000):
    d = np.concatenate([np.asarray(o) - np.asarray(r) for o, r in per_dataset.values()])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"mean": float(d.mean()), "ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "n": int(len(d))}


def adopt(cmp, sim2_delta):
    if cmp["mean"] < 0.005:
        return False, f"pooled gain {cmp['mean']:+.4f} < 0.005"
    if cmp["ci"][0] <= 0:
        return False, f"CI lower bound {cmp['ci'][0]:+.4f} ≤ 0"
    if sim2_delta < -0.01:
        return False, f"sim2 {sim2_delta:+.4f} below row 2 by more than 0.01"
    return True, "adopt"
