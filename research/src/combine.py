"""Combiners for per-person experts, and the locked dev-bank evaluation (protocol 00).

A combiner maps each expert's log-probabilities {name: (..., C)} for the same person to a
combined log-probability (..., C). Parametric combiners are fitted by minimising the mean
negative log-likelihood (+ a small L2 penalty toward their "equal average" initial
parameters) with L-BFGS; everything is tiny, so a fit takes milliseconds.

Evaluation under the true ID (identity is known during calibration):
    eval_global   leave-one-person-out: fit on 20 people's R3, score the held-out person
    eval_person   per person, stratified 2-fold on their R3 windows, repeated 5x

Usage:
    from combine import load_bank, eval_global, LinearPool, ...
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "t2-expert-weights"
EPS = 1e-7


def load_bank(name="bank_dev_seed0.npz"):
    d = np.load(OUT / name, allow_pickle=True)
    return {k: d[k] for k in d.files}


def own(bank, experts, idx=None):
    """Each window's own person's expert log-probs: {e: (n, C)} (true-ID routing)."""
    n = np.arange(len(bank["y"]))
    out = {}
    for e in experts:
        v = bank[e]
        out[e] = v[n, bank["true_idx"]] if v.ndim == 3 else v
    if idx is not None:
        out = {e: v[idx] for e, v in out.items()}
    return out


def nll(logp, y):
    logp = np.asarray(logp)
    return float(-logp[np.arange(len(y)), y].mean())


def bal(y, logp):
    return float(balanced_accuracy_score(y, np.asarray(logp).argmax(-1)))


def _t(F):
    return {e: torch.as_tensor(v, dtype=torch.float64) for e, v in F.items()}


def _logmeanexp_w(logps, w):
    """log sum_e w_e p_e, logps (E, ..., C), w (E, ...) or (E,) broadcastable."""
    while w.dim() < logps.dim():
        w = w.unsqueeze(-1)
    return torch.logsumexp(logps + torch.log(w.clamp_min(1e-12)), 0)


# --------------------------------------------------------------------------
# Combiners
# --------------------------------------------------------------------------

class Combiner:
    """Base: parameters in self.theta (dict of tensors); subclasses define combine()."""

    l2 = 1e-3

    def __init__(self, experts):
        self.experts = list(experts)
        self.theta = self.init()

    def init(self):
        return {}

    def combine(self, F, theta):
        raise NotImplementedError

    def penalty(self, theta, theta0):
        return sum(((theta[k] - theta0[k]) ** 2).sum() for k in theta)

    def fit(self, F, y):
        if not self.theta:
            return self
        F = _t(F)
        y = torch.as_tensor(y, dtype=torch.long)
        theta0 = {k: v.detach().clone() for k, v in self.init().items()}
        theta = {k: v.detach().clone().requires_grad_(True) for k, v in theta0.items()}
        opt = torch.optim.LBFGS(list(theta.values()), lr=0.5, max_iter=200,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            lp = self.combine(F, theta)
            loss = -lp[torch.arange(len(y)), y].mean() + self.l2 * self.penalty(theta, theta0)
            loss.backward()
            return loss

        opt.step(closure)
        self.theta = {k: v.detach() for k, v in theta.items()}
        return self

    def predict(self, F):
        with torch.no_grad():
            return self.combine(_t(F), self.theta).numpy()

    def describe(self):
        return {k: np.round(v.numpy(), 3).tolist() for k, v in self.theta.items()}


class Single(Combiner):
    def combine(self, F, theta):
        return torch.log_softmax(F[self.experts[0]], -1)


class EqualLinear(Combiner):
    """Plain average of probabilities (the current '+ts' rule)."""

    def combine(self, F, theta):
        L = torch.stack([F[e] for e in self.experts])
        return _logmeanexp_w(L, torch.full((len(self.experts),), 1 / len(self.experts),
                                           dtype=L.dtype))


class LinearPool(Combiner):
    """sum_e w_e p_e with w = softmax(a) learned (H1); init = equal weights."""

    def init(self):
        return {"a": torch.zeros(len(self.experts), dtype=torch.float64)}

    def combine(self, F, theta):
        L = torch.stack([F[e] for e in self.experts])
        return _logmeanexp_w(L, torch.softmax(theta["a"], 0))

    def weights(self):
        return torch.softmax(self.theta["a"], 0).numpy()


class LogLinearPool(Combiner):
    """p ∝ exp(sum_e w_e log p_e + b) (H4); unconstrained w act as temperatures too.
    init: w = 1/E, b = 0 (geometric mean). With bias=True this is stacking on log-probs."""

    def __init__(self, experts, bias=False, l2=1e-3):
        self.bias, self.l2 = bias, l2
        super().__init__(experts)

    def init(self):
        th = {"w": torch.full((len(self.experts),), 1 / len(self.experts),
                              dtype=torch.float64)}
        if self.bias:
            th["b"] = torch.zeros(1, dtype=torch.float64)
        return th

    def combine(self, F, theta):
        z = sum(w * F[e] for w, e in zip(theta["w"], self.experts))
        if self.bias:
            z = z + torch.cat([torch.zeros_like(theta["b"]), theta["b"]])  # 2-class bias
        return torch.log_softmax(z, -1)


class TempLinearPool(Combiner):
    """Per-expert temperature, then learned linear pooling (H5)."""

    def init(self):
        E = len(self.experts)
        return {"logT": torch.zeros(E, dtype=torch.float64),
                "a": torch.zeros(E, dtype=torch.float64)}

    def combine(self, F, theta):
        L = torch.stack([torch.log_softmax(F[e] / torch.exp(theta["logT"][i]), -1)
                         for i, e in enumerate(self.experts)])
        return _logmeanexp_w(L, torch.softmax(theta["a"], 0))


class TempEqualLinear(TempLinearPool):
    """Per-expert temperature, then an equal average (isolates the calibration effect)."""

    def init(self):
        return {"logT": torch.zeros(len(self.experts), dtype=torch.float64)}

    def combine(self, F, theta):
        return super().combine(F, {"logT": theta["logT"],
                                   "a": torch.zeros(len(self.experts), dtype=torch.float64)})


class ConfidenceGate(Combiner):
    """Per-window weights w_e(x) = softmax_e(a_e + g * conf_e(x)), conf = max_c p_e(c|x)
    (H6), on temperature-scaled experts."""

    def init(self):
        E = len(self.experts)
        return {"a": torch.zeros(E, dtype=torch.float64),
                "g": torch.zeros(1, dtype=torch.float64),
                "logT": torch.zeros(E, dtype=torch.float64)}

    def combine(self, F, theta):
        L = torch.stack([torch.log_softmax(F[e] / torch.exp(theta["logT"][i]), -1)
                         for i, e in enumerate(self.experts)])
        conf = L.exp().amax(-1)                                     # (E, ...)
        a = theta["a"].view(-1, *([1] * (conf.dim() - 1)))
        w = torch.softmax(a + theta["g"] * conf, 0)
        return _logmeanexp_w(L, w)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _summary(y, subj_idx, logp, K):
    per = np.array([bal(y[subj_idx == k], logp[subj_idx == k]) for k in range(K)])
    return {"bal_acc": bal(y, logp), "nll": nll(logp, y), "per_person": per}


def eval_global(bank, make, experts):
    """Leave-one-person-out for global combiners. make(experts) -> Combiner."""
    y, t = bank["y"], bank["true_idx"]
    K = len(bank["people"])
    F = own(bank, experts)
    oof = np.zeros((len(y), F[experts[0]].shape[-1]))
    for k in range(K):
        tr, te = t != k, t == k
        c = make(experts).fit({e: v[tr] for e, v in F.items()}, y[tr])
        oof[te] = c.predict({e: v[te] for e, v in F.items()})
    return _summary(y, t, oof, K) | {"oof": oof}


def eval_person(bank, make, experts, reps=5, seed=0):
    """Per-person combiners: stratified 2-fold within each person's R3, repeated."""
    y, t = bank["y"], bank["true_idx"]
    K = len(bank["people"])
    F = own(bank, experts)
    runs = []
    for r in range(reps):
        oof = np.zeros((len(y), F[experts[0]].shape[-1]))
        for k in range(K):
            idx = np.where(t == k)[0]
            skf = StratifiedKFold(2, shuffle=True, random_state=seed + r)
            for tr, te in skf.split(idx, y[idx]):
                tr, te = idx[tr], idx[te]
                c = make(experts).fit({e: v[tr] for e, v in F.items()}, y[tr])
                oof[te] = c.predict({e: v[te] for e, v in F.items()})
        runs.append(_summary(y, t, oof, K))
    return {"bal_acc": float(np.mean([s["bal_acc"] for s in runs])),
            "bal_acc_sd": float(np.std([s["bal_acc"] for s in runs])),
            "nll": float(np.mean([s["nll"] for s in runs])),
            "per_person": np.mean([s["per_person"] for s in runs], 0)}


def paired(a, b):
    """Per-person comparison of result a vs reference b."""
    d = a["per_person"] - b["per_person"]
    return {"mean_diff": float(d.mean()), "better": int((d > 0).sum()),
            "worse": int((d < 0).sum())}


def soft_route(bank, comb, experts, fp_key):
    """Soft-routed mixture: combine every person's experts on every window, then mix by
    p(person | window) from bank[fp_key] (log-probs, (n, K))."""
    Fall = {e: bank[e] for e in experts}                          # (n, K, C)
    logp_k = comb.predict(Fall)                                   # (n, K, C)
    p = (np.exp(bank[fp_key])[:, :, None] * np.exp(logp_k)).sum(1)
    return np.log(np.clip(p, EPS, 1))
