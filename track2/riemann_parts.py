"""Fit the covariance-based parts of the Track 2 mixture and export them as tensors.

Training side only (scipy / pyriemann / sklearn). The exported state dicts load
into submission.py's torch modules, which reproduce these pipelines exactly:

    fit_fingerprint   loop A's filter-bank fingerprint (research/src/torch_fp.py):
                      6 bands, full window, OAS → tangent space → standardised
                      multinomial logistic regression (C = 1) on all calibration
                      windows, labels = participant index
    fit_riemann       loop B's per-participant experts (research/expert-weights):
                      8–30 Hz, 0.5–4 s, OAS → tangent space → logistic regression
                      (C = 0.1), plus each participant's calibration reliability
                      (run-to-run balanced accuracy, chance-normalised)
    C3                loop B's combiner coefficients, fitted on the Dreyer
                      cross-fitted dev bank (research/expert-weights/experiments/
                      checkpoint-2): refit them for a new dataset / regime
"""

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

FP_BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
RIEMANN_BAND, RIEMANN_WIN_S, RIEMANN_C = (8, 30), (0.5, 4.0), 0.1
# Dreyer cross-fitted dev bank, combiner C3 (a, c0, c1; class bias for class 1)
C3 = {"a": 0.80986932, "c0": 0.96004621, "c1": 1.70110631, "b": -0.01604924}


def _sos(band, sfreq):
    return butter(4, list(band), btype="bandpass", fs=sfreq, output="sos")


def _covs(X):
    from pyriemann.estimation import Covariances
    return Covariances("oas").fit_transform(X)


def fit_fingerprint(X, person, sfreq, bands=FP_BANDS, C=1.0):
    """State dict for submission.FBFingerprint (float32, as loop A's export)."""
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.base import invsqrtm
    T = X.shape[-1]
    filters, crefs, feats = [], [], []
    for band in bands:
        sos = _sos(band, sfreq)
        Xb = sosfiltfilt(sos, X, axis=-1).astype(np.float32).astype(np.float64)
        cov = _covs(Xb)
        ts = TangentSpace(metric="riemann").fit(cov)
        feats.append(ts.transform(cov))
        filters.append(sosfiltfilt(sos, np.eye(T), axis=-1).astype(np.float32))
        crefs.append(invsqrtm(ts.reference_))
    F = np.hstack(feats)
    sc = StandardScaler().fit(F)
    lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(F), person)
    weight = lr.coef_ / sc.scale_
    bias = lr.intercept_ - weight @ sc.mean_
    if weight.shape[0] == 1:           # sklearn binary: one logit for class 1 → logits [0, z]
        weight = np.vstack([np.zeros_like(weight), weight])
        bias = np.concatenate([[0.0], bias])
    f32 = torch.float32
    return {"filters": torch.as_tensor(np.stack(filters), dtype=f32),
            "cref_isqrt": torch.as_tensor(np.stack(crefs), dtype=f32),
            "linear.weight": torch.as_tensor(weight, dtype=f32),
            "linear.bias": torch.as_tensor(bias, dtype=f32)}


class _Riemann:
    """One participant's tangent-space logistic regression (sklearn reference)."""

    def __init__(self, sfreq):
        self.sos = _sos(RIEMANN_BAND, sfreq)
        self.a, self.b = (int(round(s * sfreq)) for s in RIEMANN_WIN_S)

    def _x(self, X):
        return sosfiltfilt(self.sos, X, axis=-1)[:, :, self.a:self.b].astype(np.float64)

    def fit(self, X, y):
        from pyriemann.tangentspace import TangentSpace
        cov = _covs(self._x(X))
        self.ts = TangentSpace(metric="riemann").fit(cov)
        self.lr = LogisticRegression(C=RIEMANN_C, max_iter=5000).fit(self.ts.transform(cov), y)
        return self

    def predict(self, X):
        return self.lr.predict(self.ts.transform(_covs(self._x(X))))


def fit_riemann(X, y, person, run, n_people, n_classes, sfreq):
    """State dicts for submission.RiemannExperts and LogLinearCombiner.

    X, y, person, run: the evaluation participants' calibration windows."""
    from pyriemann.utils.base import invsqrtm
    T = X.shape[-1]
    ref = _Riemann(sfreq)
    crefs, weights, biases, rel = [], [], [], []
    for k in range(n_people):
        m = person == k
        model = _Riemann(sfreq).fit(X[m], y[m])
        crefs.append(invsqrtm(model.ts.reference_))
        coef, icpt = model.lr.coef_, model.lr.intercept_
        if n_classes == 2:                      # sklearn binary: one logit for class 1
            coef = np.vstack([np.zeros_like(coef), coef])
            icpt = np.array([0.0, icpt[0]])
        weights.append(coef)
        biases.append(icpt)
        units = np.unique(run[m])               # calibration runs / sessions
        accs = []
        for a in units:
            fa = _Riemann(sfreq).fit(X[m & (run == a)], y[m & (run == a)])
            for b in units:
                if b != a:
                    mb = m & (run == b)
                    accs.append(balanced_accuracy_score(y[mb], fa.predict(X[mb])))
        rel.append(np.mean(accs))
    rel = np.asarray(rel)
    rel_norm = 0.5 + 0.5 * (rel - 1 / n_classes) / (1 - 1 / n_classes)
    filt = np.ascontiguousarray(sosfiltfilt(ref.sos, np.eye(T), axis=-1)[:, ref.a:ref.b])
    f64 = torch.float64
    riemann = {"filt": torch.as_tensor(filt, dtype=f64),
               "cref_isqrt": torch.as_tensor(np.stack(crefs), dtype=f64),
               "weight": torch.as_tensor(np.stack(weights), dtype=f64),
               "bias": torch.as_tensor(np.stack(biases), dtype=f64)}
    class_bias = np.zeros(n_classes)
    if n_classes == 2:
        class_bias[1] = C3["b"]
    combiner = {"coef": torch.tensor([C3["a"], C3["c0"], C3["c1"]], dtype=f64),
                "class_bias": torch.as_tensor(class_bias, dtype=f64),
                "rel": torch.as_tensor(rel_norm, dtype=f64)}
    return riemann, combiner, rel, ref.b - ref.a
