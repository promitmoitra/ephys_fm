"""Per-person classical expert variants, cross-fitted over the calibration runs.

Each variant: per band, zero-phase band-pass → time window → OAS covariance → Riemannian
tangent space (reference fitted on the person's training windows); bands concatenated →
standardised → logistic regression (L2, C). One band = the original `ts` expert.

cross_fit(variant) returns log-probs in the cross-fitted bank's window order: for held-out
calibration run r = 0, 1, 2, each person's model is fitted on their other two calibration runs
and predicts run r, for every person's model on every run-r window: (2520, K, C).
test_fit(variant) fits on R1–R3 and predicts R4–R6 (confirmation only): (2520, K, C).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[1]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2")]
SFREQ = 120.0

VARIANTS = {
    # name: (bands, window seconds, C, standardise features)
    "ts": ([(8, 30)], (0.5, 4.0), 1.0, False),          # = the bank's ts expert
    "ts_C0.1": ([(8, 30)], (0.5, 4.0), 0.1, False),
    "ts_w0": ([(8, 30)], (0.0, 4.0), 1.0, False),
    "ts_broad": ([(1, 40)], (0.0, 4.0), 1.0, False),
    "ts_fb5": ([(4, 8), (8, 13), (13, 20), (20, 30), (30, 40)], (0.5, 4.0), 0.1, True),
    "ts_fb5_w0": ([(4, 8), (8, 13), (13, 20), (20, 30), (30, 40)], (0.0, 4.0), 0.1, True),
    "ts_mu_beta": ([(8, 13), (13, 30)], (0.5, 4.0), 0.1, True),
    "ts_low": ([(1, 4)], (0.0, 4.0), 1.0, False),
}


class FBTangent:
    def __init__(self, bands, C, scale):
        self.bands, self.C, self.scale = bands, C, scale

    def _feats(self, Xb, fit):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        out = []
        if fit:
            self.ts = [TangentSpace(metric="riemann") for _ in Xb]
        for X, ts in zip(Xb, self.ts):
            cov = Covariances("oas").fit_transform(X)
            out.append(ts.fit_transform(cov) if fit else ts.transform(cov))
        return np.concatenate(out, 1)

    def fit(self, Xb, y):
        Z = self._feats(Xb, True)
        self.sc = StandardScaler(with_mean=self.scale, with_std=self.scale).fit(Z)
        self.lr = LogisticRegression(C=self.C, max_iter=5000).fit(self.sc.transform(Z), y)
        return self

    def predict_logp(self, Xb):
        p = self.lr.predict_proba(self.sc.transform(self._feats(Xb, False)))
        return np.log(np.clip(p, 1e-7, 1.0))


def _load():
    from train_mixture import load_windows
    d = load_windows()
    is_eval = d["split"] == "test"
    people = sorted(np.unique(d["subject"][is_eval]), key=int)
    return {k: d[k][is_eval] for k in ("X", "y", "subject", "run")}, people


def _banded(X, bands, win):
    a, b = int(win[0] * SFREQ), int(win[1] * SFREQ)
    out = []
    for lo, hi in bands:
        sos = butter(4, [lo, hi], btype="bandpass", fs=SFREQ, output="sos")
        out.append(sosfiltfilt(sos, X, axis=-1)[:, :, a:b].astype(np.float64))
    return out


def _fit_predict(E, people, variant, train_runs, target):
    bands, win, C, scale = VARIANTS[variant]
    Xb = _banded(E["X"], bands, win)
    Xt = [x[target] for x in Xb]
    out = np.empty((int(target.sum()), len(people), 2))
    for j, s in enumerate(people):
        m = (E["subject"] == s) & np.isin(E["run"], train_runs)
        model = FBTangent(bands, C, scale).fit([x[m] for x in Xb], E["y"][m])
        out[:, j] = model.predict_logp(Xt)
    return out


def cross_fit(variant, E=None, people=None):
    if E is None:
        E, people = _load()
    return np.concatenate([
        _fit_predict(E, people, variant, [q for q in range(3) if q != r], E["run"] == r)
        for r in range(3)])


def test_fit(variant, E=None, people=None):
    if E is None:
        E, people = _load()
    return _fit_predict(E, people, variant, [0, 1, 2], E["run"] >= 3)
