"""Covariance / tangent-space features for fingerprints (fit on train, apply to test)."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

SFREQ = 120.0
BANDS = {
    "broad": [None],
    "fb": [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)],
}


def bandpass(X, band, order=4):
    if band is None:
        return X
    sos = butter(order, band, btype="bandpass", fs=SFREQ, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def tangent_features(X_tr, X_te, bands=(None,), estimator="oas", metric="riemann"):
    """Per band: covariance -> tangent space at the training mean; concatenated."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    F_tr, F_te = [], []
    for band in bands:
        cov = Covariances(estimator)
        C_tr = cov.fit_transform(bandpass(X_tr, band).astype(np.float64))
        C_te = cov.transform(bandpass(X_te, band).astype(np.float64))
        ts = TangentSpace(metric=metric).fit(C_tr)
        F_tr.append(ts.transform(C_tr))
        F_te.append(ts.transform(C_te))
    return np.hstack(F_tr), np.hstack(F_te)


def fit_predict_lr(F_tr, y_tr, F_te, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    clf.fit(F_tr, y_tr)
    return clf.predict_proba(F_te)
