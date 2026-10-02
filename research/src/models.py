"""Named fingerprint recipes: fit on (X_tr, t_tr), return probabilities on X_te.

Each name fixes every hyper-parameter, so a result is reproducible from its name, seed and the
thread count.
"""

import numpy as np

import fp
import riemann


def stochastic(name):
    return name.startswith("eegnet") or name.startswith("ens_")


def fit_predict(name, X_tr, t_tr, X_te, seed=0):
    K = int(t_tr.max()) + 1
    if name == "eegnet":
        m, _ = fp.fit_fixed(fp.make_eegnet(X_tr.shape[1], K, X_tr.shape[2]), X_tr, t_tr,
                            epochs=150, seed=seed, name=f"eegnet seed{seed}")
        return fp.predict_proba(m, X_te)
    if name.startswith("ts_"):                     # ts_broad_C1, ts_fb_C0.1, ...
        _, feat, c = name.split("_")
        F_tr, F_te = riemann.tangent_features(X_tr, X_te, bands=riemann.BANDS[feat])
        return riemann.fit_predict_lr(F_tr, t_tr, F_te, C=float(c[1:]))
    raise ValueError(name)
