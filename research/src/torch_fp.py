"""Filter-bank tangent-space fingerprint as a pure-torch module (shippable).

Training uses the scipy / pyriemann / sklearn pipeline of riemann.py; `export`
turns the fitted pipeline into tensors, and `TorchFBFingerprint` reproduces it
at inference with torch only:

  band filter   zero-phase sosfiltfilt on a fixed-length window is linear in
                the input, so it is exactly one (T x T) matrix per band
                (the filter applied to the identity);
  covariance    OAS shrinkage (sklearn.covariance.oas, centered, 1/n);
  tangent space logm(Cref^-1/2 C Cref^-1/2), upper triangle with sqrt(2)
                off-diagonal weights (pyriemann.TangentSpace, metric riemann);
  classifier    StandardScaler + LogisticRegression folded into one Linear.
"""

import numpy as np
import torch
from torch import nn

import riemann


def filter_matrix(band, n_times):
    """(T, T) matrix M with sosfiltfilt(x) == x @ M for a length-T window."""
    return riemann.bandpass(np.eye(n_times), band).astype(np.float64)


def oas(X):
    """OAS covariance of (..., C, T) windows, as sklearn.covariance.oas."""
    Xc = X - X.mean(-1, keepdim=True)
    n, p = X.shape[-1], X.shape[-2]
    S = Xc @ Xc.transpose(-1, -2) / n
    alpha = (S ** 2).mean((-1, -2))
    mu = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1) / p
    num = alpha + mu ** 2
    den = (n + 1) * (alpha - mu ** 2 / p)
    shrink = torch.where(den == 0, torch.ones_like(den), torch.clamp(num / den, max=1.0))
    eye = torch.eye(p, dtype=X.dtype, device=X.device)
    return (1 - shrink)[..., None, None] * S + (shrink * mu)[..., None, None] * eye


def logm_spd(A):
    w, V = torch.linalg.eigh(A)
    return (V * torch.log(w)[..., None, :]) @ V.transpose(-1, -2)


def upper(A):
    p = A.shape[-1]
    i, j = torch.triu_indices(p, p, device=A.device)
    coef = torch.where(i == j, 1.0, 2 ** 0.5).to(A.dtype)
    return coef * A[..., i, j]


class TorchFBFingerprint(nn.Module):
    """p(person | window) from per-band covariances; buffers hold every parameter."""

    def __init__(self, filters, cref_isqrt, weight, bias):
        super().__init__()
        self.register_buffer("filters", torch.as_tensor(filters))          # (B, T, T)
        self.register_buffer("cref_isqrt", torch.as_tensor(cref_isqrt))    # (B, C, C)
        self.linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weight))
            self.linear.bias.copy_(torch.as_tensor(bias))

    def features(self, X):
        dt = self.filters.dtype
        X = X.to(dt)
        feats = []
        for M, W in zip(self.filters, self.cref_isqrt):
            C = oas(X @ M)
            feats.append(upper(logm_spd(W @ C @ W)))
        return torch.cat(feats, -1)

    def forward(self, X):
        """Logits (B, K)."""
        F = self.features(X)
        return self.linear(F.to(self.linear.weight.dtype))


def fit_export(X_tr, t_tr, bands, C=1.0, dtype=np.float64):
    """Fit with the reference pipeline, return (TorchFBFingerprint, sklearn predict_proba fn)."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.base import invsqrtm
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    T = X_tr.shape[-1]
    Ms, Ws, F_tr, pipes = [], [], [], []
    for band in bands:
        cov = Covariances("oas")
        C_tr = cov.fit_transform(riemann.bandpass(X_tr, band).astype(np.float64))
        ts = TangentSpace(metric="riemann").fit(C_tr)
        F_tr.append(ts.transform(C_tr))
        pipes.append((band, cov, ts))
        Ms.append(filter_matrix(band, T) if band is not None else np.eye(T))
        Ws.append(invsqrtm(ts.reference_))
    F_tr = np.hstack(F_tr)
    sc = StandardScaler().fit(F_tr)
    lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(F_tr), t_tr)
    weight = lr.coef_ / sc.scale_
    bias = lr.intercept_ - weight @ sc.mean_
    model = TorchFBFingerprint(np.stack(Ms).astype(dtype), np.stack(Ws).astype(dtype),
                               weight.astype(np.float32), bias.astype(np.float32))

    def reference_proba(X):
        F = np.hstack([ts.transform(cov.transform(riemann.bandpass(X, band).astype(np.float64)))
                       for band, cov, ts in pipes])
        return lr.predict_proba(sc.transform(F))

    return model.eval(), reference_proba
