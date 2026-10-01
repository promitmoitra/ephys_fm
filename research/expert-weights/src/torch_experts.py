"""Per-person tangent-space experts + the C3 combiner as pure torch (shippable).

Training uses classical.py's scipy / pyriemann / sklearn pipeline; `export` turns every
person's fitted model into tensors:

  band filter    zero-phase sosfiltfilt on the fixed 480-sample window, then the time crop,
                 is linear in the input: one (T, T') matrix shared by all people
  covariance     OAS (sklearn.covariance.oas: centred, 1/n)
  tangent space  logm(Cref^-1/2 C Cref^-1/2), upper triangle with sqrt(2) off-diagonal
                 weights (pyriemann TangentSpace, metric riemann), per-person reference
  classifier     binary logistic regression: logit = w_k · f + b_k (no scaler for ts_C0.1)

C3 combines, per person k:
  z = a·log p_eeg + (c0 + c1·(rel_k − 0.5))·log p_ts + [0, b]
"""

import numpy as np
import torch
from torch import nn


def oas(X):
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


class TangentExperts(nn.Module):
    """K per-person binary tangent-space experts; forward -> log-probs (B, K, 2)."""

    def __init__(self, filt, cref_isqrt, weight, bias):
        super().__init__()
        self.register_buffer("filt", torch.as_tensor(filt))                # (T, T')
        self.register_buffer("cref_isqrt", torch.as_tensor(cref_isqrt))    # (K, C, C)
        self.register_buffer("weight", torch.as_tensor(weight))            # (K, F)
        self.register_buffer("bias", torch.as_tensor(bias))                # (K,)

    def forward(self, X):
        C = oas(X.to(self.filt.dtype) @ self.filt)                          # (B, C, C)
        W = self.cref_isqrt[None]                                           # (1, K, C, C)
        f = upper(logm_spd(W @ C[:, None] @ W))                             # (B, K, F)
        logit = (f * self.weight).sum(-1) + self.bias                      # (B, K)
        return torch.log_softmax(torch.stack([torch.zeros_like(logit), logit], -1), -1)


class C3Combiner(nn.Module):
    def __init__(self, a, c0, c1, b, rel):
        super().__init__()
        self.register_buffer("coef", torch.tensor([a, c0, c1, b], dtype=torch.float64))
        self.register_buffer("rel", torch.as_tensor(rel, dtype=torch.float64))   # (K,)

    def forward(self, logp_eeg, logp_ts):
        """(B, K, 2) each -> combined log-probs (B, K, 2)."""
        a, c0, c1, b = self.coef
        wc = (c0 + c1 * (self.rel - 0.5))[None, :, None]
        z = a * logp_eeg.to(wc.dtype) + wc * logp_ts.to(wc.dtype)
        z = z + torch.stack([torch.zeros_like(b), b])
        return torch.log_softmax(z, -1)


def export(E, people, variant, train_runs, n_times):
    """Fit every person's classical expert on train_runs (as classical.py) and export."""
    from pyriemann.utils.base import invsqrtm
    from scipy.signal import butter, sosfiltfilt
    from classical import FBTangent, SFREQ, VARIANTS, _banded
    bands, win, C, scale = VARIANTS[variant]
    assert len(bands) == 1 and not scale, "export supports single-band, unscaled variants"
    sos = butter(4, list(bands[0]), btype="bandpass", fs=SFREQ, output="sos")
    a, b = int(win[0] * SFREQ), int(win[1] * SFREQ)
    filt = np.ascontiguousarray(sosfiltfilt(sos, np.eye(n_times), axis=-1)[:, a:b])  # x @ filt
    Xb = _banded(E["X"], bands, win)
    crefs, ws, bs, models = [], [], [], []
    for s in people:
        m = (E["subject"] == s) & np.isin(E["run"], train_runs)
        model = FBTangent(bands, C, scale).fit([x[m] for x in Xb], E["y"][m])
        crefs.append(invsqrtm(model.ts[0].reference_))
        ws.append(model.lr.coef_[0])
        bs.append(model.lr.intercept_[0])
        models.append(model)
    return TangentExperts(filt, np.stack(crefs), np.stack(ws), np.array(bs)), models
