"""Training side of the frozen-REVE probe stream (loop C).

The encoder is frozen and loaded from the Hugging Face cache; only linear heads are trained:
a pooled multinomial head on all labelled windows, then one head per evaluation person,
fitted on their calibration windows with an L2 pull toward the pooled head.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

REVE_SFREQ = 200.0


def resample_matrix(n_times, sfreq_in, sfreq_out=REVE_SFREQ):
    """(n_times, n_out) matrix M with resample_poly(x, up, down) == x @ M."""
    from scipy.signal import resample_poly
    fr = Fraction(sfreq_out / sfreq_in).limit_denominator(1000)
    return np.ascontiguousarray(
        resample_poly(np.eye(n_times), fr.numerator, fr.denominator, axis=-1))


def positions(ch_names, bank_file):
    bank = json.loads(Path(bank_file).read_text())
    missing = [c for c in ch_names if c not in bank]
    if missing:
        raise ValueError(f"channels missing from the REVE position bank: {missing}")
    return np.asarray([bank[c] for c in ch_names], dtype=np.float64)


def embed(encoder, X, R, pos, batch=50):
    """Mean-pooled REVE features (n, 512) for windows X (n, C, T) at the data's rate."""
    from submission import standardize_clip
    Rt = torch.as_tensor(R, dtype=torch.float32)
    P = torch.as_tensor(pos, dtype=torch.float32)
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            x = standardize_clip(torch.as_tensor(X[i:i + batch], dtype=torch.float32) @ Rt)
            f = encoder(x, pos=P.expand(len(x), -1, -1), return_features=True)["features"]
            out.append(f.mean(dim=(1, 2)).float().numpy())
    return np.concatenate(out)


def _fit(Z, y, n_classes, lam, W0, b0):
    Z = torch.as_tensor(Z, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.long)
    D = Z.shape[1]
    W0 = torch.zeros(n_classes, D, dtype=torch.float64) if W0 is None else torch.as_tensor(W0)
    b0 = torch.zeros(n_classes, dtype=torch.float64) if b0 is None else torch.as_tensor(b0)
    W = W0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Z @ W.T + b, y) \
            + 0.5 * lam * (((W - W0) ** 2).sum() + ((b - b0) ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def fit_head(Z, y, n_classes, lam, W0=None, b0=None):
    """Multinomial logistic head minimising mean NLL + lam/2 · ||θ − θ0||²."""
    return _fit(Z, y, n_classes, lam, W0, b0)


def fit_person_heads(Z, y, person, n_people, n_classes, W0, b0, lam):
    W, b = [], []
    for k in range(n_people):
        m = person == k
        Wk, bk = _fit(Z[m], y[m], n_classes, lam, W0, b0)
        W.append(Wk)
        b.append(bk)
    return np.stack(W), np.stack(b)


def export(R, pos, mu, sd, W, b):
    """State dict for submission.ReveProbe. Heads act on (z − mu) / sd embeddings."""
    f32, f64 = torch.float32, torch.float64
    return {"resample": torch.as_tensor(R, dtype=f32),
            "pos": torch.as_tensor(pos, dtype=f32),
            "mu": torch.as_tensor(mu, dtype=f64), "sd": torch.as_tensor(sd, dtype=f64),
            "weight": torch.as_tensor(W, dtype=f64), "bias": torch.as_tensor(b, dtype=f64)}
