"""Complex Hermitian and 5/6-band filter-bank covariance fingerprints for Track 2.

Implements:
- Real OAS filter-bank fingerprint (6-band and 5-band without delta) via track2/riemann_parts.py.
- Complex Hermitian analytic-signal filter-bank fingerprint (6-band and 5-band) ported from
  NeuralPrint (`experiments/h-complex-hermitian-tangent/code/run_experiment.py`), plus its
  pure-PyTorch inference module `ComplexFBFingerprint`.
- Estimator-matched real Hermitian control (`herm_real_6b` / `herm_real_5b`).
"""

import numpy as np
import scipy.signal as sps
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

BANDS_6 = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
BANDS_5 = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
RHO = 0.05
MAX_ITERS_FRECHET = 50
TOL_FRECHET = 1e-5
EPS_FRECHET = 0.5


def hermitian_cov(Z: torch.Tensor, rho: float = RHO) -> torch.Tensor:
    """Trace-shrunk Hermitian sample covariance of analytic signals Z (..., C, T)."""
    n_times, n_chans = Z.shape[-1], Z.shape[-2]
    gamma = (Z @ Z.mH) / n_times
    tr = torch.diagonal(gamma.real, dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(n_chans, dtype=Z.dtype, device=Z.device)
    return (1.0 - rho) * gamma + rho * (tr / n_chans)[..., None, None] * eye


def herm_invsqrtm(H: torch.Tensor) -> torch.Tensor:
    """Inverse square root H^{-1/2} of Hermitian / symmetric positive-definite matrix H."""
    w, V = torch.linalg.eigh(H)
    w = torch.clamp(w, min=1e-12)
    return (V / torch.sqrt(w).unsqueeze(-2)) @ V.mH


def complex_frechet_mean(
    covs_hpd: torch.Tensor,
    max_iter: int = MAX_ITERS_FRECHET,
    tol: float = TOL_FRECHET,
    eps: float = EPS_FRECHET,
) -> torch.Tensor:
    """Riemannian Fréchet geometric mean on the complex HPD manifold H_{++}^C."""
    ref = covs_hpd.mean(dim=0)
    for _ in range(max_iter):
        w, V = torch.linalg.eigh(ref)
        w = torch.clamp(w, min=1e-12)
        ref_isqrt = (V / torch.sqrt(w).unsqueeze(-2)) @ V.mH
        ref_sqrt = (V * torch.sqrt(w).unsqueeze(-2)) @ V.mH

        M = ref_isqrt @ covs_hpd @ ref_isqrt
        wm, Vm = torch.linalg.eigh(M)
        wm = torch.clamp(wm, min=1e-12)
        logM = (Vm * torch.log(wm).unsqueeze(-2)) @ Vm.mH

        Vbar = logM.mean(dim=0)
        wv, Vv = torch.linalg.eigh(Vbar)
        expV = (Vv * torch.exp(eps * wv).unsqueeze(-2)) @ Vv.mH
        ref_new = ref_sqrt @ expV @ ref_sqrt

        diff = torch.linalg.norm(ref_new - ref)
        ref = ref_new
        if diff < tol:
            break
    return ref


def complex_tangent(C_hpd: torch.Tensor, cref_isqrt: torch.Tensor) -> torch.Tensor:
    """Project C_hpd (..., C, C) to tangent space at cref_isqrt = Cref^{-1/2} and vectorize
    into C^2 real coordinates: [diag(Re(L)), sqrt(2)*triu_1(Re(L)), sqrt(2)*triu_1(Im(L))]."""
    M = cref_isqrt @ C_hpd @ cref_isqrt
    w, V = torch.linalg.eigh(M)
    w = torch.clamp(w, min=1e-12)
    L = (V * torch.log(w).unsqueeze(-2)) @ V.mH
    n_chans = L.shape[-1]
    triu_i, triu_j = torch.triu_indices(n_chans, n_chans, offset=1, device=L.device)
    s2 = 2.0 ** 0.5
    diag = torch.diagonal(L.real, dim1=-2, dim2=-1)
    re_off = L.real[..., triu_i, triu_j] * s2
    im_off = L.imag[..., triu_i, triu_j] * s2
    return torch.cat([diag, re_off, im_off], dim=-1)


def real_tangent(C_spd: torch.Tensor, cref_isqrt: torch.Tensor) -> torch.Tensor:
    """Project real SPD matrix C_spd (..., C, C) to tangent space at cref_isqrt and vectorize
    into C(C+1)/2 coordinates: [diag(L), sqrt(2)*triu_1(L)]."""
    M = cref_isqrt @ C_spd @ cref_isqrt
    w, V = torch.linalg.eigh(M)
    w = torch.clamp(w, min=1e-12)
    L = (V * torch.log(w).unsqueeze(-2)) @ V.transpose(-1, -2)
    n_chans = L.shape[-1]
    triu_i, triu_j = torch.triu_indices(n_chans, n_chans, offset=1, device=L.device)
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    off = L[..., triu_i, triu_j] * (2.0 ** 0.5)
    return torch.cat([diag, off], dim=-1)


def folded_analytic_filter(band, sfreq: float, n_times: int) -> torch.Tensor:
    """Single complex64 (T, T) matrix combining zero-phase Butterworth band-pass and Hilbert."""
    sos = sps.butter(4, list(band), btype="bandpass", fs=sfreq, output="sos")
    m_real = sps.sosfiltfilt(sos, np.eye(n_times), axis=-1).astype(np.float32)
    f_cplx = sps.hilbert(m_real, axis=-1).astype(np.complex64)
    return torch.from_numpy(f_cplx)


class ComplexFBFingerprint(nn.Module):
    """Pure-PyTorch complex Hermitian filter-bank fingerprint.

    Each band applies a folded complex64 (T, T) filter+Hilbert matrix, computes the
    trace-shrunk Hermitian covariance, maps to the complex tangent space (C^2 real features
    per band), and applies a folded linear layer."""

    def __init__(self, n_bands: int, n_chans: int, n_times: int, n_subjects: int, rho: float = RHO):
        super().__init__()
        self.rho = float(rho)
        n_feat = n_bands * n_chans * n_chans
        self.register_buffer("filters", torch.zeros(n_bands, n_times, n_times, dtype=torch.complex64))
        self.register_buffer("cref_isqrt", torch.zeros(n_bands, n_chans, n_chans, dtype=torch.complex64))
        self.linear = nn.Linear(n_feat, n_subjects)

    def features(self, X: torch.Tensor) -> torch.Tensor:
        Xc = X.to(self.filters.dtype)
        feats = [
            complex_tangent(hermitian_cov(Xc @ F, rho=self.rho), W)
            for F, W in zip(self.filters, self.cref_isqrt)
        ]
        return torch.cat(feats, dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(X).to(self.linear.weight.dtype))


def _fold_lr(sc: StandardScaler, lr: LogisticRegression):
    weight = lr.coef_ / sc.scale_
    bias = lr.intercept_ - weight @ sc.mean_
    if weight.shape[0] == 1:  # binary sklearn -> 2 logits [0, z]
        weight = np.vstack([np.zeros_like(weight), weight])
        bias = np.concatenate([[0.0], bias])
    return weight, bias


def _band_analytic_covs(X: np.ndarray, band, sfreq: float, rho: float = RHO) -> torch.Tensor:
    sos = sps.butter(4, list(band), btype="bandpass", fs=sfreq, output="sos")
    Xb = sps.sosfiltfilt(sos, X, axis=-1).astype(np.float32)
    Zb = torch.from_numpy(sps.hilbert(Xb, axis=-1).astype(np.complex64))
    return hermitian_cov(Zb, rho=rho)


def fit_complex_fingerprint(X: np.ndarray, person: np.ndarray, sfreq: float, bands=BANDS_6, C: float = 1.0, rho: float = RHO):
    """Fit complex Hermitian filter-bank fingerprint on calibration windows X.

    Returns (state_dict, ref_predict_logp_fn) where state_dict loads into ComplexFBFingerprint
    and ref_predict_logp_fn(X_eval) runs the exact SciPy + sklearn reference pipeline."""
    n_times = X.shape[-1]
    filters, crefs_isqrt, cal_feats = [], [], []
    for band in bands:
        covs = _band_analytic_covs(X, band, sfreq, rho=rho)
        ref = complex_frechet_mean(covs)
        w_isqrt = herm_invsqrtm(ref)
        cal_feats.append(complex_tangent(covs, w_isqrt).numpy())
        filters.append(folded_analytic_filter(band, sfreq, n_times))
        crefs_isqrt.append(w_isqrt.to(torch.complex64))

    F_cal = np.hstack(cal_feats).astype(np.float64)
    sc = StandardScaler().fit(F_cal)
    lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(F_cal), person)
    weight, bias = _fold_lr(sc, lr)

    state = {
        "filters": torch.stack(filters).to(torch.complex64),
        "cref_isqrt": torch.stack(crefs_isqrt).to(torch.complex64),
        "linear.weight": torch.as_tensor(weight, dtype=torch.float32),
        "linear.bias": torch.as_tensor(bias, dtype=torch.float32),
    }

    def ref_predict_logp(X_eval: np.ndarray) -> np.ndarray:
        eval_feats = []
        for band, w_isqrt in zip(bands, crefs_isqrt):
            covs_te = _band_analytic_covs(X_eval, band, sfreq, rho=rho)
            eval_feats.append(complex_tangent(covs_te, w_isqrt).numpy())
        F_te = np.hstack(eval_feats).astype(np.float64)
        return lr.predict_log_proba(sc.transform(F_te))

    return state, ref_predict_logp


def fit_herm_real_control(X: np.ndarray, person: np.ndarray, sfreq: float, bands=BANDS_6, C: float = 1.0, rho: float = RHO):
    """Estimator-matched real control (Arm 2 in NeuralPrint): Re(Gamma_reg) whitened by Re(Gamma_bar)."""
    crefs_real_isqrt, cal_feats = [], []
    for band in bands:
        covs = _band_analytic_covs(X, band, sfreq, rho=rho)
        ref = complex_frechet_mean(covs)
        w_r, V_r = torch.linalg.eigh(ref.real)
        w_r = torch.clamp(w_r, min=1e-12)
        ref_r_isqrt = (V_r / torch.sqrt(w_r).unsqueeze(-2)) @ V_r.transpose(-1, -2)
        cal_feats.append(real_tangent(covs.real, ref_r_isqrt).numpy())
        crefs_real_isqrt.append(ref_r_isqrt)

    F_cal = np.hstack(cal_feats).astype(np.float64)
    sc = StandardScaler().fit(F_cal)
    lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(F_cal), person)

    def predict_logp(X_eval: np.ndarray) -> np.ndarray:
        eval_feats = []
        for band, ref_r_isqrt in zip(bands, crefs_real_isqrt):
            covs_te = _band_analytic_covs(X_eval, band, sfreq, rho=rho)
            eval_feats.append(real_tangent(covs_te.real, ref_r_isqrt).numpy())
        F_te = np.hstack(eval_feats).astype(np.float64)
        return lr.predict_log_proba(sc.transform(F_te))

    return predict_logp
