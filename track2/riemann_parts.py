"""Fit the covariance-based parts of the Track 2 mixture and export them as tensors.

Training side only (scipy / pyriemann / sklearn). The exported state dicts load
into submission.py's torch modules, which reproduce these pipelines exactly:

    fit_complex_fingerprint   6-band complex Hermitian analytic filter-bank fingerprint
                              (PR #15 / NeuralPrint), folded Butterworth + Hilbert,
                              trace-shrunk Hermitian covariance -> complex tangent space ->
                              standardised multinomial logistic regression (C = 1)
    fit_fingerprint           dispatcher for complex_6b (default) or legacy fb_riemann
    estimate_mu_peak          robust 1/f-detrended DPSS multitaper peak estimator in [7, 14] Hz
    fit_riemann               per-participant experts: default 'mu_beta_2band' (individualized
                              mu and beta passbands split at individual mu peak, 0.5-4 s crop,
                              raw tangent vectors fitted with LR C = 0.1), or legacy 'fixed_8_30'
    C3                        combiner coefficients
"""

import numpy as np
import scipy.signal as sps
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import dpss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from pyriemann.geometry.base import invsqrtm
except ImportError:
    from pyriemann.utils.base import invsqrtm
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from submission import complex_tangent, hermitian_cov

FP_BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
RIEMANN_BAND, RIEMANN_WIN_S, RIEMANN_C = (8, 30), (0.5, 4.0), 0.1
# Dreyer cross-fitted dev bank, combiner C3 (a, c0, c1; class bias for class 1)
C3 = {"a": 0.80986932, "c0": 0.96004621, "c1": 1.70110631, "b": -0.01604924}
RHO = 0.05
MAX_ITERS_FRECHET = 50
TOL_FRECHET = 1e-5
EPS_FRECHET = 0.5


def _sos(band, sfreq):
    return butter(4, list(band), btype="bandpass", fs=sfreq, output="sos")


def _covs(X):
    return Covariances("oas").fit_transform(X)


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


def _fold_lr(sc: StandardScaler, lr: LogisticRegression):
    weight = lr.coef_ / sc.scale_
    bias = lr.intercept_ - weight @ sc.mean_
    if weight.shape[0] == 1:           # sklearn binary: one logit for class 1 -> logits [0, z]
        weight = np.vstack([np.zeros_like(weight), weight])
        bias = np.concatenate([[0.0], bias])
    return weight, bias


def fit_complex_fingerprint(X, person, sfreq, bands=FP_BANDS, C=1.0, rho=RHO):
    """Fit complex Hermitian filter-bank fingerprint on calibration windows X (winning design)."""
    T = X.shape[-1]
    filters, crefs_isqrt, cal_feats = [], [], []
    for band in bands:
        sos = _sos(band, sfreq)
        Xb = sosfiltfilt(sos, X, axis=-1).astype(np.float32)
        Zb = torch.from_numpy(sps.hilbert(Xb, axis=-1).astype(np.complex64))
        covs = hermitian_cov(Zb, rho=rho)
        ref = complex_frechet_mean(covs)
        w_isqrt = herm_invsqrtm(ref)
        cal_feats.append(complex_tangent(covs, w_isqrt).numpy())

        m_real = sosfiltfilt(sos, np.eye(T), axis=-1).astype(np.float32)
        f_cplx = sps.hilbert(m_real, axis=-1).astype(np.complex64)
        filters.append(torch.from_numpy(f_cplx))
        crefs_isqrt.append(w_isqrt.to(torch.complex64))

    F_cal = np.hstack(cal_feats).astype(np.float64)
    sc = StandardScaler().fit(F_cal)
    lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(F_cal), person)
    weight, bias = _fold_lr(sc, lr)
    return {
        "filters": torch.stack(filters).to(torch.complex64),
        "cref_isqrt": torch.stack(crefs_isqrt).to(torch.complex64),
        "linear.weight": torch.as_tensor(weight, dtype=torch.float32),
        "linear.bias": torch.as_tensor(bias, dtype=torch.float32),
    }


def fit_fingerprint(X, person, sfreq, bands=FP_BANDS, C=1.0, kind="fb_riemann"):
    """Fit fingerprint covariance model. 'fb_riemann' for FBFingerprint, 'complex_6b' for ComplexFBFingerprint."""
    if kind in ("complex_6b", "complex"):
        return fit_complex_fingerprint(X, person, sfreq, bands=bands, C=C)
    # Legacy real OAS filter-bank
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
    weight, bias = _fold_lr(sc, lr)
    f32 = torch.float32
    return {
        "filters": torch.as_tensor(np.stack(filters), dtype=f32),
        "cref_isqrt": torch.as_tensor(np.stack(crefs), dtype=f32),
        "linear.weight": torch.as_tensor(weight, dtype=f32),
        "linear.bias": torch.as_tensor(bias, dtype=f32),
    }


def estimate_mu_peak(X_k: np.ndarray, sfreq: float, nw: float = 2.0, n_tapers: int = 3) -> float:
    """Estimate participant k's mu/alpha peak frequency in [7.0, 14.0] Hz from calibration windows."""
    a, b = (int(round(s * sfreq)) for s in RIEMANN_WIN_S)
    x = np.asarray(X_k[:, :, a:b], dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    T = x.shape[-1]
    tapers = dpss(T, nw, n_tapers)
    freqs = np.fft.rfftfreq(T, 1.0 / sfreq)
    spec = np.fft.rfft(x[:, :, None, :] * tapers[None, None, :, :], axis=-1)
    P_mean = np.mean(np.abs(spec) ** 2, axis=(0, 1, 2))  # (F,)

    flank = ((freqs >= 2.0) & (freqs <= 6.0)) | ((freqs >= 30.0) & (freqs <= 45.0))
    log_f = np.log10(np.clip(freqs[flank], 1e-6, None))
    log_p = np.log10(np.clip(P_mean[flank], 1e-18, None))
    A = np.vstack([np.ones_like(log_f), -log_f]).T
    b_est, chi_est = np.linalg.lstsq(A, log_p, rcond=None)[0]

    mu_mask = (freqs >= 7.0) & (freqs <= 14.0)
    idx_mu = np.where(mu_mask)[0]
    if len(idx_mu) == 0:
        return 10.0

    log_p_all = np.log10(np.clip(P_mean, 1e-18, None))
    aperiodic_all = b_est - chi_est * np.log10(np.clip(freqs, 1e-6, None))
    resid = log_p_all - aperiodic_all

    rel_idx = int(np.argmax(resid[idx_mu]))
    if float(resid[idx_mu[rel_idx]]) <= 0.05:
        return 10.0

    peak_i = int(idx_mu[rel_idx])
    if 0 < peak_i < len(freqs) - 1:
        y0, y1, y2 = resid[peak_i - 1], resid[peak_i], resid[peak_i + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            df = float(freqs[1] - freqs[0])
            return float(np.clip(freqs[peak_i] + delta * df, 7.0, 14.0))
    return float(np.clip(freqs[peak_i], 7.0, 14.0))


class _Riemann:
    """One participant's tangent-space logistic regression (legacy 1-band)."""

    def __init__(self, sfreq):
        self.sos = _sos(RIEMANN_BAND, sfreq)
        self.a, self.b = (int(round(s * sfreq)) for s in RIEMANN_WIN_S)

    def _x(self, X):
        return sosfiltfilt(self.sos, X, axis=-1)[:, :, self.a:self.b].astype(np.float64)

    def fit(self, X, y):
        cov = _covs(self._x(X))
        self.ts = TangentSpace(metric="riemann").fit(cov)
        self.lr = LogisticRegression(C=RIEMANN_C, max_iter=5000).fit(self.ts.transform(cov), y)
        return self

    def predict(self, X):
        return self.lr.predict(self.ts.transform(_covs(self._x(X))))


def fit_riemann(X, y, person, run, n_people, n_classes, sfreq, bands_mode="mu_beta_2band"):
    """State dicts for submission.RiemannExperts and LogLinearCombiner.

    Supports:
    - bands_mode='mu_beta_2band' (winning design): individualized [mu_k, beta_k] 2-band
    - bands_mode='fixed_8_30' (legacy single-band): fixed 8-30 Hz passband
    """
    T = X.shape[-1]
    a, b_win = (int(round(s * sfreq)) for s in RIEMANN_WIN_S)
    n_out = b_win - a
    f64 = torch.float64

    if bands_mode == "fixed_8_30":
        ref = _Riemann(sfreq)
        crefs, weights, biases, rel = [], [], [], []
        for k in range(n_people):
            m = person == k
            model = _Riemann(sfreq).fit(X[m], y[m])
            crefs.append(invsqrtm(model.ts.reference_))
            coef, icpt = model.lr.coef_.copy(), model.lr.intercept_.copy()
            if n_classes == 2:
                coef = np.vstack([np.zeros_like(coef), coef])
                icpt = np.array([0.0, icpt[0]])
            weights.append(coef)
            biases.append(icpt)
            units = np.unique(run[m])
            accs = []
            for u1 in units:
                fa = _Riemann(sfreq).fit(X[m & (run == u1)], y[m & (run == u1)])
                for u2 in units:
                    if u2 != u1:
                        mb = m & (run == u2)
                        accs.append(balanced_accuracy_score(y[mb], fa.predict(X[mb])))
            rel.append(np.mean(accs) if accs else 0.5)
        rel = np.asarray(rel)
        rel_norm = 0.5 + 0.5 * (rel - 1 / n_classes) / (1 - 1 / n_classes)
        filt = np.ascontiguousarray(sosfiltfilt(ref.sos, np.eye(T), axis=-1)[:, ref.a:ref.b])
        n_out = filt.shape[1]
        riemann = {
            "filt": torch.as_tensor(filt, dtype=f64),
            "cref_isqrt": torch.as_tensor(np.stack(crefs), dtype=f64),
            "weight": torch.as_tensor(np.stack(weights), dtype=f64),
            "bias": torch.as_tensor(np.stack(biases), dtype=f64),
        }
    elif bands_mode == "mu_beta_2band":
        filters_all = []  # (K, 2, T, n_out)
        crefs_all = []    # (K, 2, C, C)
        weights, biases, rel = [], [], []

        for k in range(n_people):
            m = person == k
            X_k = X[m]
            y_k = y[m]
            run_k = run[m]
            f_mu = estimate_mu_peak(X_k, sfreq)
            b_mu = (float(np.clip(f_mu - 2.5, 5.0, 11.0)), float(np.clip(f_mu + 2.5, 9.5, 16.5)))
            b_beta = (float(np.clip(f_mu + 2.5, 9.5, 16.5)), 30.0)

            k_filters = []
            k_crefs = []
            tr_feats = []
            soses = [_sos(b_mu, sfreq), _sos(b_beta, sfreq)]

            for sos in soses:
                filt_mat = np.ascontiguousarray(sosfiltfilt(sos, np.eye(T), axis=-1)[:, a:b_win])
                k_filters.append(filt_mat)

                x_tr = sosfiltfilt(sos, X_k, axis=-1)[:, :, a:b_win].astype(np.float64)
                cov_tr = _covs(x_tr)
                ts = TangentSpace(metric="riemann").fit(cov_tr)
                k_crefs.append(invsqrtm(ts.reference_))
                tr_feats.append(ts.transform(cov_tr))

            F_tr = np.hstack(tr_feats)
            lr = LogisticRegression(C=RIEMANN_C, max_iter=5000).fit(F_tr, y_k)
            coef, icpt = lr.coef_.copy(), lr.intercept_.copy()
            if n_classes == 2:
                coef = np.vstack([np.zeros_like(coef), coef])
                icpt = np.array([0.0, icpt[0]])
            weights.append(coef)
            biases.append(icpt)
            filters_all.append(np.stack(k_filters))
            crefs_all.append(np.stack(k_crefs))

            units = np.unique(run_k)
            accs = []
            for u1 in units:
                m_u1 = run_k == u1
                if len(np.unique(y_k[m_u1])) < 2:
                    continue
                f_u1 = []
                for sos in soses:
                    x_u1 = sosfiltfilt(sos, X_k[m_u1], axis=-1)[:, :, a:b_win].astype(np.float64)
                    cov_u1 = _covs(x_u1)
                    ts_u1 = TangentSpace(metric="riemann").fit(cov_u1)
                    f_u1.append((ts_u1, ts_u1.transform(cov_u1)))
                F_u1 = np.hstack([f[1] for f in f_u1])
                lr_u1 = LogisticRegression(C=RIEMANN_C, max_iter=5000).fit(F_u1, y_k[m_u1])

                for u2 in units:
                    if u2 != u1:
                        m_u2 = run_k == u2
                        if len(np.unique(y_k[m_u2])) < 2:
                            continue
                        f_u2 = []
                        for i, sos in enumerate(soses):
                            x_u2 = sosfiltfilt(sos, X_k[m_u2], axis=-1)[:, :, a:b_win].astype(np.float64)
                            cov_u2 = _covs(x_u2)
                            f_u2.append(f_u1[i][0].transform(cov_u2))
                        F_u2 = np.hstack(f_u2)
                        p_u2 = lr_u1.predict(F_u2)
                        accs.append(balanced_accuracy_score(y_k[m_u2], p_u2))
            rel.append(np.mean(accs) if accs else 0.5)

        rel = np.asarray(rel)
        rel_norm = 0.5 + 0.5 * (rel - 1 / n_classes) / (1 - 1 / n_classes)
        stacked_filters = np.stack(filters_all)
        n_out = stacked_filters.shape[-1]
        riemann = {
            "filters": torch.as_tensor(stacked_filters, dtype=f64),
            "cref_isqrt": torch.as_tensor(np.stack(crefs_all), dtype=f64),
            "weight": torch.as_tensor(np.stack(weights), dtype=f64),
            "bias": torch.as_tensor(np.stack(biases), dtype=f64),
        }
    else:
        raise ValueError(f"unknown bands_mode {bands_mode!r}")

    class_bias = np.zeros(n_classes)
    if n_classes == 2:
        class_bias[1] = C3["b"]
    combiner = {
        "coef": torch.tensor([C3["a"], C3["c0"], C3["c1"]], dtype=f64),
        "class_bias": torch.as_tensor(class_bias, dtype=f64),
        "rel": torch.as_tensor(rel_norm, dtype=f64),
    }
    return riemann, combiner, rel, n_out

