"""Per-person mu-peak band centering for `RiemannExperts` (Protocol 02, Handoff §2c).

Estimates each participant k's mu/alpha peak frequency f_{mu,k} in [7.0, 14.0] Hz strictly from
their calibration windows (`0.5–4.0 s` crop) via 1/f-detrended DPSS multitaper power spectra,
and evaluates per-person passband centering both with static calibration references and on top of
Protocol 01's test-time reference adaptation (`ema_tangent_shuffled` and `session_fp_spd`).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import dpss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [
    str(REPO / "external" / "2026-competition"),
    str(REPO / "track2"),
    str(REPO / "research" / "integration-harness" / "src"),
    str(HERE),
]

import online_recenter  # noqa: E402
import riemann_parts  # noqa: E402
from submission import oas, tangent  # noqa: E402

PROTOCOL02_ARMS = (
    "fixed_8_30_static",
    "mu_shift_wide_static",
    "mu_lower_edge_static",
    "mu_beta_2band_static",
    "fixed_8_30_ema_shuf",
    "mu_shift_wide_ema_shuf",
    "mu_lower_edge_ema_shuf",
    "mu_beta_2band_ema_shuf",
    "mu_shift_wide_sess_spd",
)


def estimate_mu_peak(X_k: np.ndarray, sfreq: float, nw: float = 2.0, n_tapers: int = 3) -> float:
    """Estimate participant k's mu/alpha peak frequency in [7.0, 14.0] Hz from calibration windows."""
    a, b = (int(round(s * sfreq)) for s in riemann_parts.RIEMANN_WIN_S)
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


def _fit_single_band_per_person(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    X_te: np.ndarray,
    bands_per_k: list[tuple[float, float]],
    K: int,
    n_classes: int,
    sfreq: float,
) -> dict:
    """Fit per-person RiemannExperts where person k uses passband `bands_per_k[k]`."""
    a, b = (int(round(s * sfreq)) for s in riemann_parts.RIEMANN_WIN_S)
    N_te = len(X_te)
    n_chans = X_cal.shape[1]
    n_feat = n_chans * (n_chans + 1) // 2

    f_cal_te = np.zeros((N_te, K, n_feat), dtype=np.float64)
    C_te_per_k = np.zeros((K, N_te, n_chans, n_chans), dtype=np.float64)
    cref_cal = np.zeros((K, n_chans, n_chans), dtype=np.float64)
    weights, biases = [], []

    for k in range(K):
        sos = butter(4, list(bands_per_k[k]), btype="bandpass", fs=sfreq, output="sos")
        m = p_cal == k
        x_tr_k = sosfiltfilt(sos, X_cal[m], axis=-1)[:, :, a:b].astype(np.float64)
        cov_tr_k = Covariances("oas").fit_transform(x_tr_k)
        ts_k = TangentSpace(metric="riemann").fit(cov_tr_k)
        lr_k = LogisticRegression(C=riemann_parts.RIEMANN_C, max_iter=5000).fit(
            ts_k.transform(cov_tr_k), y_cal[m]
        )
        coef, icpt = lr_k.coef_.copy(), lr_k.intercept_.copy()
        if n_classes == 2:
            coef = np.vstack([np.zeros_like(coef), coef])
            icpt = np.array([0.0, icpt[0]])
        weights.append(coef)
        biases.append(icpt)
        cref_cal[k] = ts_k.reference_

        x_te_k = sosfiltfilt(sos, X_te, axis=-1)[:, :, a:b].astype(np.float64)
        cov_te_k = Covariances("oas").fit_transform(x_te_k)
        C_te_per_k[k] = cov_te_k
        f_cal_te[:, k, :] = ts_k.transform(cov_te_k)

    weight_t = torch.as_tensor(np.stack(weights), dtype=torch.float64)
    bias_t = torch.as_tensor(np.stack(biases), dtype=torch.float64)
    return {
        "f_cal_te": f_cal_te,
        "C_te_per_k": C_te_per_k,
        "cref_cal": cref_cal,
        "weight": weight_t,
        "bias": bias_t,
    }


def _fit_dual_band_per_person(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    X_te: np.ndarray,
    mu_peaks: list[float],
    K: int,
    n_classes: int,
    sfreq: float,
) -> dict:
    """Fit 2-band individualized [mu_k, beta_k] tangent-space LR per person k."""
    a, b = (int(round(s * sfreq)) for s in riemann_parts.RIEMANN_WIN_S)
    N_te = len(X_te)
    n_chans = X_cal.shape[1]
    n_feat = 2 * (n_chans * (n_chans + 1) // 2)

    f_cal_te = np.zeros((N_te, K, n_feat), dtype=np.float64)
    weights, biases = [], []

    for k in range(K):
        f_mu = mu_peaks[k]
        b_mu = (float(np.clip(f_mu - 2.5, 5.0, 11.0)), float(np.clip(f_mu + 2.5, 9.5, 16.5)))
        b_beta = (float(np.clip(f_mu + 2.5, 9.5, 16.5)), 30.0)
        m = p_cal == k

        tr_feats, te_feats = [], []
        for band in (b_mu, b_beta):
            sos = butter(4, list(band), btype="bandpass", fs=sfreq, output="sos")
            x_tr = sosfiltfilt(sos, X_cal[m], axis=-1)[:, :, a:b].astype(np.float64)
            cov_tr = Covariances("oas").fit_transform(x_tr)
            ts = TangentSpace(metric="riemann").fit(cov_tr)
            tr_feats.append(ts.transform(cov_tr))

            x_te = sosfiltfilt(sos, X_te, axis=-1)[:, :, a:b].astype(np.float64)
            cov_te = Covariances("oas").fit_transform(x_te)
            te_feats.append(ts.transform(cov_te))

        F_tr = np.hstack(tr_feats)
        F_te = np.hstack(te_feats)
        sc = StandardScaler().fit(F_tr)
        lr = LogisticRegression(C=riemann_parts.RIEMANN_C, max_iter=5000).fit(sc.transform(F_tr), y_cal[m])
        w = lr.coef_ / sc.scale_
        icpt = lr.intercept_ - w @ sc.mean_
        if n_classes == 2:
            w = np.vstack([np.zeros_like(w), w])
            icpt = np.array([0.0, icpt[0]])
        weights.append(w)
        biases.append(icpt)
        f_cal_te[:, k, :] = F_te

    return {
        "f_cal_te": f_cal_te,
        "weight": torch.as_tensor(np.stack(weights), dtype=torch.float64),
        "bias": torch.as_tensor(np.stack(biases), dtype=torch.float64),
    }


@torch.inference_mode()
def evaluate_mu_arms_on_split(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    X_te: np.ndarray,
    fp_logp: np.ndarray,
    K: int,
    n_classes: int,
    sfreq: float,
    alpha: float = 0.05,
    n0: float = 10.0,
    perm_seeds=(0, 1, 2),
) -> dict:
    """Evaluate all 9 Protocol 02 arms on a dataset split."""
    mu_peaks = [estimate_mu_peak(X_cal[p_cal == k], sfreq) for k in range(K)]

    bands_fixed = [(8.0, 30.0) for _ in range(K)]
    bands_wide = [
        (
            float(np.clip(8.0 + (f_mu - 10.0), 5.5, 11.0)),
            float(np.clip(30.0 + (f_mu - 10.0), 26.0, 34.0)),
        )
        for f_mu in mu_peaks
    ]
    bands_lower = [(float(np.clip(f_mu - 2.0, 5.5, 11.0)), 30.0) for f_mu in mu_peaks]

    m_fixed = _fit_single_band_per_person(X_cal, y_cal, p_cal, X_te, bands_fixed, K, n_classes, sfreq)
    m_wide = _fit_single_band_per_person(X_cal, y_cal, p_cal, X_te, bands_wide, K, n_classes, sfreq)
    m_lower = _fit_single_band_per_person(X_cal, y_cal, p_cal, X_te, bands_lower, K, n_classes, sfreq)
    m_dual = _fit_dual_band_per_person(X_cal, y_cal, p_cal, X_te, mu_peaks, K, n_classes, sfreq)

    N = len(X_te)
    gamma = np.exp(fp_logp).astype(np.float64)
    gamma = gamma / gamma.sum(axis=-1, keepdims=True)

    def _static_and_ema_shuf(mod_dict: dict):
        f_cal = mod_dict["f_cal_te"]
        w, b = mod_dict["weight"], mod_dict["bias"]
        lp_static = online_recenter._score_tangent(torch.from_numpy(f_cal), w, b)
        lp_ema_shuf = online_recenter._average_logp_over_permutations(
            lambda perm: online_recenter._score_tangent(
                torch.from_numpy(online_recenter._ema_tangent_seq(f_cal[perm], gamma[perm], alpha=alpha)), w, b
            ),
            N,
            perm_seeds=perm_seeds,
        )
        return lp_static, lp_ema_shuf

    lp_fix_st, lp_fix_em = _static_and_ema_shuf(m_fixed)
    lp_wid_st, lp_wid_em = _static_and_ema_shuf(m_wide)
    lp_low_st, lp_low_em = _static_and_ema_shuf(m_lower)
    lp_dua_st, lp_dua_em = _static_and_ema_shuf(m_dual)

    # Full-session SPD Fréchet mean adaptation on mu_shift_wide
    n_chans = X_cal.shape[1]
    n_feat = n_chans * (n_chans + 1) // 2
    f_wid_spd = torch.zeros((N, K, n_feat), dtype=torch.float64)
    for k in range(K):
        covs_k = np.concatenate([m_wide["cref_cal"][k : k + 1], m_wide["C_te_per_k"][k]], axis=0)
        weights_k = np.concatenate([[n0], gamma[:, k]])
        c_mean_k = mean_riemann(covs_k, sample_weight=weights_k)
        W_k = torch.from_numpy(invsqrtm(c_mean_k))
        C_k_t = torch.from_numpy(m_wide["C_te_per_k"][k])
        f_wid_spd[:, k, :] = tangent(C_k_t, W_k[None])
    lp_wid_spd = online_recenter._score_tangent(f_wid_spd, m_wide["weight"], m_wide["bias"])

    return {
        "mu_peaks": mu_peaks,
        "bands_wide": bands_wide,
        "bands_lower": bands_lower,
        "arms_logp": {
            "fixed_8_30_static": lp_fix_st,
            "mu_shift_wide_static": lp_wid_st,
            "mu_lower_edge_static": lp_low_st,
            "mu_beta_2band_static": lp_dua_st,
            "fixed_8_30_ema_shuf": lp_fix_em,
            "mu_shift_wide_ema_shuf": lp_wid_em,
            "mu_lower_edge_ema_shuf": lp_low_em,
            "mu_beta_2band_ema_shuf": lp_dua_em,
            "mu_shift_wide_sess_spd": lp_wid_spd,
        },
    }
