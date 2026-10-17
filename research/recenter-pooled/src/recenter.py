"""Fingerprint-routed reference-covariance re-centering for Track 2 task decoding (§2a).

Implements:
- Level 1 standalone tangent-space task decoders:
  - `ts_global_pooled`: 1 global reference covariance + 1 pooled LR (8–30 Hz, 0.5–4 s, C=0.1).
  - `ts_recenter_routed` / `ts_recenter_oracle` / `ts_recenter_whitener`: per-person calibration
    reference covariance W_k = Cref_k^{-1/2} + 1 shared pooled LR (8–30 Hz, 0.5–4 s, C=0.1),
    plus pure-PyTorch module `RecenterRiemann`.
  - `ts_recenter_fb_routed` / `ts_recenter_fb_oracle`: 5-band (4–45 Hz, 0.5–4 s) per-person
    reference re-centering + StandardScaler + 1 shared pooled LR (C=0.1).
  - `ts_per_person_routed` / `ts_per_person_oracle`: shipped per-person `RiemannExperts` (FM-2
    per-person retraining baseline on the same calibration data).
- Level 2 Track 2 mixture integration arms (`E1`, `E2`, `E3` and their oracle ceilings).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [
    str(REPO / "external" / "2026-competition"),
    str(REPO / "track2"),
    str(REPO / "research" / "integration-harness" / "src"),
]

import metrics  # noqa: E402
import riemann_parts  # noqa: E402
from submission import LogLinearCombiner, oas, tangent  # noqa: E402

BANDS_5 = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]
RIEMANN_BAND = riemann_parts.RIEMANN_BAND
RIEMANN_WIN_S = riemann_parts.RIEMANN_WIN_S
RIEMANN_C = riemann_parts.RIEMANN_C


class RecenterRiemann(nn.Module):
    """Pure-PyTorch pooled task decoder on per-participant re-centered tangent space (float64).

    Stores K per-participant reference whiteners Cref_k^{-1/2} and ONE shared linear classifier
    (weight, bias) across all K participants."""

    def __init__(self, n_subjects: int, n_chans: int, n_times: int, n_times_out: int, n_classes: int):
        super().__init__()
        n_feat = n_chans * (n_chans + 1) // 2
        d = torch.float64
        self.register_buffer("filt", torch.zeros(n_times, n_times_out, dtype=d))
        self.register_buffer("cref_isqrt", torch.zeros(n_subjects, n_chans, n_chans, dtype=d))
        self.register_buffer("weight", torch.zeros(n_classes, n_feat, dtype=d))
        self.register_buffer("bias", torch.zeros(n_classes, dtype=d))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Per-participant class log-probabilities (B, K, C) using shared (weight, bias)."""
        C = oas(X.to(self.filt.dtype) @ self.filt)              # (B, c, c)
        f = tangent(C[:, None], self.cref_isqrt[None])          # (B, K, F)
        logits = torch.einsum("bkf,cf->bkc", f, self.weight) + self.bias
        return torch.log_softmax(logits, dim=-1)

    def forward_whitener(self, X: torch.Tensor, fp_logp: torch.Tensor) -> torch.Tensor:
        """Single-pass whitener interpolation: W_bar(x) = sum_k p(k|x) Cref_k^{-1/2} -> (B, C)."""
        C = oas(X.to(self.filt.dtype) @ self.filt)              # (B, c, c)
        p = torch.exp(fp_logp.to(self.filt.dtype))              # (B, K)
        W_bar = torch.einsum("bk,kij->bij", p, self.cref_isqrt) # (B, c, c)
        f = tangent(C, W_bar)                                   # (B, F)
        logits = f @ self.weight.T + self.bias
        return torch.log_softmax(logits, dim=-1)


def _crop_filter(band, sfreq: float, n_times: int):
    sos = butter(4, list(band), btype="bandpass", fs=sfreq, output="sos")
    a, b = (int(round(s * sfreq)) for s in RIEMANN_WIN_S)
    filt = np.ascontiguousarray(sosfiltfilt(sos, np.eye(n_times), axis=-1)[:, a:b])
    return sos, a, b, filt


def _lr_weights(lr: LogisticRegression, n_classes: int):
    coef, icpt = lr.coef_.copy(), lr.intercept_.copy()
    if n_classes == 2:
        coef = np.vstack([np.zeros_like(coef), coef])
        icpt = np.array([0.0, icpt[0]])
    return coef, icpt


def fit_global_pooled_8_30(X_train: np.ndarray, y_train: np.ndarray, n_classes: int, sfreq: float):
    """Uncentered pooled baseline: 1 global Fréchet mean + 1 pooled LR (8–30 Hz, 0.5–4 s)."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    sos, a, b, _ = _crop_filter(RIEMANN_BAND, sfreq, X_train.shape[-1])
    x_tr = sosfiltfilt(sos, X_train, axis=-1)[:, :, a:b].astype(np.float64)
    cov_tr = Covariances("oas").fit_transform(x_tr)
    ts = TangentSpace(metric="riemann").fit(cov_tr)
    lr = LogisticRegression(C=RIEMANN_C, max_iter=5000).fit(ts.transform(cov_tr), y_train)

    def predict_logp(X_eval: np.ndarray) -> np.ndarray:
        x_te = sosfiltfilt(sos, X_eval, axis=-1)[:, :, a:b].astype(np.float64)
        cov_te = Covariances("oas").fit_transform(x_te)
        return lr.predict_log_proba(ts.transform(cov_te))

    return predict_logp


def fit_recenter_8_30(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    eval_person_train: np.ndarray,
    K_eval: int,
    n_classes: int,
    sfreq: float,
    unit_train: np.ndarray | None = None,
) -> dict:
    """Fit per-subject reference covariances and ONE pooled LR on re-centered tangent vectors.

    subj_train identifies every subject in X_train (including training-pool subjects on dreyer_sim2).
    eval_person_train in {0..K_eval-1} (or -1 for pool subjects) identifies the K_eval evaluation
    participants whose reference matrices are exported for test-time routing."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.geometry.base import invsqrtm

    n_times = X_train.shape[-1]
    sos, a, b, filt = _crop_filter(RIEMANN_BAND, sfreq, n_times)
    x_tr = sosfiltfilt(sos, X_train, axis=-1)[:, :, a:b].astype(np.float64)
    cov_tr = Covariances("oas").fit_transform(x_tr)

    n_feat = X_train.shape[1] * (X_train.shape[1] + 1) // 2
    v_tr = np.zeros((len(X_train), n_feat), dtype=np.float64)
    eval_crefs = [None] * K_eval

    for s in np.unique(subj_train):
        m = subj_train == s
        ts_s = TangentSpace(metric="riemann").fit(cov_tr[m])
        v_tr[m] = ts_s.transform(cov_tr[m])
        k_vals = np.unique(eval_person_train[m])
        if len(k_vals) == 1 and k_vals[0] >= 0:
            eval_crefs[int(k_vals[0])] = invsqrtm(ts_s.reference_)

    lr = LogisticRegression(C=RIEMANN_C, max_iter=5000).fit(v_tr, y_train)
    coef, icpt = _lr_weights(lr, n_classes)

    f64 = torch.float64
    state_dict = {
        "filt": torch.as_tensor(filt, dtype=f64),
        "cref_isqrt": torch.as_tensor(np.stack(eval_crefs), dtype=f64),
        "weight": torch.as_tensor(coef, dtype=f64),
        "bias": torch.as_tensor(icpt, dtype=f64),
    }

    mod = RecenterRiemann(K_eval, X_train.shape[1], n_times, b - a, n_classes)
    mod.load_state_dict(state_dict)

    @torch.inference_mode()
    def per_person_logp(X_eval: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(X_eval)
        return torch.cat([mod(Xt[i : i + 256]) for i in range(0, len(Xt), 256)], dim=0).numpy()

    @torch.inference_mode()
    def whitener_interp_logp(X_eval: np.ndarray, fp_logp: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(X_eval)
        fpt = torch.as_tensor(fp_logp)
        return torch.cat(
            [mod.forward_whitener(Xt[i : i + 256], fpt[i : i + 256]) for i in range(0, len(Xt), 256)],
            dim=0,
        ).numpy()

    # Optional leave-one-unit-out calibration reliability of the pooled re-centered model
    rel_norm = None
    if unit_train is not None:
        cal_mask = eval_person_train >= 0
        units = np.unique(unit_train[cal_mask])
        if len(units) >= 2:
            oof_pred = np.zeros(len(X_train), dtype=int)
            for u in units:
                tr_u = ~cal_mask | (unit_train != u)
                ho_u = cal_mask & (unit_train == u)
                # Fit per-subject references on tr_u only (no leakage from held-out unit)
                v_u = np.zeros((int(tr_u.sum()), n_feat), dtype=np.float64)
                ts_by_k = {}
                cov_tr_u = cov_tr[tr_u]
                subj_tr_u = subj_train[tr_u]
                ep_tr_u = eval_person_train[tr_u]
                for s in np.unique(subj_tr_u):
                    ms = subj_tr_u == s
                    ts_s = TangentSpace(metric="riemann").fit(cov_tr_u[ms])
                    v_u[ms] = ts_s.transform(cov_tr_u[ms])
                    kv = np.unique(ep_tr_u[ms])
                    if len(kv) == 1 and kv[0] >= 0:
                        ts_by_k[int(kv[0])] = ts_s
                lr_u = LogisticRegression(C=RIEMANN_C, max_iter=5000).fit(v_u, y_train[tr_u])
                for k in range(K_eval):
                    mk = ho_u & (eval_person_train == k)
                    if mk.any():
                        oof_pred[mk] = lr_u.predict(ts_by_k[k].transform(cov_tr[mk]))
            rels = []
            for k in range(K_eval):
                mk = cal_mask & (eval_person_train == k)
                rels.append(balanced_accuracy_score(y_train[mk], oof_pred[mk]))
            rels = np.asarray(rels)
            rel_norm = 0.5 + 0.5 * (rels - 1.0 / n_classes) / (1.0 - 1.0 / n_classes)

    return {
        "state_dict": state_dict,
        "n_times_out": b - a,
        "cal_tangent": v_tr,
        "per_person_logp": per_person_logp,
        "whitener_interp_logp": whitener_interp_logp,
        "rel_norm": rel_norm,
    }


def fit_recenter_fb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    eval_person_train: np.ndarray,
    K_eval: int,
    n_classes: int,
    sfreq: float,
    bands=BANDS_5,
) -> dict:
    """Multi-band (4–45 Hz, 0.5–4 s) per-person reference re-centering + StandardScaler + 1 pooled LR."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    n_times = X_train.shape[-1]
    n_feat_band = X_train.shape[1] * (X_train.shape[1] + 1) // 2
    band_ts_eval = []
    band_soses = []
    v_bands_tr = []

    for band in bands:
        sos, a, b, _ = _crop_filter(band, sfreq, n_times)
        band_soses.append((sos, a, b))
        x_b = sosfiltfilt(sos, X_train, axis=-1)[:, :, a:b].astype(np.float64)
        cov_b = Covariances("oas").fit_transform(x_b)
        v_b = np.zeros((len(X_train), n_feat_band), dtype=np.float64)
        ts_eval_k = [None] * K_eval
        for s in np.unique(subj_train):
            m = subj_train == s
            ts_s = TangentSpace(metric="riemann").fit(cov_b[m])
            v_b[m] = ts_s.transform(cov_b[m])
            kv = np.unique(eval_person_train[m])
            if len(kv) == 1 and kv[0] >= 0:
                ts_eval_k[int(kv[0])] = ts_s
        v_bands_tr.append(v_b)
        band_ts_eval.append(ts_eval_k)

    F_tr = np.hstack(v_bands_tr)
    sc = StandardScaler().fit(F_tr)
    lr = LogisticRegression(C=RIEMANN_C, max_iter=5000).fit(sc.transform(F_tr), y_train)

    def per_person_logp(X_eval: np.ndarray) -> np.ndarray:
        covs_te = []
        for sos, a, b in band_soses:
            x_te = sosfiltfilt(sos, X_eval, axis=-1)[:, :, a:b].astype(np.float64)
            covs_te.append(Covariances("oas").fit_transform(x_te))
        out = np.zeros((len(X_eval), K_eval, n_classes), dtype=np.float64)
        for k in range(K_eval):
            f_k = np.hstack([ band_ts_eval[b_idx][k].transform(covs_te[b_idx]) for b_idx in range(len(bands)) ])
            out[:, k] = lr.predict_log_proba(sc.transform(f_k))
        return out

    return {"per_person_logp": per_person_logp}


def _c3_combine(eeg_logp: np.ndarray, riemann_logp: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Combine (n, K, C) eeg_logp and riemann_logp with C3 as shipped."""
    K, C = eeg_logp.shape[1:]
    comb = LogLinearCombiner(K, C)
    comb.coef.copy_(torch.tensor([riemann_parts.C3[k] for k in ("a", "c0", "c1")], dtype=torch.float64))
    if C == 2:
        comb.class_bias[1] = riemann_parts.C3["b"]
    comb.rel.copy_(torch.as_tensor(rel, dtype=torch.float64))
    with torch.inference_mode():
        return comb(torch.as_tensor(eeg_logp), torch.as_tensor(riemann_logp)).numpy()


def level2_arms(bank: dict, recenter_logp: np.ndarray, recenter_rel: np.ndarray | None = None) -> dict:
    """Compute Level 2 integration arms (n, C) log-probabilities from a harness bank and
    `recenter_logp` (n, K, C)."""
    n, K, C = bank["eeg_logp"].shape
    idx = np.arange(n)
    person = bank["person"]
    fp_logp = bank["fp_logp"]
    rel_shipped = bank["rel"]
    rel_pool = np.full(K, float(np.mean(rel_shipped))) if recenter_rel is None else recenter_rel

    # E1: 1 pooled EEGNet + 1 pooled re-centered Riemannian (no per-person classifiers)
    pooled_eeg_k = np.broadcast_to(bank["pooled_logp"][:, None, :], (n, K, C)).copy()
    e1_k = _c3_combine(pooled_eeg_k, recenter_logp, rel_pool)
    e1_routed = metrics.route(fp_logp, e1_k)
    e1_oracle = e1_k[idx, person]

    # E2: K per-person EEGNet experts + 1 pooled re-centered Riemannian
    e2_k = _c3_combine(bank["eeg_logp"], recenter_logp, rel_pool)
    e2_routed = metrics.route(fp_logp, e2_k)
    e2_oracle = e2_k[idx, person]

    # E3: K per-person EEGNet experts + equal log-pool of per-person Riemann and pooled re-centered Riemann
    shrunk_riemann = 0.5 * bank["riemann_logp"] + 0.5 * recenter_logp
    shrunk_riemann = shrunk_riemann - np.log(np.exp(shrunk_riemann).sum(-1, keepdims=True))
    e3_k = _c3_combine(bank["eeg_logp"], shrunk_riemann, rel_shipped)
    e3_routed = metrics.route(fp_logp, e3_k)
    e3_oracle = e3_k[idx, person]

    return {
        "E1_fully_pooled_recenter": e1_routed,
        "E1_oracle": e1_oracle,
        "E2_eeg_experts_plus_recenter": e2_routed,
        "E2_oracle": e2_oracle,
        "E3_eeg_experts_plus_shrunk_riemann": e3_routed,
        "E3_oracle": e3_oracle,
    }
