"""Test-time reference covariance adaptation for per-person RiemannExperts (Protocol 01).

Keeps each person k's calibrated task classifier (w_k, b_k) from `RiemannExperts` (`row 2`),
and adapts person k's reference covariance Cref_k during the test session using fingerprint
posteriors p(k | x) to counter cross-day electrode/impedance drift.

Implements all 10 pre-registered arms from `experiments/01-protocol/protocol.md`, including
mandatory FM-10 random-permutation controls for all sequential/batch arms.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from scipy.signal import sosfiltfilt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [
    str(REPO / "external" / "2026-competition"),
    str(REPO / "track2"),
    str(REPO / "research" / "integration-harness" / "src"),
]

import riemann_parts  # noqa: E402
from submission import oas, tangent  # noqa: E402

PROTOCOL01_ARMS = (
    "static_cal_ref",
    "ema_tangent_ordered",
    "ema_tangent_shuffled",
    "ema_spd_ordered",
    "ema_spd_shuffled",
    "batch64_ordered",
    "batch64_shuffled",
    "session_fp_tangent",
    "session_fp_spd",
    "oracle_session_spd",
)


def fit_per_person_experts(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    K: int,
    n_classes: int,
    sfreq: float,
) -> dict:
    """Fit shipped per-person `RiemannExperts` on calibration windows."""
    T = X_cal.shape[-1]
    ref = riemann_parts._Riemann(sfreq)
    crefs_isqrt, crefs_cal, log_crefs_cal, weights, biases = [], [], [], [], []
    for k in range(K):
        m = p_cal == k
        model = riemann_parts._Riemann(sfreq).fit(X_cal[m], y_cal[m])
        c_ref = model.ts.reference_
        crefs_cal.append(c_ref)
        crefs_isqrt.append(invsqrtm(c_ref))
        vals, vecs = np.linalg.eigh(c_ref)
        log_crefs_cal.append((vecs * np.log(np.clip(vals, 1e-12, None))) @ vecs.T)
        coef, icpt = model.lr.coef_.copy(), model.lr.intercept_.copy()
        if n_classes == 2:
            coef = np.vstack([np.zeros_like(coef), coef])
            icpt = np.array([0.0, icpt[0]])
        weights.append(coef)
        biases.append(icpt)
    filt = np.ascontiguousarray(sosfiltfilt(ref.sos, np.eye(T), axis=-1)[:, ref.a : ref.b])
    f64 = torch.float64
    sd = {
        "filt": torch.as_tensor(filt, dtype=f64),
        "cref_isqrt": torch.as_tensor(np.stack(crefs_isqrt), dtype=f64),
        "weight": torch.as_tensor(np.stack(weights), dtype=f64),
        "bias": torch.as_tensor(np.stack(biases), dtype=f64),
    }
    return {
        "state_dict": sd,
        "cref_cal": np.stack(crefs_cal),
        "log_cref_cal": np.stack(log_crefs_cal),
    }


def _score_tangent(f: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> np.ndarray:
    """Apply per-person linear readouts (weight, bias) to tangent features f (N, K, F) -> (N, K, C)."""
    logits = torch.einsum("nkf,kcf->nkc", f, weight) + bias
    return torch.log_softmax(logits, dim=-1).numpy()


def _ema_tangent_seq(f_cal: np.ndarray, gamma: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Causal online tangent-mean EMA: mu_k^(t) = (1 - alpha*gamma_tk)*mu_k^(t-1) + alpha*gamma_tk*f_tk."""
    N, K, F = f_cal.shape
    out = np.empty_like(f_cal)
    mu = np.zeros((K, F), dtype=np.float64)
    for t in range(N):
        eta = (alpha * gamma[t])[:, None]  # (K, 1)
        mu = (1.0 - eta) * mu + eta * f_cal[t]
        out[t] = f_cal[t] - mu
    return out


def _ema_spd_seq(
    C_te: torch.Tensor,
    log_C_te: np.ndarray,
    log_cref_cal: np.ndarray,
    gamma: np.ndarray,
    alpha: float = 0.05,
) -> torch.Tensor:
    """Causal online Log-Euclidean EMA of reference covariance on SPD manifold."""
    N, c, _ = C_te.shape
    K = log_cref_cal.shape[0]
    F = c * (c + 1) // 2
    f_out = torch.empty((N, K, F), dtype=torch.float64)
    L = log_cref_cal.copy()  # (K, c, c)
    for t in range(N):
        eta = (alpha * gamma[t])[:, None, None]  # (K, 1, 1)
        L = (1.0 - eta) * L + eta * log_C_te[t][None]
        Lt = torch.from_numpy(L)
        vals, vecs = torch.linalg.eigh(Lt)
        W_t = (vecs * torch.exp(-0.5 * vals).unsqueeze(-2)) @ vecs.transpose(-1, -2)  # (K, c, c)
        f_out[t] = tangent(C_te[t : t + 1, None], W_t[None])[0]
    return f_out


def _batch_tangent(f_cal: np.ndarray, gamma: np.ndarray, batch_size: int = 64, n0: float = 10.0) -> np.ndarray:
    """Stateless within-batch tangent shift subtraction shrunk toward calibration with pseudo-count n0."""
    N, K, F = f_cal.shape
    out = np.empty_like(f_cal)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        g_b = gamma[start:end]  # (B, K)
        f_b = f_cal[start:end]  # (B, K, F)
        num = np.einsum("bk,bkf->kf", g_b, f_b)  # (K, F)
        den = (n0 + g_b.sum(axis=0))[:, None]    # (K, 1)
        mu_b = num / den
        out[start:end] = f_b - mu_b[None]
    return out


def _average_logp_over_permutations(fn_logp, N: int, perm_seeds=(0, 1, 2)) -> np.ndarray:
    """Run an order-dependent function fn_logp(perm) on 3 random window permutations and average probabilities."""
    probs = []
    for s in perm_seeds:
        perm = np.random.default_rng(s).permutation(N)
        inv_perm = np.argsort(perm)
        lp_perm = fn_logp(perm)
        probs.append(np.exp(lp_perm[inv_perm]))
    p_mean = np.clip(np.mean(probs, axis=0), 1e-12, 1.0)
    p_mean = p_mean / p_mean.sum(axis=-1, keepdims=True)
    return np.log(p_mean)


@torch.inference_mode()
def compute_all_arms(
    experts: dict,
    X_te: np.ndarray,
    fp_logp: np.ndarray,
    p_te: np.ndarray,
    alpha: float = 0.05,
    n0: float = 10.0,
    perm_seeds=(0, 1, 2),
) -> dict[str, np.ndarray]:
    """Compute per-person task log-probabilities (N, K, C) for all 10 Protocol 01 arms."""
    sd = experts["state_dict"]
    filt = sd["filt"]
    cref_isqrt = sd["cref_isqrt"]
    weight = sd["weight"]
    bias = sd["bias"]
    cref_cal = experts["cref_cal"]
    log_cref_cal = experts["log_cref_cal"]

    Xt = torch.as_tensor(X_te, dtype=filt.dtype)
    C_te = oas(Xt @ filt)                                    # (N, c, c)
    C_te_np = C_te.numpy()
    f_cal_t = tangent(C_te[:, None], cref_isqrt[None])       # (N, K, F)
    f_cal = f_cal_t.numpy()

    N, K, _ = f_cal.shape
    gamma = np.exp(fp_logp).astype(np.float64)
    gamma = gamma / gamma.sum(axis=-1, keepdims=True)

    # Precompute matrix logarithms log(C_t) once for SPD EMA arms
    vals_c, vecs_c = torch.linalg.eigh(C_te)
    log_C_te = ((vecs_c * torch.log(vals_c.clamp_min(1e-12)).unsqueeze(-2)) @ vecs_c.transpose(-1, -2)).numpy()

    # 1. static_cal_ref (shipped RiemannExperts)
    lp_static = _score_tangent(f_cal_t, weight, bias)

    # 2. ema_tangent_ordered
    f_ema_tan_ord = _ema_tangent_seq(f_cal, gamma, alpha=alpha)
    lp_ema_tan_ord = _score_tangent(torch.from_numpy(f_ema_tan_ord), weight, bias)

    # 3. ema_tangent_shuffled (FM-10 control)
    lp_ema_tan_shuf = _average_logp_over_permutations(
        lambda perm: _score_tangent(
            torch.from_numpy(_ema_tangent_seq(f_cal[perm], gamma[perm], alpha=alpha)), weight, bias
        ),
        N,
        perm_seeds=perm_seeds,
    )

    # 4. ema_spd_ordered
    f_ema_spd_ord = _ema_spd_seq(C_te, log_C_te, log_cref_cal, gamma, alpha=alpha)
    lp_ema_spd_ord = _score_tangent(f_ema_spd_ord, weight, bias)

    # 5. ema_spd_shuffled (FM-10 control)
    lp_ema_spd_shuf = _average_logp_over_permutations(
        lambda perm: _score_tangent(
            _ema_spd_seq(C_te[perm], log_C_te[perm], log_cref_cal, gamma[perm], alpha=alpha), weight, bias
        ),
        N,
        perm_seeds=perm_seeds,
    )

    # 6. batch64_ordered
    f_b64_ord = _batch_tangent(f_cal, gamma, batch_size=64, n0=n0)
    lp_b64_ord = _score_tangent(torch.from_numpy(f_b64_ord), weight, bias)

    # 7. batch64_shuffled (FM-10 control)
    lp_b64_shuf = _average_logp_over_permutations(
        lambda perm: _score_tangent(
            torch.from_numpy(_batch_tangent(f_cal[perm], gamma[perm], batch_size=64, n0=n0)), weight, bias
        ),
        N,
        perm_seeds=perm_seeds,
    )

    # 8. session_fp_tangent (order-invariant full-session tangent shift)
    num_s = np.einsum("nk,nkf->kf", gamma, f_cal)
    den_s = (n0 + gamma.sum(axis=0))[:, None]
    f_sess_tan = f_cal - (num_s / den_s)[None]
    lp_sess_tan = _score_tangent(torch.from_numpy(f_sess_tan), weight, bias)

    # 9. session_fp_spd (order-invariant full-session weighted Riemannian Fréchet mean + geodesic transport)
    W_sess_spd = np.zeros_like(cref_cal)
    for k in range(K):
        covs_k = np.concatenate([cref_cal[k : k + 1], C_te_np], axis=0)
        weights_k = np.concatenate([[n0], gamma[:, k]])
        c_mean_k = mean_riemann(covs_k, sample_weight=weights_k)
        W_sess_spd[k] = invsqrtm(c_mean_k)
    f_sess_spd = tangent(C_te[:, None], torch.from_numpy(W_sess_spd)[None])
    lp_sess_spd = _score_tangent(f_sess_spd, weight, bias)

    # 10. oracle_session_spd (true-ID test-session Riemannian Fréchet mean per person — exact ceiling)
    W_oracle_spd = np.zeros_like(cref_cal)
    for k in range(K):
        mk = p_te == k
        c_mean_oracle = mean_riemann(C_te_np[mk]) if mk.any() else cref_cal[k]
        W_oracle_spd[k] = invsqrtm(c_mean_oracle)
    f_oracle_spd = tangent(C_te[:, None], torch.from_numpy(W_oracle_spd)[None])
    lp_oracle_spd = _score_tangent(f_oracle_spd, weight, bias)

    return {
        "static_cal_ref": lp_static,
        "ema_tangent_ordered": lp_ema_tan_ord,
        "ema_tangent_shuffled": lp_ema_tan_shuf,
        "ema_spd_ordered": lp_ema_spd_ord,
        "ema_spd_shuffled": lp_ema_spd_shuf,
        "batch64_ordered": lp_b64_ord,
        "batch64_shuffled": lp_b64_shuf,
        "session_fp_tangent": lp_sess_tan,
        "session_fp_spd": lp_sess_spd,
        "oracle_session_spd": lp_oracle_spd,
    }
