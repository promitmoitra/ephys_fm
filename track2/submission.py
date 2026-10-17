"""Track 2 submission: fingerprint-routed mixture of per-participant experts.

Design (track2/README.md, "Current best model"): a pooled EEGNet is fine-tuned
once per evaluation participant on that participant's labeled calibration
sessions. A per-participant Riemannian expert (tangent-space logistic
regression) is combined with it in log space, weighted by how reliable the
Riemannian expert was on the participant's calibration data. ``predict(X)``
receives no subject IDs, so a fingerprint recognises the participant from the
window itself and mixes the per-participant experts by its posterior:

    p(class | x) = sum_s p(subject = s | x) * p_s(class | x)
    log p_s(c | x) ∝ a·log p_eegnet_s(c | x)
                     + (c0 + c1·(rel_s − 0.5))·log p_riemann_s(c | x) + b_c

Configurable in config.json (older packages, which lack the keys, load as
the original design):
    "fingerprint": "fb_riemann" (filter-bank covariance model) | "eegnet"
    "riemann_experts": true | false (EEGNet experts only)

Shipped files (read from ``meta["submission_dir"]``):
    mixture.pt   {"fingerprint": sd, "experts": [sd, ...],
                  "riemann": sd, "combiner": sd}   (last two if enabled)
    config.json  training channel names, n_times, n_classes, expert labels,
                 component flags and shapes

Self-contained per the contract: all inference code lives here and imports
only the worker image's stack (torch, braindecode). The covariance models
are exact torch re-implementations of the scipy / pyriemann / sklearn
pipelines they were fitted with (track2/riemann_parts.py).
"""

import json
import os
from pathlib import Path

import torch
from torch import nn
from braindecode.models import EEGNet

from benchmark_utils.base_solver import CompetSolver


# --------------------------------------------------------------------------
# Covariance / tangent-space building blocks
# --------------------------------------------------------------------------

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


def tangent(C, cref_isqrt):
    """pyriemann TangentSpace (metric riemann): upper triangle of
    logm(Cref^-1/2 C Cref^-1/2), off-diagonal entries weighted by sqrt(2)."""
    w, V = torch.linalg.eigh(cref_isqrt @ C @ cref_isqrt)
    L = (V * torch.log(w)[..., None, :]) @ V.transpose(-1, -2)
    p = L.shape[-1]
    i, j = torch.triu_indices(p, p, device=L.device)
    coef = torch.where(i == j, 1.0, 2 ** 0.5).to(L.dtype)
    return coef * L[..., i, j]


class FBFingerprint(nn.Module):
    """p(subject | window) from per-band covariances (loop A).

    Each band's zero-phase band-pass on the fixed-length window is one (T, T)
    matrix; per band OAS covariance → tangent space at the calibration mean;
    bands concatenated → standardised multinomial logistic regression, folded
    into one linear layer."""

    def __init__(self, n_bands, n_chans, n_times, n_subjects):
        super().__init__()
        n_feat = n_bands * n_chans * (n_chans + 1) // 2
        self.register_buffer("filters", torch.zeros(n_bands, n_times, n_times))
        self.register_buffer("cref_isqrt", torch.zeros(n_bands, n_chans, n_chans))
        self.linear = nn.Linear(n_feat, n_subjects)

    def features(self, X):
        """Concatenated per-band tangent features (B, F), before the linear layer."""
        X = X.to(self.filters.dtype)
        return torch.cat([tangent(oas(X @ M), W)
                          for M, W in zip(self.filters, self.cref_isqrt)], -1)

    def forward(self, X):
        """Logits (B, K)."""
        return self.linear(self.features(X).to(self.linear.weight.dtype))


def hermitian_cov(Z, rho=0.05):
    """Trace-shrunk Hermitian sample covariance of analytic signals Z (..., C, T)."""
    n_times, n_chans = Z.shape[-1], Z.shape[-2]
    gamma = (Z @ Z.mH) / n_times
    tr = torch.diagonal(gamma.real, dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(n_chans, dtype=Z.dtype, device=Z.device)
    return (1.0 - rho) * gamma + rho * (tr / n_chans)[..., None, None] * eye


def complex_tangent(C_hpd, cref_isqrt):
    """Project C_hpd (..., C, C) to tangent space at cref_isqrt = Cref^-1/2 and vectorize
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


class ComplexFBFingerprint(nn.Module):
    """p(subject | window) from per-band complex Hermitian covariances (PR #15).

    Each band's zero-phase band-pass + Hilbert transform on the fixed-length window
    is one complex64 (T, T) matrix; per band trace-shrunk Hermitian covariance →
    complex tangent space at the complex Fréchet geometric mean; bands concatenated →
    standardised multinomial logistic regression, folded into one linear layer."""

    def __init__(self, n_bands, n_chans, n_times, n_subjects, rho=0.05):
        super().__init__()
        self.rho = float(rho)
        n_feat = n_bands * n_chans * n_chans
        self.register_buffer("filters", torch.zeros(n_bands, n_times, n_times, dtype=torch.complex64))
        self.register_buffer("cref_isqrt", torch.zeros(n_bands, n_chans, n_chans, dtype=torch.complex64))
        self.linear = nn.Linear(n_feat, n_subjects)

    def features(self, X):
        """Concatenated per-band complex tangent features (B, F)."""
        Xc = X.to(self.filters.dtype)
        feats = [
            complex_tangent(hermitian_cov(Xc @ F, rho=self.rho), W)
            for F, W in zip(self.filters, self.cref_isqrt)
        ]
        return torch.cat(feats, dim=-1)

    def forward(self, X):
        """Logits (B, K)."""
        return self.linear(self.features(X).to(self.linear.weight.dtype))


class RiemannExperts(nn.Module):
    """K per-participant tangent-space logistic regressions (loop B); float64.

    Supports:
    - 2-band individualized mu/beta decomposition ('mu_beta_2band', winning design):
      filters: (K, 2, n_times, n_times_out)
      cref_isqrt: (K, 2, n_chans, n_chans)
      weight: (K, n_classes, 2 * n_feat)
    - legacy 1-band ('fixed_8_30'):
      filt: (n_times, n_times_out)
      cref_isqrt: (K, n_chans, n_chans)
      weight: (K, n_classes, n_feat)

    Online test-time adaptation:
    - Maintains a causal running tangent barycenter state `running_mu: (K, F)`
      updated by: mu_k <- (1 - alpha * gamma_tk) * mu_k + alpha * gamma_tk * f_tk,
      subtracting mu_k from the test window's tangent vector before classification."""

    def __init__(self, n_subjects, n_chans, n_times, n_times_out, n_classes,
                 n_bands=2, online_adaptation=True, online_alpha=0.05):
        super().__init__()
        self.n_subjects = n_subjects
        self.n_chans = n_chans
        self.n_bands = n_bands
        self.n_times = n_times
        self.n_times_out = n_times_out
        self.n_classes = n_classes
        self.n_single_feat = n_chans * (n_chans + 1) // 2
        self.n_feat = self.n_bands * self.n_single_feat
        self.online_adaptation = online_adaptation
        self.alpha = float(online_alpha)
        d = torch.float64

        if n_bands == 1:
            self.register_buffer("filt", torch.zeros(n_times, n_times_out, dtype=d))
            self.register_buffer("cref_isqrt", torch.zeros(n_subjects, n_chans, n_chans, dtype=d))
        else:
            self.register_buffer("filters", torch.zeros(n_subjects, n_bands, n_times, n_times_out, dtype=d))
            self.register_buffer("cref_isqrt", torch.zeros(n_subjects, n_bands, n_chans, n_chans, dtype=d))

        self.register_buffer("weight", torch.zeros(n_subjects, n_classes, self.n_feat, dtype=d))
        self.register_buffer("bias", torch.zeros(n_subjects, n_classes, dtype=d))
        self.register_buffer("running_mu", torch.zeros(n_subjects, self.n_feat, dtype=d), persistent=False)

    def reset_adaptation(self):
        """Reset online running tangent barycenter to zero."""
        self.running_mu = torch.zeros(self.n_subjects, self.n_feat,
                                      dtype=self.weight.dtype,
                                      device=self.weight.device)

    def forward(self, X, p_subject=None):
        """X: (B, C, T) float32/float64. p_subject: optional (B, K) posterior for online adaptation."""
        if hasattr(self, "filt"):
            C = oas(X.to(self.filt.dtype) @ self.filt)                      # (B, C, C)
            f = tangent(C[:, None], self.cref_isqrt[None])                  # (B, K, F)
        else:
            Xd = X.to(self.filters.dtype)
            f_all = []
            for k in range(self.n_subjects):
                f_k = []
                for b in range(self.n_bands):
                    X_kb = Xd @ self.filters[k, b]                         # (B, C, T_out)
                    C_kb = oas(X_kb)                                       # (B, C, C)
                    f_kb = tangent(C_kb, self.cref_isqrt[k, b])            # (B, F_single)
                    f_k.append(f_kb)
                f_all.append(torch.cat(f_k, dim=-1))                       # (B, n_bands * F_single)
            f = torch.stack(f_all, dim=1)                                  # (B, K, F)

        if self.online_adaptation and p_subject is not None:
            gamma = p_subject.to(f.dtype)                                   # (B, K)
            f_adapted = torch.empty_like(f)
            for t in range(f.shape[0]):
                eta = (self.alpha * gamma[t]).unsqueeze(-1)                 # (K, 1)
                self.running_mu = (1.0 - eta) * self.running_mu + eta * f[t]
                f_adapted[t] = f[t] - self.running_mu
            f = f_adapted

        logits = torch.einsum("bkf,kcf->bkc", f, self.weight) + self.bias
        return torch.log_softmax(logits, -1)


class LogLinearCombiner(nn.Module):
    """Reliability-weighted log-linear pooling of the two experts (loop B, C3)."""

    def __init__(self, n_subjects, n_classes):
        super().__init__()
        d = torch.float64
        self.register_buffer("coef", torch.zeros(3, dtype=d))          # a, c0, c1
        self.register_buffer("class_bias", torch.zeros(n_classes, dtype=d))
        self.register_buffer("rel", torch.zeros(n_subjects, dtype=d))  # chance-normalised

    def forward(self, logp_eeg, logp_riemann):
        a, c0, c1 = self.coef
        w = (c0 + c1 * (self.rel - 0.5))[None, :, None]
        z = a * logp_eeg.to(w.dtype) + w * logp_riemann + self.class_bias
        return torch.log_softmax(z, -1)


class PortfolioCombiner(nn.Module):
    """N-stream reliability-weighted log-linear pooling (loop C):
    z = Σ_s w_s·log p_s + (c0 + c1·(rel − 0.5))·log p_riemann + class_bias."""

    def __init__(self, n_streams, n_people, n_classes):
        super().__init__()
        d = torch.float64
        self.register_buffer("w", torch.zeros(n_streams, dtype=d))
        self.register_buffer("c", torch.zeros(2, dtype=d))
        self.register_buffer("class_bias", torch.zeros(n_classes, dtype=d))
        self.register_buffer("rel", torch.zeros(n_people, dtype=d))

    def forward(self, neural, logp_riemann):
        z = sum(w * lp.to(self.w.dtype) for w, lp in zip(self.w, neural))
        wc = (self.c[0] + self.c[1] * (self.rel - 0.5))[None, :, None]
        return torch.log_softmax(z + wc * logp_riemann + self.class_bias, -1)


# ShallowFBCSPNet time constants (braindecode defaults assume 250 Hz), rescaled to sfreq;
# the same numbers as track2/models.py, which this self-contained file cannot import.
_SHALLOW_REF = dict(filter_time_length=25, pool_time_length=75, pool_time_stride=15)


def _shallow(n_chans, n_classes, n_times, sfreq):
    from braindecode.models import ShallowFBCSPNet
    s = sfreq / 250.0
    kw = {k: max(1, int(round(v * s))) for k, v in _SHALLOW_REF.items()}
    return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times,
                           final_conv_length="auto", **kw)


def standardize_clip(X, clip=15.0):
    """Per-window, per-channel z-score clipped at ±clip SD (REVE's pretraining input)."""
    mu = X.mean(-1, keepdim=True)
    sd = X.std(-1, keepdim=True).clamp_min(1e-6)
    return ((X - mu) / sd).clamp(-clip, clip)


def load_reve_encoder(positions_dir, _constructor=None):
    """Frozen REVE from the pre-staged Hugging Face cache, fully offline.

    braindecode's REVE reads its position bank from $REVE_POSITIONS_PATH/reve_positions.json
    when it is constructed; the submission ships that file, plus reve_kwargs.json (the
    constructor arguments probe P0 found necessary, possibly {})."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["REVE_POSITIONS_PATH"] = str(positions_dir)
    if _constructor is None:
        from braindecode.models import REVE
        kw_file = Path(positions_dir) / "reve_kwargs.json"
        kwargs = json.loads(kw_file.read_text()) if kw_file.exists() else {}

        def _constructor():
            # local_files_only: huggingface_hub reads HF_HUB_OFFLINE once at import (before
            # this function runs), so the env var alone would not stop network attempts.
            return REVE.from_pretrained("brain-bzh/reve-base", local_files_only=True, **kwargs)
    enc = _constructor().eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


class ReveProbe(nn.Module):
    """Frozen REVE encoder + per-participant linear heads (loop C); forward -> (B, K, C)."""

    def __init__(self, encoder, n_people, n_classes, n_chans, n_times, n_out, emb_dim=512):
        super().__init__()
        self.encoder = encoder           # not in this module's state dict (frozen, cached)
        self.register_buffer("resample", torch.zeros(n_times, n_out))
        self.register_buffer("pos", torch.zeros(n_chans, 3))
        d = torch.float64
        self.register_buffer("mu", torch.zeros(emb_dim, dtype=d))
        self.register_buffer("sd", torch.ones(emb_dim, dtype=d))
        self.register_buffer("weight", torch.zeros(n_people, n_classes, emb_dim, dtype=d))
        self.register_buffer("bias", torch.zeros(n_people, n_classes, dtype=d))

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("encoder.")}

    def load_state_dict(self, state, strict=True):
        """The frozen encoder is not in the shipped state; every probe buffer must be."""
        result = super().load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("encoder.")]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(f"ReveProbe state: missing {missing}, "
                               f"unexpected {result.unexpected_keys}")
        return result

    def forward(self, X):
        x = standardize_clip(X.to(self.resample.dtype) @ self.resample)
        f = self.encoder(x, pos=self.pos.expand(len(x), -1, -1), return_features=True)
        z = (f["features"].mean(dim=(1, 2)).to(self.mu.dtype) - self.mu) / self.sd
        logits = torch.einsum("bd,kcd->bkc", z, self.weight) + self.bias
        return torch.log_softmax(logits, -1)


# --------------------------------------------------------------------------
# The mixture
# --------------------------------------------------------------------------

class FingerprintMixture(nn.Module):
    """Fingerprint-weighted mixture of per-participant experts."""

    def __init__(self, n_chans, n_times, n_classes, n_experts, channel_index=None,
                 fingerprint="complex_6b", n_bands=6, riemann_n_times_out=None,
                 riemann_n_bands=2, online_adaptation=True, online_alpha=0.05,
                 shallow=False, reve_encoder=None, n_reve_out=None, combiner="C3", sfreq=120.0):
        super().__init__()
        self.fingerprint_kind = fingerprint
        if fingerprint == "eegnet":
            self.fingerprint = EEGNet(n_chans=n_chans, n_outputs=n_experts, n_times=n_times)
        elif fingerprint == "fb_riemann":
            self.fingerprint = FBFingerprint(n_bands, n_chans, n_times, n_experts)
        elif fingerprint in ("complex_6b", "complex"):
            self.fingerprint = ComplexFBFingerprint(n_bands, n_chans, n_times, n_experts)
        else:
            raise ValueError(f"unknown fingerprint {fingerprint!r}")
        self.experts = nn.ModuleList(
            EEGNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
            for _ in range(n_experts))
        self.shallow_experts = nn.ModuleList(
            _shallow(n_chans, n_classes, n_times, sfreq) for _ in range(n_experts)
        ) if shallow else nn.ModuleList()
        self.reve = (ReveProbe(reve_encoder, n_experts, n_classes, n_chans, n_times, n_reve_out)
                     if reve_encoder is not None else None)
        self.combiner_kind = combiner
        if combiner == "C3" and (shallow or reve_encoder is not None):
            raise ValueError("extra expert streams are only combined by the 'portfolio' combiner; "
                             "C3 would compute and then ignore them")
        self.riemann = self.combiner = None
        if riemann_n_times_out is not None:
            self.riemann = RiemannExperts(n_experts, n_chans, n_times,
                                          riemann_n_times_out, n_classes,
                                          n_bands=riemann_n_bands,
                                          online_adaptation=online_adaptation,
                                          online_alpha=online_alpha)
            if combiner == "C3":
                self.combiner = LogLinearCombiner(n_experts, n_classes)
            elif combiner == "portfolio":
                n_streams = 1 + int(shallow) + int(reve_encoder is not None)
                self.combiner = PortfolioCombiner(n_streams, n_experts, n_classes)
            else:
                raise ValueError(f"unknown combiner {combiner!r}")
        elif combiner != "C3" or shallow or reve_encoder is not None:
            raise ValueError("extra expert streams need the Riemannian experts and a combiner")
        # Reorders the evaluation channels into the training order.
        self.register_buffer(
            "channel_index",
            torch.as_tensor(channel_index if channel_index is not None
                            else range(n_chans), dtype=torch.long),
            persistent=False)

    def reset_adaptation(self):
        """Reset running adaptation state."""
        if self.riemann is not None:
            self.riemann.reset_adaptation()

    def expert_logp(self, X, p_subject=None):
        """Per-participant class log-probabilities (B, K, C), before routing.

        Neural streams in the combiner's order: EEGNet, ShallowFBCSPNet, REVE."""
        neural = [torch.stack([torch.log_softmax(e(X), 1) for e in self.experts], 1)]
        if len(self.shallow_experts):
            neural.append(torch.stack([torch.log_softmax(e(X), 1)
                                       for e in self.shallow_experts], 1))
        if self.reve is not None:
            neural.append(self.reve(X))
        if self.riemann is None:
            return neural[0]
        lp_riemann = self.riemann(X, p_subject=p_subject)
        if self.combiner_kind == "C3":
            return self.combiner(neural[0], lp_riemann)
        return self.combiner(neural, lp_riemann)

    def forward(self, X):
        """Mixture class probabilities, (B, n_classes)."""
        X = X.index_select(1, self.channel_index)
        p_subject = torch.softmax(self.fingerprint(X), dim=1)            # (B, K)
        p_class = self.expert_logp(X, p_subject=p_subject).exp()          # (B, K, C)
        return (p_subject.unsqueeze(-1).to(p_class.dtype) * p_class).sum(1)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32,
                            device=self.channel_index.device)
        return self(X).argmax(dim=1)                                     # (B,)


def build_model(meta, config, state=None):
    """Mixture for the evaluation ``meta``; maps channels by name."""
    train_chs = config["ch_names"]
    eval_chs = list(meta["ch_names"])
    missing = [c for c in train_chs if c not in eval_chs]
    if missing:
        raise ValueError(f"evaluation data lacks training channels {missing}")
    if meta["n_times"] != config["n_times"]:
        raise ValueError(f"n_times {meta['n_times']} != trained {config['n_times']}")
    if meta["n_classes"] != config["n_classes"]:
        raise ValueError(f"n_classes {meta['n_classes']} != trained {config['n_classes']}")
    if "sfreq" in config and "sfreq" in meta and abs(meta["sfreq"] - config["sfreq"]) > 1e-6:
        raise ValueError(f"sfreq {meta['sfreq']} != trained {config['sfreq']}")
    use_riemann = config.get("riemann_experts", False)
    encoder = load_reve_encoder(meta["submission_dir"]) if config.get("reve_probe") else None

    # Resolve fingerprint kind with backward compatibility
    fp_kind = config.get("fingerprint")
    if fp_kind is None:
        if state is not None and "fingerprint" in state:
            fp_sd = state["fingerprint"]
            if "filters" in fp_sd and fp_sd["filters"].dtype in (torch.complex64, torch.complex128):
                fp_kind = "complex_6b"
            elif "filters" in fp_sd:
                fp_kind = "fb_riemann"
            else:
                fp_kind = "eegnet"
        else:
            fp_kind = "complex_6b"

    # Resolve riemann bands mode
    riemann_n_bands = config.get("riemann_n_bands")
    if riemann_n_bands is None:
        if config.get("riemann_bands") == "mu_beta_2band":
            riemann_n_bands = 2
        elif state is not None and "riemann" in state and "filters" in state["riemann"]:
            riemann_n_bands = state["riemann"]["filters"].shape[1]
        else:
            riemann_n_bands = 1

    model = FingerprintMixture(
        n_chans=len(train_chs), n_times=config["n_times"],
        n_classes=config["n_classes"], n_experts=len(config["experts"]),
        channel_index=[eval_chs.index(c) for c in train_chs],
        fingerprint=fp_kind,
        n_bands=config.get("fingerprint_n_bands", 6),
        riemann_n_times_out=config["riemann_n_times_out"] if use_riemann else None,
        riemann_n_bands=riemann_n_bands,
        online_adaptation=config.get("online_adaptation", True),
        online_alpha=config.get("online_alpha", 0.05),
        shallow=config.get("shallow_experts", False), reve_encoder=encoder,
        n_reve_out=config.get("reve_n_out"), combiner=config.get("combiner", "C3"),
        sfreq=config.get("sfreq", 120.0))
    if state is not None:
        model.fingerprint.load_state_dict(state["fingerprint"])
        for expert, sd in zip(model.experts, state["experts"]):
            expert.load_state_dict(sd)
        shallow_sd = state.get("shallow_experts", [])
        if len(shallow_sd) != len(model.shallow_experts):
            raise ValueError(f"config expects {len(model.shallow_experts)} shallow experts, "
                             f"the state holds {len(shallow_sd)}")
        for expert, sd in zip(model.shallow_experts, shallow_sd):
            expert.load_state_dict(sd)
        if model.reve is not None:
            model.reve.load_state_dict(state["reve"])
        if use_riemann:
            model.riemann.load_state_dict(state["riemann"])
            model.combiner.load_state_dict(state["combiner"])
    model.reset_adaptation()
    return model.to(meta["device"]).eval()


class Solver(CompetSolver):

    name = "FingerprintMixture"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        sub = meta["submission_dir"]
        config = json.loads((sub / "config.json").read_text())
        state = torch.load(sub / "mixture.pt", map_location=meta["device"],
                           weights_only=True)
        return build_model(meta, config, state)
