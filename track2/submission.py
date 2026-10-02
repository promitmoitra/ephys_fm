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

    def forward(self, X):
        """Logits (B, K)."""
        X = X.to(self.filters.dtype)
        f = torch.cat([tangent(oas(X @ M), W)
                       for M, W in zip(self.filters, self.cref_isqrt)], -1)
        return self.linear(f.to(self.linear.weight.dtype))


class RiemannExperts(nn.Module):
    """K per-participant tangent-space logistic regressions (loop B); float64.

    Shared band-pass-and-crop matrix (T, T'), per-participant reference
    Cref^-1/2 and class weights. forward -> log-probabilities (B, K, C)."""

    def __init__(self, n_subjects, n_chans, n_times, n_times_out, n_classes):
        super().__init__()
        n_feat = n_chans * (n_chans + 1) // 2
        d = torch.float64
        self.register_buffer("filt", torch.zeros(n_times, n_times_out, dtype=d))
        self.register_buffer("cref_isqrt", torch.zeros(n_subjects, n_chans, n_chans, dtype=d))
        self.register_buffer("weight", torch.zeros(n_subjects, n_classes, n_feat, dtype=d))
        self.register_buffer("bias", torch.zeros(n_subjects, n_classes, dtype=d))

    def forward(self, X):
        C = oas(X.to(self.filt.dtype) @ self.filt)                      # (B, c, c)
        f = tangent(C[:, None], self.cref_isqrt[None])                    # (B, K, F)
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
            return REVE.from_pretrained("brain-bzh/reve-base", **kwargs)
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
                 fingerprint="eegnet", n_bands=6, riemann_n_times_out=None, shallow=False,
                 reve_encoder=None, n_reve_out=None, combiner="C3", sfreq=120.0):
        super().__init__()
        self.fingerprint_kind = fingerprint
        if fingerprint == "eegnet":
            self.fingerprint = EEGNet(n_chans=n_chans, n_outputs=n_experts, n_times=n_times)
        elif fingerprint == "fb_riemann":
            self.fingerprint = FBFingerprint(n_bands, n_chans, n_times, n_experts)
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
        self.riemann = self.combiner = None
        if riemann_n_times_out is not None:
            self.riemann = RiemannExperts(n_experts, n_chans, n_times,
                                          riemann_n_times_out, n_classes)
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

    def expert_logp(self, X):
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
        if self.combiner_kind == "C3":
            return self.combiner(neural[0], self.riemann(X))
        return self.combiner(neural, self.riemann(X))

    def forward(self, X):
        """Mixture class probabilities, (B, n_classes)."""
        X = X.index_select(1, self.channel_index)
        p_subject = torch.softmax(self.fingerprint(X), dim=1)            # (B, K)
        p_class = self.expert_logp(X).exp()                              # (B, K, C)
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
    model = FingerprintMixture(
        n_chans=len(train_chs), n_times=config["n_times"],
        n_classes=config["n_classes"], n_experts=len(config["experts"]),
        channel_index=[eval_chs.index(c) for c in train_chs],
        fingerprint=config.get("fingerprint", "eegnet"),
        n_bands=config.get("fingerprint_n_bands", 6),
        riemann_n_times_out=config["riemann_n_times_out"] if use_riemann else None,
        shallow=config.get("shallow_experts", False), reve_encoder=encoder,
        n_reve_out=config.get("reve_n_out"), combiner=config.get("combiner", "C3"),
        sfreq=config.get("sfreq", 120.0))
    if state is not None:
        model.fingerprint.load_state_dict(state["fingerprint"])
        for expert, sd in zip(model.experts, state["experts"]):
            expert.load_state_dict(sd)
        for expert, sd in zip(model.shallow_experts, state.get("shallow_experts", [])):
            expert.load_state_dict(sd)
        if model.reve is not None:
            model.reve.load_state_dict(state["reve"])
        if use_riemann:
            model.riemann.load_state_dict(state["riemann"])
            model.combiner.load_state_dict(state["combiner"])
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
