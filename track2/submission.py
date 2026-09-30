"""Track 2 submission: fingerprint-routed mixture of per-participant EEGNets.

Design (validated on BNCI2014_001 in experiments/fingerprint_tangermann):
a pooled EEGNet is fine-tuned once per evaluation participant on that
participant's labeled calibration sessions; a fingerprint EEGNet recognises
the participant from the window itself, since ``predict(X)`` receives no
subject IDs. The prediction mixes the per-participant experts by the
fingerprint's posterior:

    p(class | x) = sum_s p(subject = s | x) * p_s(class | x)

Shipped files (read from ``meta["submission_dir"]``):
    mixture.pt   state dicts: {"fingerprint": sd, "experts": [sd, ...]}
    config.json  training channel names, n_times, n_classes, expert labels

Self-contained per the contract: all inference code lives here and imports
only the worker image's stack (torch, braindecode).
"""

import json

import torch
from torch import nn
from braindecode.models import EEGNet

from benchmark_utils.base_solver import CompetSolver


class FingerprintMixture(nn.Module):
    """Fingerprint-weighted mixture of per-participant EEGNet experts."""

    def __init__(self, n_chans, n_times, n_classes, n_experts, channel_index=None):
        super().__init__()
        self.fingerprint = EEGNet(n_chans=n_chans, n_outputs=n_experts,
                                  n_times=n_times)
        self.experts = nn.ModuleList(
            EEGNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
            for _ in range(n_experts))
        # Reorders the evaluation channels into the training order.
        self.register_buffer(
            "channel_index",
            torch.as_tensor(channel_index if channel_index is not None
                            else range(n_chans), dtype=torch.long),
            persistent=False)

    def forward(self, X):
        """Mixture class probabilities, (B, n_classes)."""
        X = X.index_select(1, self.channel_index)
        p_subject = torch.softmax(self.fingerprint(X), dim=1)            # (B, K)
        p_class = torch.stack([torch.softmax(e(X), dim=1)
                               for e in self.experts], dim=1)            # (B, K, C)
        return (p_subject.unsqueeze(-1) * p_class).sum(1)

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
    model = FingerprintMixture(
        n_chans=len(train_chs), n_times=config["n_times"],
        n_classes=config["n_classes"], n_experts=len(config["experts"]),
        channel_index=[eval_chs.index(c) for c in train_chs])
    if state is not None:
        model.fingerprint.load_state_dict(state["fingerprint"])
        for expert, sd in zip(model.experts, state["experts"]):
            expert.load_state_dict(sd)
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
