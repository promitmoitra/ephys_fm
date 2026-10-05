"""Benchmark-page solver: pooled ShallowFBCSPNet trained on the warm-up train split only."""
import json
import torch
from braindecode.models import ShallowFBCSPNet
from benchmark_utils.base_solver import CompetSolver

_REF = dict(filter_time_length=25, pool_time_length=75, pool_time_stride=15)


class _Model(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self.net(X.float()).argmax(1)


class Solver(CompetSolver):
    name = "ShallowFBCSPNet-pooled"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        cfg = json.loads((meta["submission_dir"] / "config.json").read_text())
        s = meta["sfreq"] / 250.0
        net = ShallowFBCSPNet(n_chans=meta["n_chans"], n_outputs=meta["n_classes"],
                              n_times=meta["n_times"], final_conv_length="auto",
                              **{k: max(1, int(round(v * s))) for k, v in _REF.items()})
        net.load_state_dict(torch.load(meta["submission_dir"] / "model.pt",
                                       map_location=meta["device"], weights_only=True))
        assert cfg["ch_names"] == list(meta["ch_names"]), "channel order differs"
        return _Model(net).to(meta["device"]).eval()
