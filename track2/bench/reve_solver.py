"""Benchmark-page solver: frozen REVE + one pooled linear head (warm-up train split only)."""
import importlib.util
import json

import torch
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "REVE-probe-pooled"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        sub = meta["submission_dir"]
        spec = importlib.util.spec_from_file_location("track2_submission", sub / "submission_lib.py")
        lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib)
        cfg = json.loads((sub / "config.json").read_text())
        assert cfg["ch_names"] == list(meta["ch_names"]), "channel order differs"
        probe = lib.ReveProbe(lib.load_reve_encoder(sub), n_people=1,
                              n_classes=meta["n_classes"], n_chans=meta["n_chans"],
                              n_times=meta["n_times"], n_out=cfg["n_out"])
        probe.load_state_dict(torch.load(sub / "model.pt", map_location=meta["device"],
                                         weights_only=True))

        class _M(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            @torch.inference_mode()
            def predict(self, X):
                self.eval()
                return self.p(X)[:, 0].argmax(-1)

        return _M(probe).to(meta["device"]).eval()
