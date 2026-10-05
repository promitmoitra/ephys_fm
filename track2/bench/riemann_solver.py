"""Benchmark-page solver: pooled filter-bank tangent-space classifier (train split only)."""
import json
import torch
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "Riemann-FB-pooled"
    requirements = []

    def load_model(self, meta):
        import importlib.util
        sub = meta["submission_dir"]
        spec = importlib.util.spec_from_file_location("track2_submission", sub / "submission_lib.py")
        lib = importlib.util.module_from_spec(spec); spec.loader.exec_module(lib)
        cfg = json.loads((sub / "config.json").read_text())
        m = lib.FBFingerprint(cfg["n_bands"], meta["n_chans"], meta["n_times"], meta["n_classes"])
        m.load_state_dict(torch.load(sub / "model.pt", map_location=meta["device"], weights_only=True))

        class _M(torch.nn.Module):
            def __init__(self, f):
                super().__init__(); self.f = f

            @torch.inference_mode()
            def predict(self, X):
                return self.f(X).argmax(1)
        return _M(m).to(meta["device"]).eval()
