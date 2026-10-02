import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(HERE)]
import reve_parts  # noqa: E402
from models import make_model  # noqa: E402
from submission import FBFingerprint  # noqa: E402

CHS = [f"C{i}" for i in range(27)]
POS_DIR = REPO / "outputs" / "t2-portfolio" / "reve_positions"


def solver_cls(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "bench" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Solver


def meta(folder):
    return {"n_chans": 27, "n_times": 480, "n_classes": 2, "sfreq": 120.0, "ch_names": CHS,
            "device": "cpu", "submission_dir": Path(folder)}


class TestBenchSolvers(unittest.TestCase):
    def test_shallow_solver_predicts_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(make_model("shallow", 27, 2, 480, 120.0).state_dict(), Path(tmp) / "model.pt")
            (Path(tmp) / "config.json").write_text(json.dumps({"ch_names": CHS}))
            m = solver_cls("shallow_solver")().load_model(meta(tmp))
            p = m.predict(torch.randn(3, 27, 480))
            self.assertEqual(tuple(p.shape), (3,))
            self.assertTrue(set(p.tolist()) <= {0, 1})

    def test_riemann_solver_predicts_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copyfile(HERE / "submission.py", Path(tmp) / "submission_lib.py")
            torch.save(FBFingerprint(6, 27, 480, 2).state_dict(), Path(tmp) / "model.pt")
            (Path(tmp) / "config.json").write_text(json.dumps({"ch_names": CHS, "n_bands": 6}))
            m = solver_cls("riemann_solver")().load_model(meta(tmp))
            self.assertEqual(tuple(m.predict(torch.randn(3, 27, 480)).shape), (3,))

    def test_reve_solver_predicts_labels(self):
        if not (POS_DIR / "reve_kwargs.json").exists():
            self.skipTest("REVE positions not cached")
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copyfile(HERE / "submission.py", Path(tmp) / "submission_lib.py")
            for f in ("reve_positions.json", "reve_kwargs.json"):
                shutil.copyfile(POS_DIR / f, Path(tmp) / f)
            bank = json.loads((POS_DIR / "reve_positions.json").read_text())
            chs = list(bank)[:27]
            rng = np.random.default_rng(0)
            state = reve_parts.export(reve_parts.resample_matrix(480, 120.0),
                                      reve_parts.positions(chs, POS_DIR / "reve_positions.json"),
                                      np.zeros(512), np.ones(512),
                                      rng.standard_normal((1, 2, 512)), np.zeros((1, 2)))
            torch.save(state, Path(tmp) / "model.pt")
            (Path(tmp) / "config.json").write_text(json.dumps({"ch_names": chs, "n_out": 800}))
            mt = meta(tmp) | {"ch_names": chs}
            m = solver_cls("reve_solver")().load_model(mt)
            self.assertEqual(tuple(m.predict(torch.randn(2, 27, 480)).shape), (2,))


if __name__ == "__main__":
    unittest.main()
