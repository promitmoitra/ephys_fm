import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2"), str(HERE / "src")]
import pipeline  # noqa: E402
import riemann_parts  # noqa: E402
from submission import FBFingerprint  # noqa: E402


def toy(K=3, C=3, chans=6, per=40, seed=0):
    """Toy dataset: person-specific spatial scaling (identity) + class-specific band power."""
    rng = np.random.default_rng(seed)
    X, y, person, unit, calib, test = [], [], [], [], [], []
    for k in range(K):
        gain = 1 + 2 * rng.random(chans)
        for sess in (0, 1):
            for i in range(per):
                c = i % C
                t = np.arange(480) / 120.0
                sig = rng.standard_normal((chans, 480)) + np.sin(2 * np.pi * (8 + 6 * c) * t) * (c + 1)
                X.append(sig * gain[:, None]); y.append(c); person.append(k)
                unit.append((i * 2) // per if sess == 0 else -1)
                calib.append(sess == 0); test.append(sess == 1)
    calib = np.array(calib); unit = np.array(unit)
    val = calib & (unit == 1)
    return {"X": np.array(X, np.float32), "y": np.array(y), "person": np.array(person),
            "unit": unit, "calib": calib, "val": val, "test": np.array(test),
            "people": np.arange(K), "n_classes": C, "sfreq": 120.0, "name": "toy"}


class TestPipeline(unittest.TestCase):
    def test_features_then_linear_equals_forward(self):
        d = toy()
        st = riemann_parts.fit_fingerprint(d["X"][d["calib"]], d["person"][d["calib"]], 120.0)
        fp = FBFingerprint(6, 6, 480, 3); fp.load_state_dict(st)
        X = torch.from_numpy(d["X"][:5])
        with torch.inference_mode():
            self.assertTrue(torch.allclose(fp(X), fp.linear(fp.features(X)), atol=1e-5))

    def test_bank_keys_and_shapes(self):
        d = toy()
        path = pipeline.run(d, seed=0, threads=1, epochs=1, ft_epochs=1,
                            out_dir=Path(self._tmp()))
        b = np.load(path)
        n, K, C = int(d["test"].sum()), 3, 3
        self.assertEqual(b["fp_logp"].shape, (n, K))
        self.assertEqual(b["eeg_logp"].shape, (n, K, C))
        self.assertEqual(b["riemann_logp"].shape, (n, K, C))
        self.assertEqual(b["pooled_logp"].shape, (n, C))
        self.assertEqual(b["fp_dist"].shape, (n,))
        self.assertGreater(float(b["fp_dist_ref"]), 0)
        self.assertEqual(b["fp_cv_logp_calib"].shape, (int(d["calib"].sum()), K))
        self.assertEqual(b["rel"].shape, (K,))
        for k in ("fp_logp", "eeg_logp", "riemann_logp", "pooled_logp", "fp_cv_logp_calib"):
            np.testing.assert_allclose(np.exp(b[k]).sum(-1), 1.0, atol=1e-4, err_msg=k)
        self.assertGreater((b["fp_logp"].argmax(1) == b["person"]).mean(), 0.9)   # identity is easy
        self.assertEqual(b["fp_feat_test"].shape[0], n)                           # loop D inputs
        self.assertEqual(b["fp_feat_calib"].shape[0], int(d["calib"].sum()))

    def _tmp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        return self._td.name


if __name__ == "__main__":
    unittest.main()
