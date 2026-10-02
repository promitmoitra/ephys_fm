import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent / "external" / "2026-competition"), str(HERE)]
import reve_parts  # noqa: E402
from submission import ReveProbe, load_reve_encoder, standardize_clip  # noqa: E402


class StubEncoder(torch.nn.Module):
    """Mimics REVE's return_features output: (B, C, P, 512) from (B, C, T)."""

    def forward(self, x, pos=None, return_features=False):
        B, C, T = x.shape
        P = 4
        f = x[:, :, : P * 128].reshape(B, C, P, 128).repeat(1, 1, 1, 4)
        return {"features": f, "cls_token": None}


class TestReveParts(unittest.TestCase):
    def test_resample_matrix_matches_scipy(self):
        from scipy.signal import resample_poly
        R = reve_parts.resample_matrix(480, 120.0, 200.0)
        self.assertEqual(R.shape, (480, 800))
        x = np.random.default_rng(0).standard_normal((3, 27, 480))
        np.testing.assert_allclose(x @ R, resample_poly(x, 5, 3, axis=-1), atol=1e-10)

    def test_resample_matrix_500hz(self):
        self.assertEqual(reve_parts.resample_matrix(2000, 500.0, 200.0).shape, (2000, 800))

    def test_positions_missing_channel_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "reve_positions.json"
            f.write_text(json.dumps({"C3": [0.1, 0.2, 0.3], "C4": [0.4, 0.5, 0.6]}))
            np.testing.assert_allclose(reve_parts.positions(["C4", "C3"], f),
                                       [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]])
            with self.assertRaisesRegex(ValueError, "EOG1"):
                reve_parts.positions(["C3", "EOG1"], f)

    def test_standardize_clip_flat_channel_no_nan(self):
        X = torch.randn(2, 3, 100) * 50
        X[:, 1] = 7.0                                              # flat channel
        Z = standardize_clip(X)
        self.assertFalse(torch.isnan(Z).any())
        self.assertLessEqual(Z.abs().max().item(), 15.0)
        self.assertTrue(torch.allclose(Z[:, 1], torch.zeros_like(Z[:, 1])))

    def test_fit_head_separable(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((200, 16))
        y = (Z[:, 0] > 0).astype(int)
        W, b = reve_parts.fit_head(Z, y, n_classes=2, lam=1e-3)
        self.assertGreater(((Z @ W.T + b).argmax(1) == y).mean(), 0.95)

    def test_person_heads_shrink_to_pooled(self):
        rng = np.random.default_rng(1)
        Z = rng.standard_normal((60, 8)); y = rng.integers(0, 2, 60); p = np.repeat([0, 1], 30)
        W0, b0 = reve_parts.fit_head(Z, y, 2, lam=1e-3)
        W, b = reve_parts.fit_person_heads(Z, y, p, 2, 2, W0, b0, lam=1e6)
        np.testing.assert_allclose(W[0], W0, atol=1e-3)

    def test_probe_forward_shapes_and_export_roundtrip(self):
        rng = np.random.default_rng(2)
        R = reve_parts.resample_matrix(480, 120.0, 200.0)
        pos = rng.standard_normal((27, 3))
        mu, sd = np.zeros(512), np.ones(512)
        W, b = rng.standard_normal((3, 2, 512)), rng.standard_normal((3, 2))
        state = reve_parts.export(R, pos, mu, sd, W, b)
        probe = ReveProbe(StubEncoder(), n_people=3, n_classes=2, n_chans=27, n_times=480,
                          n_out=800)
        probe.load_state_dict(state)
        lp = probe(torch.randn(5, 27, 480))
        self.assertEqual(tuple(lp.shape), (5, 3, 2))
        self.assertTrue(torch.allclose(lp.exp().sum(-1), torch.ones(5, 3, dtype=lp.dtype)))

    def test_probe_load_rejects_missing_head_buffers(self):
        rng = np.random.default_rng(3)
        state = reve_parts.export(reve_parts.resample_matrix(480, 120.0, 200.0),
                                  rng.standard_normal((27, 3)), np.zeros(512), np.ones(512),
                                  rng.standard_normal((3, 2, 512)), rng.standard_normal((3, 2)))
        del state["weight"]
        probe = ReveProbe(StubEncoder(), n_people=3, n_classes=2, n_chans=27, n_times=480,
                          n_out=800)
        with self.assertRaisesRegex(RuntimeError, "weight"):
            probe.load_state_dict(state)

    def test_load_encoder_sets_offline_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                load_reve_encoder(Path(tmp), _constructor=lambda: torch.nn.Identity())
            finally:
                self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
                self.assertEqual(os.environ.get("REVE_POSITIONS_PATH"), tmp)


if __name__ == "__main__":
    unittest.main()
