import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "track2"), str(HERE / "src")]
import nets  # noqa: E402
from models import make_model  # noqa: E402


class TestNets(unittest.TestCase):
    def test_split_reproduces_forward(self):
        for arch in ("eegnet", "shallow"):
            m = make_model(arch, 22, 4, 480, 120.0).eval()
            trunk, head = nets.split_trunk_head(m)
            x = torch.randn(3, 22, 480)
            with torch.inference_mode():
                self.assertTrue(torch.allclose(m(x), head(trunk(x)), atol=1e-6), arch)
                self.assertEqual(trunk(x).shape[1], nets.feature_channels(arch), arch)

    def test_film_layer_exists(self):
        for arch in ("eegnet", "shallow"):
            names = [n for n, _ in make_model(arch, 22, 4, 480, 120.0).named_children()]
            self.assertIn(nets.FILM_AFTER[arch], names)


if __name__ == "__main__":
    unittest.main()
