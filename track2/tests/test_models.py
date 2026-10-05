import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import make_model, shallow_kwargs  # noqa: E402


class TestModels(unittest.TestCase):
    def test_shallow_constants_scale_with_sfreq(self):
        self.assertEqual(shallow_kwargs(250.0), dict(filter_time_length=25,
                         pool_time_length=75, pool_time_stride=15))
        self.assertEqual(shallow_kwargs(120.0), dict(filter_time_length=12,
                         pool_time_length=36, pool_time_stride=7))
        self.assertEqual(shallow_kwargs(500.0), dict(filter_time_length=50,
                         pool_time_length=150, pool_time_stride=30))

    def test_output_shapes(self):
        for arch in ("eegnet", "shallow"):
            m = make_model(arch, n_chans=27, n_outputs=3, n_times=480, sfreq=120.0).eval()
            self.assertEqual(tuple(m(torch.zeros(4, 27, 480)).shape), (4, 3), arch)

    def test_unknown_arch_raises(self):
        with self.assertRaises(ValueError):
            make_model("resnet", 27, 2, 480, 120.0)


if __name__ == "__main__":
    unittest.main()
