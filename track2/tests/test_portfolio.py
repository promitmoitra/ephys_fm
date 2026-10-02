import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(HERE),
                str(REPO / "research" / "expert-weights" / "src"),
                str(REPO / "research" / "expert-portfolio" / "src")]
from submission import LogLinearCombiner, PortfolioCombiner, build_model  # noqa: E402
from portfolio import PortfolioLogLinear  # noqa: E402
from combine import RelLogLinear  # noqa: E402


def logp(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(*shape, generator=g, dtype=torch.float64), -1)


class TestPortfolio(unittest.TestCase):
    def test_inference_one_stream_equals_c3(self):
        K, C = 4, 2
        rel = torch.rand(K, dtype=torch.float64)
        c3 = LogLinearCombiner(K, C)
        c3.coef.copy_(torch.tensor([0.81, 0.96, 1.70]))
        c3.class_bias.copy_(torch.tensor([0.0, -0.016])); c3.rel.copy_(rel)
        pc = PortfolioCombiner(1, K, C)
        pc.w.copy_(torch.tensor([0.81])); pc.c.copy_(torch.tensor([0.96, 1.70]))
        pc.class_bias.copy_(torch.tensor([0.0, -0.016])); pc.rel.copy_(rel)
        e, r = logp(5, K, C, seed=1), logp(5, K, C, seed=2)
        self.assertTrue(torch.allclose(c3(e, r), pc([e], r)))

    def test_research_one_stream_equals_rel_loglinear(self):
        rng = np.random.default_rng(0)
        n, C = 300, 2
        F = {"eegnet": torch.log_softmax(torch.tensor(rng.standard_normal((n, C))), -1).numpy(),
             "ts": torch.log_softmax(torch.tensor(rng.standard_normal((n, C))), -1).numpy(),
             "rel": rng.uniform(0.4, 1.0, (n, 1))}
        y = rng.integers(0, 2, n)
        a = RelLogLinear(["eegnet", "ts", "rel"]).fit(F, y).predict(F)
        b = PortfolioLogLinear(["eegnet", "ts", "rel"]).fit(F, y).predict(F)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_old_config_loads_without_new_keys(self):
        sub = REPO / "outputs" / "t2-integration" / "fb_c3" / "submission"
        if not sub.exists():
            self.skipTest("integration package not present")
        config = json.loads((sub / "config.json").read_text())
        state = torch.load(sub / "mixture.pt", map_location="cpu", weights_only=True)
        meta = {"ch_names": config["ch_names"], "n_times": config["n_times"],
                "n_classes": config["n_classes"], "device": "cpu",
                "sfreq": config["sfreq"]}
        m = build_model(meta, config, state)
        self.assertIsNone(m.reve)
        self.assertEqual(len(m.shallow_experts), 0)

    def test_sfreq_mismatch_raises(self):
        sub = REPO / "outputs" / "t2-integration" / "fb_c3" / "submission"
        if not sub.exists():
            self.skipTest("integration package not present")
        config = json.loads((sub / "config.json").read_text())
        meta = {"ch_names": config["ch_names"], "n_times": config["n_times"],
                "n_classes": config["n_classes"], "device": "cpu", "sfreq": 500.0}
        with self.assertRaisesRegex(ValueError, "sfreq"):
            build_model(meta, config, None)

    def test_inline_shallow_matches_models(self):
        import submission
        from models import make_model
        for sf, n_times in ((120.0, 480), (250.0, 1000), (500.0, 2000)):
            a = submission._shallow(27, 2, n_times, sf)
            b = make_model("shallow", 27, 2, n_times, sf)
            self.assertEqual([p.shape for p in a.parameters()],
                             [p.shape for p in b.parameters()], sf)

    def test_masks_default_and_sim2(self):
        import numpy as np
        from train_mixture import masks
        d = {"subject": np.array(["1", "1", "61", "61", "2", "2"]),
             "run": np.array([0, 4, 0, 4, 1, 5]),
             "split": np.array(["train", "train", "test", "test", "train", "train"])}
        tr, cal, hid, people = masks(d)
        self.assertEqual(people, ["61"])
        self.assertEqual(tr.tolist(), [True, True, False, False, True, True])
        self.assertEqual(cal.tolist(), [False, False, True, False, False, False])
        tr, cal, hid, people = masks(d, eval_people=["2"])
        self.assertEqual(people, ["2"])
        self.assertEqual(tr.tolist(), [True, True, True, True, False, False])  # 61 becomes training
        self.assertEqual(cal.tolist(), [False, False, False, False, True, False])
        self.assertEqual(hid.tolist(), [False, False, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
