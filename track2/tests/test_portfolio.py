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

    def test_research_combiner_handles_four_classes(self):
        rng = np.random.default_rng(1)
        F = {k: torch.log_softmax(torch.tensor(rng.standard_normal((5, 4))), -1).numpy()
             for k in ("eegnet", "ts")} | {"rel": rng.uniform(0.4, 1.0, (5, 1))}
        comb = PortfolioLogLinear(["eegnet", "ts", "rel"])
        comb.theta["b"] = torch.zeros(3, dtype=torch.float64)
        lp = comb.predict(F)
        self.assertEqual(lp.shape, (5, 4))
        np.testing.assert_allclose(np.exp(lp).sum(-1), 1.0, atol=1e-9)

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


class TestFingerprintTwoClasses(unittest.TestCase):
    def test_two_class_fingerprint_exports_two_logits(self):
        """sklearn's binary LR has one weight row; the export must still give 2 logits that
        reproduce its probabilities (2 enrolled people, or a 2-class page solver)."""
        import riemann_parts
        from submission import FBFingerprint
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 6, 480)).astype(np.float32)
        lab = np.repeat([0, 1], 30)
        X[lab == 1] *= np.linspace(0.5, 2.0, 6)[None, :, None].astype(np.float32)
        st = riemann_parts.fit_fingerprint(X, lab, 120.0)
        fp = FBFingerprint(len(riemann_parts.FP_BANDS), 6, 480, 2)
        fp.load_state_dict(st)
        with torch.inference_mode():
            p = torch.softmax(fp(torch.from_numpy(X)), 1).numpy()
        self.assertEqual(p.shape, (60, 2))
        self.assertGreater((p.argmax(1) == lab).mean(), 0.9)


class TestReviewFixes(unittest.TestCase):
    """Final-review findings I1, I2, I3, M2, M3, M4: each pinned before its fix."""

    def test_i1_extra_streams_need_the_portfolio_combiner(self):
        from submission import FingerprintMixture
        with self.assertRaisesRegex(ValueError, "portfolio"):
            FingerprintMixture(27, 480, 2, 3, riemann_n_times_out=420, shallow=True, combiner="C3")
        from train_mixture import check_stream_args
        with self.assertRaises(SystemExit):
            check_stream_args(shallow=True, reve=False, combiner="C3", coef_path=None)
        with self.assertRaises(SystemExit):
            check_stream_args(shallow=False, reve=False, combiner="portfolio", coef_path=None)
        check_stream_args(shallow=True, reve=False, combiner="portfolio", coef_path="x.json")

    def test_i2_non_original_runs_never_write_the_original_package(self):
        from train_mixture import OUT, resolve_out
        rel = Path("outputs/track2_dreyer_sim")                       # README-style relative path
        self.assertEqual(resolve_out(OUT, original=True), OUT)
        for out in (OUT, rel, OUT / "." ):
            with self.assertRaises(SystemExit):
                resolve_out(out, original=False)
        self.assertEqual(resolve_out(Path("/tmp/elsewhere"), original=False), Path("/tmp/elsewhere"))

    def test_i3_research_combiner_fits_three_classes(self):
        rng = np.random.default_rng(2)
        n, C = 90, 3
        y = rng.integers(0, C, n)
        lsm = lambda z: torch.log_softmax(torch.tensor(z), -1).numpy()   # noqa: E731
        F = {"eegnet": lsm(3 * np.eye(C)[y] + rng.standard_normal((n, C))),
             "ts": lsm(rng.standard_normal((n, C))), "rel": rng.uniform(0.4, 1.0, (n, 1))}
        comb = PortfolioLogLinear(["eegnet", "ts", "rel"], n_classes=C).fit(F, y)
        self.assertEqual(comb.theta["b"].shape[0], C - 1)
        self.assertEqual(comb.predict(F).shape, (n, C))

    def test_i3_m3_packaging_checks_streams_and_bias_length(self):
        from train_mixture import portfolio_state
        rel = torch.zeros(4, dtype=torch.float64)
        coef = {"streams": ["eegnet", "shallow"], "w": [0.5, 0.4], "c": [0.8, 1.3], "b": [0.1, -0.2]}
        st = portfolio_state(coef, ["eegnet", "shallow"], n_classes=3, rel=rel)
        self.assertEqual(st["class_bias"].tolist(), [0.0, 0.1, -0.2])
        with self.assertRaisesRegex(ValueError, "order"):
            portfolio_state(coef | {"streams": ["eegnet", "reve"]}, ["eegnet", "shallow"], 3, rel)
        with self.assertRaisesRegex(ValueError, "bias"):
            portfolio_state(coef, ["eegnet", "shallow"], n_classes=2, rel=rel)
        with self.assertRaisesRegex(ValueError, "streams"):
            portfolio_state({k: v for k, v in coef.items() if k != "streams"}, ["eegnet", "shallow"], 3, rel)

    def test_m2_short_shallow_state_raises(self):
        sub = REPO / "outputs" / "t2-integration" / "fb_c3" / "submission"
        if not sub.exists():
            self.skipTest("integration package not present")
        config = json.loads((sub / "config.json").read_text())
        state = torch.load(sub / "mixture.pt", map_location="cpu", weights_only=True)
        config = config | {"shallow_experts": True, "combiner": "portfolio"}
        state = dict(state) | {"shallow_experts": [], "combiner": {
            "w": torch.zeros(2, dtype=torch.float64), "c": torch.zeros(2, dtype=torch.float64),
            "class_bias": torch.zeros(2, dtype=torch.float64), "rel": state["combiner"]["rel"]}}
        meta = {"ch_names": config["ch_names"], "n_times": 480, "n_classes": 2, "device": "cpu",
                "sfreq": 120.0}
        with self.assertRaisesRegex(ValueError, "shallow"):
            build_model(meta, config, state)

    def test_m4_masks_validates_eval_people(self):
        from train_mixture import masks
        d = {"subject": np.array(["1", "1", "61", "61", "2", "2", "70", "70"]),
             "run": np.array([0, 4, 0, 4, 1, 5, 0, 4]),
             "split": np.array(["train", "train", "test", "test", "train", "train", "val", "val"])}
        with self.assertRaisesRegex(ValueError, "not found"):
            masks(d, eval_people=["2", "3"])
        with self.assertRaisesRegex(ValueError, "val"):
            masks(d, eval_people=["70"])
