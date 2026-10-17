import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2"), str(HERE / "src")]
import metrics  # noqa: E402


def logp(*shape, seed=0):
    z = np.random.default_rng(seed).standard_normal(shape)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


class TestMetrics(unittest.TestCase):
    def test_c3_works_for_four_classes(self):
        b = {"eeg_logp": logp(5, 3, 4), "riemann_logp": logp(5, 3, 4, seed=1), "rel": np.array([.5, .7, .9])}
        lp = metrics.c3_logp(b)
        self.assertEqual(lp.shape, (5, 3, 4))
        np.testing.assert_allclose(np.exp(lp).sum(-1), 1, atol=1e-9)

    def test_oracle_row_picks_true_person(self):
        b = {"eeg_logp": logp(4, 2, 2), "riemann_logp": logp(4, 2, 2, seed=1), "rel": np.array([.6, .8]),
             "fp_logp": np.log(np.full((4, 2), .5)), "pooled_logp": logp(4, 2, seed=2),
             "person": np.array([0, 1, 0, 1])}
        r = metrics.rows(b)
        np.testing.assert_allclose(r["oracle"], metrics.c3_logp(b)[np.arange(4), b["person"]])

    def test_pooled_comparison_counts_people_not_datasets(self):
        cmp = metrics.pooled_comparison({"a": (np.array([.8, .9]), np.array([.7, .9])),
                                         "b": (np.array([.6]), np.array([.65]))})
        self.assertEqual(cmp["n"], 3)
        self.assertAlmostEqual(cmp["mean"], (0.1 + 0 - 0.05) / 3)
        self.assertEqual((cmp["better"], cmp["worse"]), (1, 1))

    def test_adopt_rule(self):
        self.assertTrue(metrics.adopt({"mean": .01, "ci": (.002, .02)}, sim2_delta=-.005)[0])
        self.assertFalse(metrics.adopt({"mean": .01, "ci": (-.001, .02)}, sim2_delta=0)[0])
        self.assertFalse(metrics.adopt({"mean": .004, "ci": (.001, .01)}, sim2_delta=0)[0])
        self.assertFalse(metrics.adopt({"mean": .02, "ci": (.01, .03)}, sim2_delta=-.011)[0])

    def test_seed_mean_scores_averages_seeds_per_person(self):
        import tempfile
        y = np.array([0, 1, 0, 1])
        person = np.array([0, 0, 1, 1])
        preds = {0: np.array([0, 1, 1, 0]),   # person 0 right, person 1 wrong
                 1: np.array([0, 1, 0, 1])}   # both right
        with tempfile.TemporaryDirectory() as d:
            for s, p in preds.items():
                np.savez(Path(d) / f"bank_toy_seed{s}.npz", y=y, person=person, eeg_logp=logp(4, 2, 2),
                         pred=p)
            sc = metrics.seed_mean_scores("toy", lambda b: np.log(np.eye(2)[b["pred"]] + 1e-9),
                                          seeds=(0, 1), bank_dir=d)
        np.testing.assert_allclose(sc, [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
