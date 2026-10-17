import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import evaluate_2x2  # noqa: E402


class TestEvaluate2x2(unittest.TestCase):
    def test_id_person_metrics_recall_and_nll(self):
        person = np.array([0, 0, 1, 1])
        # Person 0: first window right, second wrong. Person 1: both right.
        probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.2, 0.8],
        ])
        fp_logp = np.log(probs)
        rec, nll = evaluate_2x2.id_person_metrics(person, fp_logp, K=2)
        np.testing.assert_allclose(rec, [0.5, 1.0])
        np.testing.assert_allclose(
            nll,
            [-0.5 * (np.log(0.8) + np.log(0.3)), -0.5 * (np.log(0.9) + np.log(0.8))],
        )

    def test_clopper_pearson_small_n(self):
        lo, hi = evaluate_2x2.clopper_pearson_ci(4, 4)
        self.assertAlmostEqual(lo, 0.3976, places=3)
        self.assertEqual(hi, 1.0)
        lo0, hi0 = evaluate_2x2.clopper_pearson_ci(0, 4)
        self.assertEqual(lo0, 0.0)
        self.assertAlmostEqual(hi0, 1.0 - 0.3976, places=3)

    def test_criterion_b_verdict_enforces_all_three_gates(self):
        good_id_acc = {"mean": 0.015, "ci": (0.002, 0.030)}
        flat_id_acc = {"mean": 0.002, "ci": (-0.005, 0.010)}
        good_id_nll = {"mean": 0.035, "ci": (0.005, 0.070)}
        flat_id_nll = {"mean": 0.005, "ci": (-0.010, 0.020)}
        good_task = {"mean": 0.002, "ci": (-0.001, 0.006)}
        bad_task = {"mean": -0.003, "ci": (-0.008, 0.001)}

        # Passes via ID accuracy
        ok, _ = evaluate_2x2.criterion_b_verdict(good_id_acc, flat_id_nll, good_task, 0.0, 1e-5)
        self.assertTrue(ok)
        # Passes via ID NLL reduction
        ok, _ = evaluate_2x2.criterion_b_verdict(flat_id_acc, good_id_nll, good_task, 0.0, 1e-5)
        self.assertTrue(ok)
        # Fails when neither ID acc nor ID NLL clears bar
        ok, _ = evaluate_2x2.criterion_b_verdict(flat_id_acc, flat_id_nll, good_task, 0.0, 1e-5)
        self.assertFalse(ok)
        # Fails when task degrades
        ok, _ = evaluate_2x2.criterion_b_verdict(good_id_acc, good_id_nll, bad_task, 0.0, 1e-5)
        self.assertFalse(ok)
        # Fails when sim2 degrades > 0.002
        ok, _ = evaluate_2x2.criterion_b_verdict(good_id_acc, good_id_nll, good_task, -0.003, 1e-5)
        self.assertFalse(ok)
        # Fails when export parity >= 1e-4
        ok, _ = evaluate_2x2.criterion_b_verdict(good_id_acc, good_id_nll, good_task, 0.0, 2e-4)
        self.assertFalse(ok)

    def test_routing_split_acc_reads_bank_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            self.assertTrue(evaluate_2x2.BANK_DIR.exists())
            # Create a tiny synthetic bank_toy_seed0.npz
            np.savez(
                td_path / "bank_toy_seed0.npz",
                y=np.array([0, 1, 0, 1]),
                person=np.array([0, 0, 1, 1]),
                eeg_logp=np.log(np.full((4, 2, 2), 0.5)),
                riemann_logp=np.log(np.full((4, 2, 2), 0.5)),
                rel=np.array([0.5, 0.5]),
            )
            fp_ref = np.log(np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.9, 0.1]]))
            res = evaluate_2x2._routing_split_acc("toy", fp_ref, fp_ref, seeds=(0,), bank_dir=td_path)
            self.assertEqual(res["n_right"], 2)
            self.assertEqual(res["n_wrong"], 2)

    def test_format_markdown_report_reads_json_dynamically(self):
        res_json = HERE / "results" / "2x2_results.json"
        import json
        data_dict = json.loads(res_json.read_text())
        md = evaluate_2x2.format_markdown_report(data_dict)
        for arm in evaluate_2x2.ARMS:
            self.assertIn(arm, md)
        self.assertIn("complex_6b", md)
        self.assertIn("Criterion A", md)
        self.assertIn("Criterion B", md)


if __name__ == "__main__":
    unittest.main()
