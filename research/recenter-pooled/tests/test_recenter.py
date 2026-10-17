import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "external" / "2026-competition"),
    str(REPO / "track2"),
    str(REPO / "research" / "integration-harness" / "src"),
    str(HERE / "src"),
]
import recenter  # noqa: E402


def _synthetic_shifted_dataset(K=3, n_classes=3, n_chans=4, per=30, seed=0):
    """Synthetic dataset where each person has a distinct spatial mixing matrix A_k,
    and task class c amplifies channel c in the person's intrinsic space."""
    rng = np.random.default_rng(seed)
    t = np.arange(480) / 120.0
    X_cal, y_cal, p_cal = [], [], []
    X_te, y_te, p_te = [], [], []
    # Distinct orthogonal + diagonal scaling per person so uncentered covs clash across people
    perm = [[0, 1, 2, 3], [1, 2, 0, 3], [2, 0, 1, 3]]
    for k in range(K):
        P = np.eye(n_chans)[perm[k % 3]] * (1.0 + 3.0 * k)
        for sess in (0, 1):
            for i in range(per):
                c = i % n_classes
                src = rng.standard_normal((n_chans, 480)) * 0.3
                src[c] += 2.0 * np.sin(2 * np.pi * 12.0 * t)
                x = (P @ src).astype(np.float32)
                if sess == 0:
                    X_cal.append(x)
                    y_cal.append(c)
                    p_cal.append(k)
                else:
                    X_te.append(x)
                    y_te.append(c)
                    p_te.append(k)
    return (
        np.stack(X_cal),
        np.array(y_cal),
        np.array(p_cal),
        np.stack(X_te),
        np.array(y_te),
        np.array(p_te),
    )


class TestRecenter(unittest.TestCase):
    def test_recenter_mean_is_zero_per_person(self):
        X_cal, y_cal, p_cal, X_te, y_te, p_te = _synthetic_shifted_dataset()
        model = recenter.fit_recenter_8_30(X_cal, y_cal, p_cal, p_cal, K_eval=3, n_classes=3, sfreq=120.0)
        # Each person's calibration tangent vectors must have zero mean (identity covariance)
        for k in range(3):
            v_k = model["cal_tangent"][p_cal == k]
            self.assertLess(float(np.abs(v_k.mean(axis=0)).max()), 1e-4)

    def test_whitener_interpolation_matches_oracle_at_one_hot_posterior(self):
        X_cal, y_cal, p_cal, X_te, y_te, p_te = _synthetic_shifted_dataset()
        model = recenter.fit_recenter_8_30(X_cal, y_cal, p_cal, p_cal, K_eval=3, n_classes=3, sfreq=120.0)
        # One-hot posterior on true person
        one_hot = np.full((len(p_te), 3), -30.0)
        one_hot[np.arange(len(p_te)), p_te] = 0.0
        one_hot = one_hot - np.log(np.exp(one_hot).sum(-1, keepdims=True))

        lp_oracle = model["per_person_logp"](X_te)[np.arange(len(p_te)), p_te]
        lp_interp = model["whitener_interp_logp"](X_te, one_hot)
        np.testing.assert_allclose(lp_interp, lp_oracle, atol=1e-5)

    def test_pytorch_recenter_module_matches_reference_for_2_and_4_classes(self):
        for n_classes in (2, 4):
            X_cal, y_cal, p_cal, X_te, _, _ = _synthetic_shifted_dataset(
                K=3, n_classes=n_classes, n_chans=4, per=24, seed=n_classes
            )
            model = recenter.fit_recenter_8_30(
                X_cal, y_cal, p_cal, p_cal, K_eval=3, n_classes=n_classes, sfreq=120.0
            )
            mod = recenter.RecenterRiemann(3, 4, 480, model["n_times_out"], n_classes)
            mod.load_state_dict(model["state_dict"])
            Xt = torch.from_numpy(X_te[:8])
            with torch.inference_mode():
                lp_torch = mod(Xt).numpy()
            lp_ref = model["per_person_logp"](X_te[:8])
            np.testing.assert_allclose(lp_torch, lp_ref, atol=1e-5)

    def test_level2_integration_arms_handle_2_3_and_4_classes(self):
        rng = np.random.default_rng(7)
        for C in (2, 3, 4):
            n, K = 10, 3
            z_fp = rng.standard_normal((n, K))
            fp_logp = z_fp - np.log(np.exp(z_fp).sum(-1, keepdims=True))
            z_eeg = rng.standard_normal((n, K, C))
            eeg_logp = z_eeg - np.log(np.exp(z_eeg).sum(-1, keepdims=True))
            z_rie = rng.standard_normal((n, K, C))
            riemann_logp = z_rie - np.log(np.exp(z_rie).sum(-1, keepdims=True))
            z_rec = rng.standard_normal((n, K, C))
            recenter_logp = z_rec - np.log(np.exp(z_rec).sum(-1, keepdims=True))
            z_pld = rng.standard_normal((n, C))
            pooled_logp = z_pld - np.log(np.exp(z_pld).sum(-1, keepdims=True))
            bank = {
                "fp_logp": fp_logp,
                "eeg_logp": eeg_logp,
                "riemann_logp": riemann_logp,
                "pooled_logp": pooled_logp,
                "rel": np.array([0.6, 0.7, 0.8]),
                "person": np.arange(n) % K,
            }
            arms = recenter.level2_arms(bank, recenter_logp, recenter_rel=np.array([0.65, 0.65, 0.65]))
            for name in (
                "E1_fully_pooled_recenter",
                "E1_oracle",
                "E2_eeg_experts_plus_recenter",
                "E2_oracle",
                "E3_eeg_experts_plus_shrunk_riemann",
                "E3_oracle",
            ):
                self.assertEqual(arms[name].shape, (n, C), (C, name))
                np.testing.assert_allclose(np.exp(arms[name]).sum(-1), 1.0, atol=1e-6)


class TestEvaluateRecenter(unittest.TestCase):
    def test_clopper_pearson_and_paired_summary(self):
        import evaluate_recenter

        lo, hi = evaluate_recenter.clopper_pearson_ci(14, 17)
        self.assertAlmostEqual(lo, 0.566, places=3)
        self.assertAlmostEqual(hi, 0.962, places=3)

        a = {"bnci2014": [0.70, 0.80], "bnci2015": [0.60], "zhou2016": [0.90]}
        b = {"bnci2014": [0.65, 0.80], "bnci2015": [0.55], "zhou2016": [0.92]}
        s = evaluate_recenter.paired_summary(a, b)
        self.assertEqual(s["better"], 2)
        self.assertEqual(s["worse"], 1)
        self.assertEqual(s["tied"], 1)
        self.assertIn("win_ci95", s)


if __name__ == "__main__":
    unittest.main()

