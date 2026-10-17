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


class TestOnlineRecenter(unittest.TestCase):
    def _synthetic_cross_day_drift_dataset(self, K=3, n_classes=2, n_chans=4, per=40, seed=11):
        rng = np.random.default_rng(seed)
        t = np.arange(480) / 120.0
        X_cal, y_cal, p_cal = [], [], []
        X_te, y_te, p_te = [], [], []
        # Day-2 multiplicative impedance drift per person that biases one task channel over the other
        drifts = [
            np.diag([0.25, 3.0, 1.0, 1.0]),
            np.diag([1.0, 0.25, 3.0, 1.0]),
            np.diag([3.0, 1.0, 0.25, 1.0]),
        ]
        for k in range(K):
            for sess in (0, 1):
                D = np.eye(n_chans) if sess == 0 else drifts[k]
                for i in range(per):
                    c = i % n_classes
                    src = rng.standard_normal((n_chans, 480)) * 0.4
                    # Person-specific task channel pair
                    ch_task = (k + c) % n_chans
                    src[ch_task] += 1.6 * np.sin(2 * np.pi * 12.0 * t)
                    x = (D @ src).astype(np.float32)
                    if sess == 0:
                        X_cal.append(x)
                        y_cal.append(c)
                        p_cal.append(k)
                    else:
                        X_te.append(x)
                        y_te.append(c)
                        p_te.append(k)
        p_te_arr = np.array(p_te)
        # Near-accurate fingerprint log-posteriors
        fp_logits = np.full((len(p_te_arr), K), -4.0)
        fp_logits[np.arange(len(p_te_arr)), p_te_arr] = 0.0
        fp_logp = fp_logits - np.log(np.exp(fp_logits).sum(-1, keepdims=True))
        return (
            np.stack(X_cal),
            np.array(y_cal),
            np.array(p_cal),
            np.stack(X_te),
            np.array(y_te),
            p_te_arr,
            fp_logp,
        )

    def test_online_adaptation_recovers_cross_day_session_drift_on_synthetic(self):
        import online_recenter

        X_cal, y_cal, p_cal, X_te, y_te, p_te, fp_logp = self._synthetic_cross_day_drift_dataset()
        experts = online_recenter.fit_per_person_experts(X_cal, y_cal, p_cal, K=3, n_classes=2, sfreq=120.0)
        arms = online_recenter.compute_all_arms(experts, X_te, fp_logp, p_te)

        # Under severe Day-2 channel scaling drift, static calibration reference drops sharply,
        # while both causal shuffled EMA and SPD session adaptation recover high accuracy.
        acc_static = float((arms["static_cal_ref"][np.arange(len(y_te)), p_te].argmax(-1) == y_te).mean())
        acc_ema_shuf = float((arms["ema_tangent_shuffled"][np.arange(len(y_te)), p_te].argmax(-1) == y_te).mean())
        acc_spd = float((arms["session_fp_spd"][np.arange(len(y_te)), p_te].argmax(-1) == y_te).mean())
        self.assertGreater(acc_ema_shuf, acc_static + 0.15)
        self.assertGreater(acc_spd, acc_static + 0.15)

    def test_session_adaptation_is_permutation_invariant(self):
        import online_recenter

        X_cal, y_cal, p_cal, X_te, _, p_te, fp_logp = self._synthetic_cross_day_drift_dataset(per=16)
        experts = online_recenter.fit_per_person_experts(X_cal, y_cal, p_cal, K=3, n_classes=2, sfreq=120.0)
        arms_orig = online_recenter.compute_all_arms(experts, X_te, fp_logp, p_te)

        perm = np.random.default_rng(99).permutation(len(X_te))
        inv_perm = np.argsort(perm)
        arms_perm = online_recenter.compute_all_arms(experts, X_te[perm], fp_logp[perm], p_te[perm])

        for name in ("session_fp_tangent", "session_fp_spd", "oracle_session_spd"):
            np.testing.assert_allclose(arms_orig[name], arms_perm[name][inv_perm], atol=1e-5)

    def test_causal_ema_has_zero_future_leakage(self):
        import online_recenter

        X_cal, y_cal, p_cal, X_te, _, p_te, fp_logp = self._synthetic_cross_day_drift_dataset(per=12)
        experts = online_recenter.fit_per_person_experts(X_cal, y_cal, p_cal, K=3, n_classes=2, sfreq=120.0)
        arms_1 = online_recenter.compute_all_arms(experts, X_te, fp_logp, p_te)

        X_te_mod = X_te.copy()
        X_te_mod[15:] *= 50.0  # Corrupt future windows >= 15
        arms_2 = online_recenter.compute_all_arms(experts, X_te_mod, fp_logp, p_te)

        for name in ("ema_tangent_ordered", "ema_spd_ordered"):
            np.testing.assert_allclose(arms_1[name][:15], arms_2[name][:15], atol=1e-6)


class TestMuBand(unittest.TestCase):
    def test_estimate_mu_peak_recovers_true_oscillation_frequency(self):
        import mu_band

        rng = np.random.default_rng(42)
        sfreq = 120.0
        t = np.arange(480) / sfreq
        for target_f in (8.2, 12.4):
            X = rng.standard_normal((20, 4, 480)).astype(np.float32) * 0.3
            X += (2.5 * np.sin(2 * np.pi * target_f * t))[None, None, :].astype(np.float32)
            f_est = mu_band.estimate_mu_peak(X, sfreq=sfreq)
            self.assertAlmostEqual(f_est, target_f, delta=0.4)

    def test_mu_shifted_band_beats_fixed_8_30_when_mu_is_near_7hz(self):
        import mu_band

        rng = np.random.default_rng(19)
        sfreq = 120.0
        t = np.arange(480) / sfreq
        X_cal, y_cal, p_cal = [], [], []
        X_te, y_te, p_te = [], [], []
        # Person 0 has a slow 7.2 Hz mu rhythm; task modulates channel c at 7.0 Hz
        for sess in (0, 1):
            for k, f_mu in enumerate((7.2, 10.5)):
                for i in range(30):
                    c = i % 2
                    x = rng.standard_normal((4, 480)) * 1.5
                    # Baseline mu peak on channel 3 so estimate_mu_peak detects 7.2 Hz for k=0
                    x[3] += 3.0 * np.sin(2 * np.pi * f_mu * t)
                    # Task modulation at 5.8 Hz for k=0 (inside [5.5, 27.2] Hz, attenuated by [8, 30] Hz)
                    f_task = 5.8 if k == 0 else 10.0
                    x[c] += 1.4 * np.sin(2 * np.pi * f_task * t)
                    if sess == 0:
                        X_cal.append(x.astype(np.float32))
                        y_cal.append(c)
                        p_cal.append(k)
                    else:
                        X_te.append(x.astype(np.float32))
                        y_te.append(c)
                        p_te.append(k)
        X_cal, y_cal, p_cal = np.stack(X_cal), np.array(y_cal), np.array(p_cal)
        X_te, y_te, p_te = np.stack(X_te), np.array(y_te), np.array(p_te)
        fp_logits = np.full((len(p_te), 2), -6.0)
        fp_logits[np.arange(len(p_te)), p_te] = 0.0
        fp_logp = fp_logits - np.log(np.exp(fp_logits).sum(-1, keepdims=True))

        res = mu_band.evaluate_mu_arms_on_split(X_cal, y_cal, p_cal, X_te, fp_logp, K=2, n_classes=2, sfreq=sfreq)
        k0 = p_te == 0
        acc_fixed = float((res["arms_logp"]["fixed_8_30_static"][k0, 0].argmax(-1) == y_te[k0]).mean())
        acc_shift = float((res["arms_logp"]["mu_shift_wide_static"][k0, 0].argmax(-1) == y_te[k0]).mean())
        self.assertLess(res["mu_peaks"][0], 7.7)
        self.assertGreater(acc_shift, acc_fixed + 0.15)


if __name__ == "__main__":
    unittest.main()



