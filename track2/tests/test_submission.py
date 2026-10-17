"""Tests for Track 2 submission: Complex Hermitian fingerprint, 2-band RiemannExperts, and online adaptation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
import sys
sys.path[:0] = [
    str(REPO / "external" / "2026-competition"),
    str(HERE),
]

import riemann_parts  # noqa: E402
from submission import (  # noqa: E402
    ComplexFBFingerprint,
    FBFingerprint,
    FingerprintMixture,
    RiemannExperts,
    build_model,
    complex_tangent,
    hermitian_cov,
    oas,
    tangent,
)


class TestComplexFingerprint(unittest.TestCase):
    def test_hermitian_cov_is_hpd(self):
        B, C, T = 4, 8, 120
        Z = torch.randn(B, C, T, dtype=torch.complex64)
        cov = hermitian_cov(Z, rho=0.05)
        self.assertEqual(cov.shape, (B, C, C))
        # Check Hermitian: cov == cov.mH
        self.assertTrue(torch.allclose(cov, cov.mH, atol=1e-5))
        # Check strictly positive eigenvalues
        vals = torch.linalg.eigvalsh(cov)
        self.assertTrue((vals > 0).all())

    def test_complex_tangent_dimensions(self):
        B, C = 4, 8
        Z = torch.randn(B, C, 120, dtype=torch.complex64)
        cov = hermitian_cov(Z, rho=0.05)
        # reference inverse square root
        ref = cov.mean(dim=0)
        w, V = torch.linalg.eigh(ref)
        ref_isqrt = (V / torch.sqrt(w).unsqueeze(-2)) @ V.mH
        feat = complex_tangent(cov, ref_isqrt)
        self.assertEqual(feat.shape, (B, C * C))
        self.assertEqual(feat.dtype, torch.float32)

    def test_complex_fb_fingerprint_forward_and_batch_invariance(self):
        B, n_bands, n_chans, n_times, n_subjects = 5, 6, 8, 240, 4
        fp = ComplexFBFingerprint(n_bands, n_chans, n_times, n_subjects)
        fp.eval()

        X = torch.randn(B, n_chans, n_times, dtype=torch.float32)
        out = fp(X)
        self.assertEqual(out.shape, (B, n_subjects))

        # Batch invariance: window 0 output must be identical whether evaluated alone or in batch
        out_single = fp(X[:1])
        self.assertTrue(torch.allclose(out[0], out_single[0], atol=1e-5))


class TestRiemannExpertsMultiBand(unittest.TestCase):
    def test_2band_riemann_experts_forward_shapes(self):
        B, K, C, T, T_out, n_classes = 6, 4, 8, 240, 210, 2
        rex = RiemannExperts(
            n_subjects=K,
            n_chans=C,
            n_times=T,
            n_times_out=T_out,
            n_classes=n_classes,
            n_bands=2,
            online_adaptation=True,
            online_alpha=0.05,
        )
        rex.cref_isqrt.copy_(torch.eye(C, dtype=torch.float64).expand(K, 2, C, C))
        rex.filters.copy_(torch.randn_like(rex.filters))
        self.assertEqual(rex.filters.shape, (K, 2, T, T_out))
        self.assertEqual(rex.cref_isqrt.shape, (K, 2, C, C))
        n_feat = 2 * (C * (C + 1) // 2)
        self.assertEqual(rex.weight.shape, (K, n_classes, n_feat))
        self.assertEqual(rex.bias.shape, (K, n_classes))
        self.assertEqual(rex.running_mu.shape, (K, n_feat))

        X = torch.randn(B, C, T, dtype=torch.float64)
        p_subj = torch.softmax(torch.randn(B, K, dtype=torch.float64), dim=-1)
        logp = rex(X, p_subject=p_subj)
        self.assertEqual(logp.shape, (B, K, n_classes))
        # check probabilities sum to 1
        probs = logp.exp().sum(dim=-1)
        self.assertTrue(torch.allclose(probs, torch.ones_like(probs), atol=1e-6))

    def test_legacy_1band_riemann_experts_forward_shapes(self):
        B, K, C, T, T_out, n_classes = 4, 3, 6, 120, 100, 2
        rex = RiemannExperts(
            n_subjects=K,
            n_chans=C,
            n_times=T,
            n_times_out=T_out,
            n_classes=n_classes,
            n_bands=1,
            online_adaptation=False,
        )
        rex.cref_isqrt.copy_(torch.eye(C, dtype=torch.float64).expand(K, C, C))
        rex.filt.copy_(torch.randn_like(rex.filt))
        self.assertTrue(hasattr(rex, "filt"))
        self.assertEqual(rex.filt.shape, (T, T_out))
        self.assertEqual(rex.cref_isqrt.shape, (K, C, C))
        n_feat = C * (C + 1) // 2
        self.assertEqual(rex.weight.shape, (K, n_classes, n_feat))

        X = torch.randn(B, C, T, dtype=torch.float64)
        logp = rex(X)
        self.assertEqual(logp.shape, (B, K, n_classes))


class TestOnlineAdaptationCausality(unittest.TestCase):
    def test_zero_future_leakage(self):
        """Window t's adapted features and logits must not depend on windows t+1..N."""
        B, K, C, T, T_out, n_classes = 6, 3, 6, 120, 100, 2
        rex = RiemannExperts(
            n_subjects=K, n_chans=C, n_times=T, n_times_out=T_out,
            n_classes=n_classes, n_bands=2, online_adaptation=True, online_alpha=0.05,
        )
        rex.cref_isqrt.copy_(torch.eye(C, dtype=torch.float64).expand(K, 2, C, C))
        rex.filters.copy_(torch.randn_like(rex.filters))
        rex.eval()

        X = torch.randn(B, C, T, dtype=torch.float64)
        p_subj = torch.softmax(torch.randn(B, K, dtype=torch.float64), dim=-1)

        # Run windows 0..2
        rex.reset_adaptation()
        logp_partial = rex(X[:3], p_subject=p_subj[:3])

        # Run full sequence with modified future windows 3..5
        X_perturbed = X.clone()
        X_perturbed[3:] = torch.randn(3, C, T, dtype=torch.float64) * 100.0
        p_perturbed = p_subj.clone()
        p_perturbed[3:] = torch.softmax(torch.randn(3, K, dtype=torch.float64), dim=-1)

        rex.reset_adaptation()
        logp_perturbed = rex(X_perturbed, p_subject=p_perturbed)

        # Windows 0..2 must be EXACTLY identical
        self.assertTrue(torch.allclose(logp_partial, logp_perturbed[:3], atol=1e-12))

    def test_chunked_batch_equivalence(self):
        """Processing a stream in chunks of 2 must match processing in one call."""
        B, K, C, T, T_out, n_classes = 8, 3, 6, 120, 100, 2
        rex = RiemannExperts(
            n_subjects=K, n_chans=C, n_times=T, n_times_out=T_out,
            n_classes=n_classes, n_bands=2, online_adaptation=True, online_alpha=0.05,
        )
        rex.cref_isqrt.copy_(torch.eye(C, dtype=torch.float64).expand(K, 2, C, C))
        rex.filters.copy_(torch.randn_like(rex.filters))
        rex.eval()

        X = torch.randn(B, C, T, dtype=torch.float64)
        p_subj = torch.softmax(torch.randn(B, K, dtype=torch.float64), dim=-1)

        # Full run
        rex.reset_adaptation()
        logp_full = rex(X, p_subject=p_subj)

        # Chunked run in chunks of 2
        rex.reset_adaptation()
        logp_chunks = []
        for i in range(0, B, 2):
            logp_chunks.append(rex(X[i : i + 2], p_subject=p_subj[i : i + 2]))
        logp_chunked = torch.cat(logp_chunks, dim=0)

        self.assertTrue(torch.allclose(logp_full, logp_chunked, atol=1e-12))


class TestMuPeakEstimation(unittest.TestCase):
    def test_estimate_mu_peak_detects_oscillation(self):
        sfreq = 120.0
        n_times = int(4.0 * sfreq)
        t = np.arange(n_times) / sfreq
        rng = np.random.default_rng(42)
        for target_f in (8.2, 11.5, 12.8):
            trials = []
            for _ in range(20):
                noise = rng.standard_normal((4, n_times)) * 0.3
                osc = (2.5 * np.sin(2 * np.pi * target_f * t))[None, :]
                trials.append(noise + osc)
            X = np.stack(trials)
            peak = riemann_parts.estimate_mu_peak(X, sfreq)
            self.assertAlmostEqual(peak, target_f, delta=0.5)


class TestFullMixtureWiring(unittest.TestCase):
    def test_fit_and_package_roundtrip(self):
        rng = np.random.default_rng(42)
        K, C, T = 3, 6, 240
        sfreq = 120.0
        n_classes = 2

        X_list, y_list, p_list, run_list = [], [], [], []
        for k in range(K):
            for r in range(3):
                for i in range(8):
                    X_list.append(rng.standard_normal((C, T)).astype(np.float32))
                    y_list.append(i % n_classes)  # balanced classes per run
                    p_list.append(k)
                    run_list.append(r)
        X = np.stack(X_list)
        y = np.array(y_list)
        person = np.array(p_list)
        run = np.array(run_list)

        # Fit complex fingerprint
        fp_state = riemann_parts.fit_complex_fingerprint(X, person, sfreq, bands=[(8, 13), (13, 20)])

        # Fit 2-band riemann experts
        riemann_sd, combiner_sd, rel, n_out = riemann_parts.fit_riemann(
            X, y, person, run, K, n_classes, sfreq, bands_mode="mu_beta_2band"
        )
        self.assertEqual(riemann_sd["filters"].shape[1], 2)
        self.assertEqual(riemann_sd["cref_isqrt"].shape[1], 2)

        # Package state dict
        ch_names = [f"Ch{i}" for i in range(C)]
        from models import make_model
        experts_sd = [make_model("eegnet", C, n_classes, T, sfreq).state_dict() for _ in range(K)]
        state = {
            "fingerprint": fp_state,
            "experts": experts_sd,
            "riemann": riemann_sd,
            "combiner": combiner_sd,
        }
        config = {
            "ch_names": ch_names,
            "n_times": T,
            "n_classes": n_classes,
            "sfreq": sfreq,
            "experts": [f"S{i}" for i in range(K)],
            "fingerprint": "complex_6b",
            "fingerprint_n_bands": 2,
            "riemann_experts": True,
            "riemann_bands": "mu_beta_2band",
            "riemann_n_bands": 2,
            "riemann_n_times_out": int(n_out),
            "online_adaptation": True,
            "online_alpha": 0.05,
            "combiner": "C3",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            torch.save(state, tmp / "mixture.pt")
            (tmp / "config.json").write_text(json.dumps(config))

            meta = {
                "ch_names": ch_names,
                "n_times": T,
                "n_classes": n_classes,
                "sfreq": sfreq,
                "device": "cpu",
                "submission_dir": tmp,
            }
            model = build_model(meta, config, state)
            self.assertIsInstance(model.fingerprint, ComplexFBFingerprint)
            self.assertIsInstance(model.riemann, RiemannExperts)
            self.assertEqual(model.riemann.n_bands, 2)
            self.assertTrue(model.riemann.online_adaptation)

            # Test predict(X)
            preds = model.predict(torch.from_numpy(X[:6]))
            self.assertEqual(preds.shape, (6,))
            self.assertTrue(set(preds.tolist()) <= {0, 1})


if __name__ == "__main__":
    unittest.main()
