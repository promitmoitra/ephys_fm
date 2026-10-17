import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.signal as sps
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import complex_fp  # noqa: E402


class TestComplexFP(unittest.TestCase):
    def test_hermitian_cov_and_frechet_mean_are_hpd(self):
        torch.manual_seed(0)
        Z = torch.randn(16, 5, 120, dtype=torch.complex64)
        covs = complex_fp.hermitian_cov(Z, rho=0.05)
        self.assertEqual(covs.shape, (16, 5, 5))
        self.assertTrue(torch.allclose(covs, covs.mH, atol=1e-6))
        w, _ = torch.linalg.eigh(covs)
        self.assertTrue((w > 0).all())

        ref = complex_fp.complex_frechet_mean(covs)
        self.assertEqual(ref.shape, (5, 5))
        self.assertTrue(torch.allclose(ref, ref.mH, atol=1e-6))
        w_ref, _ = torch.linalg.eigh(ref)
        self.assertTrue((w_ref > 0).all())

    def test_complex_tangent_dimension_and_isometry(self):
        torch.manual_seed(1)
        N, C, T = 10, 4, 120
        Z = torch.randn(N, C, T, dtype=torch.complex64)
        covs = complex_fp.hermitian_cov(Z)
        ref = complex_fp.complex_frechet_mean(covs)
        cref_isqrt = complex_fp.herm_invsqrtm(ref)
        vec = complex_fp.complex_tangent(covs, cref_isqrt)
        self.assertEqual(vec.shape, (N, C * C))

        # Norm in vectorized tangent space equals Frobenius norm of log(cref_isqrt @ C @ cref_isqrt)
        M = cref_isqrt @ covs @ cref_isqrt
        w, V = torch.linalg.eigh(M)
        L = (V * torch.log(torch.clamp(w, min=1e-12)).unsqueeze(-2)) @ V.mH
        frob_sq = (L.real ** 2 + L.imag ** 2).sum(dim=(-1, -2))
        vec_sq = (vec ** 2).sum(dim=-1)
        self.assertTrue(torch.allclose(frob_sq, vec_sq, atol=1e-4))

    def test_zero_lag_real_covariance_has_zero_imaginary_tangent(self):
        torch.manual_seed(2)
        N, C = 8, 5
        A = torch.randn(N, C, C)
        real_cov = (A @ A.transpose(-1, -2)) / C + 0.1 * torch.eye(C)
        ref_real = real_cov.mean(dim=0).to(torch.complex64)
        cref_isqrt = complex_fp.herm_invsqrtm(ref_real)
        vec = complex_fp.complex_tangent(real_cov.to(torch.complex64), cref_isqrt)
        n_im = C * (C - 1) // 2
        im_part = vec[:, -n_im:]
        self.assertLess(float(im_part.abs().max()), 1e-5)

    def test_folded_hilbert_filter_matches_scipy(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((6, 4, 240)).astype(np.float32)
        band = (8, 13)
        sfreq = 120.0
        sos = sps.butter(4, list(band), btype="bandpass", fs=sfreq, output="sos")
        Xb = sps.sosfiltfilt(sos, X, axis=-1).astype(np.float32)
        Zb_ref = sps.hilbert(Xb, axis=-1).astype(np.complex64)

        F = complex_fp.folded_analytic_filter(band, sfreq, 240)
        Zb_torch = (torch.from_numpy(X).to(torch.complex64) @ F).numpy()
        np.testing.assert_allclose(Zb_torch, Zb_ref, atol=2e-5, rtol=1e-4)

    def test_torch_export_matches_scipy_sklearn_for_2_and_3_classes(self):
        rng = np.random.default_rng(4)
        for K in (2, 3):
            X_cal, p_cal, X_te = [], [], []
            for k in range(K):
                gain = 1.0 + k * 0.5 + rng.random((4, 1)).astype(np.float32)
                for _ in range(16):
                    X_cal.append(rng.standard_normal((4, 120)).astype(np.float32) * gain)
                    p_cal.append(k)
                for _ in range(6):
                    X_te.append(rng.standard_normal((4, 120)).astype(np.float32) * gain)
            X_cal = np.stack(X_cal)
            p_cal = np.array(p_cal)
            X_te = np.stack(X_te)

            st, ref_fn = complex_fp.fit_complex_fingerprint(
                X_cal, p_cal, sfreq=120.0, bands=complex_fp.BANDS_5
            )
            mod = complex_fp.ComplexFBFingerprint(
                len(complex_fp.BANDS_5), n_chans=4, n_times=120, n_subjects=K
            )
            mod.load_state_dict(st)
            Xt = torch.from_numpy(X_te)
            with torch.inference_mode():
                logits = mod(Xt)
                self.assertTrue(torch.allclose(logits, mod.linear(mod.features(Xt)), atol=1e-5))
                p_torch = torch.softmax(logits.double(), dim=-1).numpy()
            p_ref = np.exp(ref_fn(X_te))
            self.assertLess(float(np.abs(p_torch - p_ref).max()), 1e-4, f"K={K}")


if __name__ == "__main__":
    unittest.main()
