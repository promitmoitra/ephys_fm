"""Pre-registered evaluation of Protocol 01: Test-time reference covariance adaptation for per-person
task experts (`RiemannExperts`).

Evaluates all 10 pre-registered arms from `experiments/01-protocol/protocol.md` (plus `complex_6b`
fingerprint variants) on the 17 cross-day dev people (`bnci2014`, `bnci2015`, `zhou2016`) and
14 fresh `dreyer_sim2` people, at both Level 1 (standalone `RiemannExperts`) and Level 2 (end-to-end
Track 2 `C3` mixture across seeds 0–2).

Never loads the 8 locked holdout people.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import scipy.signal as sps
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
REPO = EXP_ROOT.parents[1]
sys.path[:0] = [
    str(REPO / "external" / "2026-competition"),
    str(REPO / "track2"),
    str(REPO / "research" / "integration-harness" / "src"),
    str(HERE),
]

import data  # noqa: E402
import evaluate_recenter  # noqa: E402
import metrics  # noqa: E402
import online_recenter  # noqa: E402
import recenter  # noqa: E402

DEV_DATASETS = ("bnci2014", "bnci2015", "zhou2016")
SEEDS = (0, 1, 2)
BANK_DIR = REPO / "outputs" / "t2-int-harness"
RESULTS_DIR = EXP_ROOT / "results"

BANDS_6 = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]

ALL_ARMS = (
    *online_recenter.PROTOCOL01_ARMS,
    "oracle_session_spd_oracle_id",
    "complex6b_static_cal_ref",
    "complex6b_ema_tangent_shuffled",
    "complex6b_session_fp_spd",
)


def _complex_6b_logp(X_cal: np.ndarray, p_cal: np.ndarray, X_te: np.ndarray, sfreq: float, rho: float = 0.05) -> np.ndarray:
    """Self-contained complex_6b fingerprint log-posterior (N_te, K) fitted on calibration windows."""
    n_chans = X_cal.shape[1]
    eye = torch.eye(n_chans, dtype=torch.complex64)
    triu_i, triu_j = torch.triu_indices(n_chans, n_chans, offset=1)
    s2 = 2.0 ** 0.5

    def _band_covs(X: np.ndarray, band) -> torch.Tensor:
        sos = sps.butter(4, list(band), btype="bandpass", fs=sfreq, output="sos")
        Xb = sps.sosfiltfilt(sos, X, axis=-1).astype(np.float32)
        Zb = torch.from_numpy(sps.hilbert(Xb, axis=-1).astype(np.complex64))
        gamma = (Zb @ Zb.mH) / Zb.shape[-1]
        tr = torch.diagonal(gamma.real, dim1=-2, dim2=-1).sum(-1)
        return (1.0 - rho) * gamma + rho * (tr / n_chans)[..., None, None] * eye

    def _frechet(covs: torch.Tensor) -> torch.Tensor:
        ref = covs.mean(dim=0)
        for _ in range(50):
            w, V = torch.linalg.eigh(ref)
            w = torch.clamp(w, min=1e-12)
            ref_isqrt = (V / torch.sqrt(w).unsqueeze(-2)) @ V.mH
            ref_sqrt = (V * torch.sqrt(w).unsqueeze(-2)) @ V.mH
            M = ref_isqrt @ covs @ ref_isqrt
            wm, Vm = torch.linalg.eigh(M)
            wm = torch.clamp(wm, min=1e-12)
            logM = (Vm * torch.log(wm).unsqueeze(-2)) @ Vm.mH
            Vbar = logM.mean(dim=0)
            wv, Vv = torch.linalg.eigh(Vbar)
            expV = (Vv * torch.exp(0.5 * wv).unsqueeze(-2)) @ Vv.mH
            ref_new = ref_sqrt @ expV @ ref_sqrt
            if torch.linalg.norm(ref_new - ref) < 1e-5:
                ref = ref_new
                break
            ref = ref_new
        w, V = torch.linalg.eigh(ref)
        w = torch.clamp(w, min=1e-12)
        return (V / torch.sqrt(w).unsqueeze(-2)) @ V.mH

    def _tan(covs: torch.Tensor, w_isqrt: torch.Tensor) -> np.ndarray:
        M = w_isqrt @ covs @ w_isqrt
        w, V = torch.linalg.eigh(M)
        w = torch.clamp(w, min=1e-12)
        L = (V * torch.log(w).unsqueeze(-2)) @ V.mH
        diag = torch.diagonal(L.real, dim1=-2, dim2=-1)
        re_off = L.real[..., triu_i, triu_j] * s2
        im_off = L.imag[..., triu_i, triu_j] * s2
        return torch.cat([diag, re_off, im_off], dim=-1).numpy()

    cal_feats, te_feats = [], []
    for band in BANDS_6:
        c_cal = _band_covs(X_cal, band)
        w_isqrt = _frechet(c_cal)
        cal_feats.append(_tan(c_cal, w_isqrt))
        te_feats.append(_tan(_band_covs(X_te, band), w_isqrt))
    F_cal = np.hstack(cal_feats).astype(np.float64)
    F_te = np.hstack(te_feats).astype(np.float64)
    sc = StandardScaler().fit(F_cal)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(F_cal), p_cal)
    return lr.predict_log_proba(sc.transform(F_te))


def _load_bank(name: str, seed: int) -> dict:
    z = np.load(BANK_DIR / f"bank_{name}_seed{seed}.npz")
    return {k: z[k] for k in z.files}


def _evaluate_dataset(name: str) -> dict:
    d = data.load_dreyer_sim2() if name == "dreyer_sim2" else data.load(name, part="dev")
    seeds = (0,) if name == "dreyer_sim2" else SEEDS

    cal_m = d["calib"]
    X_cal = d["X"][cal_m]
    y_cal = d["y"][cal_m]
    p_cal = d["person"][cal_m]
    X_te = d["X"][d["test"]]
    K = len(d["people"])
    n_classes = d["n_classes"]
    sfreq = d["sfreq"]

    bank0 = _load_bank(name, 0)
    y_te = bank0["y"]
    p_te = bank0["person"]
    fp_logp = bank0["fp_logp"]
    idx = np.arange(len(y_te))

    # Fit per-person RiemannExperts on calibration data
    experts = online_recenter.fit_per_person_experts(X_cal, y_cal, p_cal, K, n_classes, sfreq)
    arms_k_logp = online_recenter.compute_all_arms(experts, X_te, fp_logp, p_te)

    # Verify static_cal_ref matches shipped bank["riemann_logp"]
    max_diff_bank = float(np.abs(np.exp(arms_k_logp["static_cal_ref"]) - np.exp(bank0["riemann_logp"])).max())
    assert max_diff_bank < 1e-5, f"static_cal_ref mismatch vs bank: {max_diff_bank}"

    # Also compute complex_6b fingerprint posteriors and adapted arms
    fp_c6b_logp = _complex_6b_logp(X_cal, p_cal, X_te, sfreq)
    arms_c6b_k_logp = online_recenter.compute_all_arms(experts, X_te, fp_c6b_logp, p_te)

    # Level 1: Standalone per-person RiemannExperts (routed by fp_logp; oracle_session_spd also with oracle ID)
    l1_routed_logp = {}
    for arm in online_recenter.PROTOCOL01_ARMS:
        l1_routed_logp[arm] = metrics.route(fp_logp, arms_k_logp[arm])
    l1_routed_logp["oracle_session_spd_oracle_id"] = arms_k_logp["oracle_session_spd"][idx, p_te]
    l1_routed_logp["complex6b_static_cal_ref"] = metrics.route(fp_c6b_logp, arms_c6b_k_logp["static_cal_ref"])
    l1_routed_logp["complex6b_ema_tangent_shuffled"] = metrics.route(
        fp_c6b_logp, arms_c6b_k_logp["ema_tangent_shuffled"]
    )
    l1_routed_logp["complex6b_session_fp_spd"] = metrics.route(fp_c6b_logp, arms_c6b_k_logp["session_fp_spd"])

    level1 = {}
    for arm, lp in l1_routed_logp.items():
        scores = metrics.person_scores(y_te, p_te, lp, K).tolist()
        level1[arm] = {"per_person": scores, "mean": float(np.mean(scores))}

    # Level 2: End-to-end Track 2 C3 mixture across seeds
    l2_names = ("row1_pooled_eegnet", "row2_current", "row3_oracle", *ALL_ARMS)
    l2_seed_scores = {arm: [] for arm in l2_names}

    for s in seeds:
        bank_s = _load_bank(name, s)
        ref_rows = metrics.rows(bank_s)
        eeg_logp = bank_s["eeg_logp"]
        rel = bank_s["rel"]

        l2_seed_scores["row1_pooled_eegnet"].append(metrics.person_scores(y_te, p_te, ref_rows["pooled"], K))
        l2_seed_scores["row2_current"].append(metrics.person_scores(y_te, p_te, ref_rows["current"], K))
        l2_seed_scores["row3_oracle"].append(metrics.person_scores(y_te, p_te, ref_rows["oracle"], K))

        for arm in online_recenter.PROTOCOL01_ARMS:
            c3_k = recenter._c3_combine(eeg_logp, arms_k_logp[arm], rel)
            lp_routed = metrics.route(fp_logp, c3_k)
            l2_seed_scores[arm].append(metrics.person_scores(y_te, p_te, lp_routed, K))

        c3_ora = recenter._c3_combine(eeg_logp, arms_k_logp["oracle_session_spd"], rel)
        l2_seed_scores["oracle_session_spd_oracle_id"].append(
            metrics.person_scores(y_te, p_te, c3_ora[idx, p_te], K)
        )

        for short_name, full_name in (
            ("static_cal_ref", "complex6b_static_cal_ref"),
            ("ema_tangent_shuffled", "complex6b_ema_tangent_shuffled"),
            ("session_fp_spd", "complex6b_session_fp_spd"),
        ):
            c3_c6b = recenter._c3_combine(eeg_logp, arms_c6b_k_logp[short_name], rel)
            lp_c6b = metrics.route(fp_c6b_logp, c3_c6b)
            l2_seed_scores[full_name].append(metrics.person_scores(y_te, p_te, lp_c6b, K))

    level2 = {}
    for arm in l2_names:
        mean_per_person = np.mean(l2_seed_scores[arm], axis=0).tolist()
        level2[arm] = {"per_person": mean_per_person, "mean": float(np.mean(mean_per_person))}

    return {
        "people": [str(p) for p in d["people"]],
        "n_classes": int(n_classes),
        "n_cal_windows": int(len(X_cal)),
        "n_test_windows": int(len(X_te)),
        "max_diff_vs_bank_riemann": max_diff_bank,
        "level1": level1,
        "level2": level2,
    }


def evaluate_all() -> dict:
    per_ds = {}
    for name in (*DEV_DATASETS, "dreyer_sim2"):
        print(f"Evaluating Protocol 01 (test-time reference adaptation) on {name}...", flush=True)
        per_ds[name] = _evaluate_dataset(name)

    def _collect_l1(arm: str) -> dict[str, list[float]]:
        return {ds: per_ds[ds]["level1"][arm]["per_person"] for ds in DEV_DATASETS}

    def _collect_l2(arm: str) -> dict[str, list[float]]:
        return {ds: per_ds[ds]["level2"][arm]["per_person"] for ds in DEV_DATASETS}

    l1_pooled_17 = {
        arm: float(np.mean([x for ds in DEV_DATASETS for x in per_ds[ds]["level1"][arm]["per_person"]]))
        for arm in ALL_ARMS
    }
    l2_names = ("row1_pooled_eegnet", "row2_current", "row3_oracle", *ALL_ARMS)
    l2_pooled_17 = {
        arm: float(np.mean([x for ds in DEV_DATASETS for x in per_ds[ds]["level2"][arm]["per_person"]]))
        for arm in l2_names
    }

    l1_comparisons = {}
    for arm in ALL_ARMS:
        if arm != "static_cal_ref":
            l1_comparisons[f"{arm}_vs_static_cal_ref"] = evaluate_recenter.paired_summary(
                _collect_l1(arm), _collect_l1("static_cal_ref")
            )

    # FM-10 order-dependence checks (ordered minus shuffled)
    fm10_checks = {
        "L1_ema_tangent_ordered_minus_shuffled": evaluate_recenter.paired_summary(
            _collect_l1("ema_tangent_ordered"), _collect_l1("ema_tangent_shuffled")
        ),
        "L1_ema_spd_ordered_minus_shuffled": evaluate_recenter.paired_summary(
            _collect_l1("ema_spd_ordered"), _collect_l1("ema_spd_shuffled")
        ),
        "L1_batch64_ordered_minus_shuffled": evaluate_recenter.paired_summary(
            _collect_l1("batch64_ordered"), _collect_l1("batch64_shuffled")
        ),
        "L2_ema_tangent_ordered_minus_shuffled": evaluate_recenter.paired_summary(
            _collect_l2("ema_tangent_ordered"), _collect_l2("ema_tangent_shuffled")
        ),
        "L2_ema_spd_ordered_minus_shuffled": evaluate_recenter.paired_summary(
            _collect_l2("ema_spd_ordered"), _collect_l2("ema_spd_shuffled")
        ),
        "L2_batch64_ordered_minus_shuffled": evaluate_recenter.paired_summary(
            _collect_l2("batch64_ordered"), _collect_l2("batch64_shuffled")
        ),
    }

    l2_comparisons = {}
    for arm in ALL_ARMS:
        if arm == "static_cal_ref":
            continue
        comp_row2 = evaluate_recenter.paired_summary(_collect_l2(arm), _collect_l2("row2_current"))
        dreyer_delta = float(
            per_ds["dreyer_sim2"]["level2"][arm]["mean"] - per_ds["dreyer_sim2"]["level2"]["row2_current"]["mean"]
        )
        ok, reason = metrics.adopt(comp_row2, dreyer_delta)
        comp_row2["dreyer_sim2_delta_vs_row2"] = dreyer_delta
        comp_row2["passes_adopt_rule"] = bool(ok)
        comp_row2["adopt_reason"] = reason
        l2_comparisons[f"{arm}_vs_row2"] = comp_row2

    return {
        "datasets": per_ds,
        "level1_pooled_17_mean": l1_pooled_17,
        "level2_pooled_17_mean": l2_pooled_17,
        "level1_comparisons_17": l1_comparisons,
        "fm10_order_checks_17": fm10_checks,
        "level2_comparisons_17": l2_comparisons,
    }


def format_markdown_report(res: dict) -> str:
    ds = res["datasets"]
    l1_17 = res["level1_pooled_17_mean"]
    l2_17 = res["level2_pooled_17_mean"]
    c1 = res["level1_comparisons_17"]
    fm10 = res["fm10_order_checks_17"]
    c2 = res["level2_comparisons_17"]

    lines = [
        "# Protocol 01 Results: Test-Time Reference Covariance Adaptation for Per-Person Task Experts",
        "",
        "Generated automatically by `research/recenter-pooled/src/evaluate_online_recenter.py`.",
        "Evaluation uses the 17 cross-day dev people (`bnci2014` 6, `bnci2015` 8, `zhou2016` 3) and",
        "14 fresh `dreyer_sim2` people. The 8 locked holdout people were never loaded.",
        "",
        "## 1. Level 1 — Standalone Per-Person `RiemannExperts` Task Decoding (Test Balanced Accuracy)",
        "",
        "| Arm | Test-Time $\\bar{C}_k$ Adaptation | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | $\\Delta$ vs `static_cal_ref` (95% CI) | B/W/T | `dreyer_sim2` (14) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    arm_desc = {
        "static_cal_ref": "Fixed calibration $\\bar{C}_k^{\\text{cal}}$ (`row 2` Riemann stream)",
        "ema_tangent_ordered": "Causal tangent EMA ($\\alpha=0.05$, dataset order)",
        "ema_tangent_shuffled": "Causal tangent EMA ($\\alpha=0.05$, **FM-10 shuffled**)",
        "ema_spd_ordered": "Causal SPD Log-Euclidean EMA ($\\alpha=0.05$, dataset order)",
        "ema_spd_shuffled": "Causal SPD Log-Euclidean EMA ($\\alpha=0.05$, **FM-10 shuffled**)",
        "batch64_ordered": "Stateless batch-64 tangent shift ($N_0=10$, dataset order)",
        "batch64_shuffled": "Stateless batch-64 tangent shift ($N_0=10$, **FM-10 shuffled**)",
        "session_fp_tangent": "Full-session `fp_logp`-weighted tangent shift ($N_0=10$)",
        "session_fp_spd": "Full-session `fp_logp`-weighted SPD Fréchet mean ($N_0=10$)",
        "oracle_session_spd": "Oracle test-session $\\bar{C}_k^{\\text{test}}$ (routed by `fp_logp`)",
        "oracle_session_spd_oracle_id": "Oracle test-session $\\bar{C}_k^{\\text{test}}$ + **oracle ID**",
        "complex6b_static_cal_ref": "Fixed $\\bar{C}_k^{\\text{cal}}$ routed by `complex_6b`",
        "complex6b_ema_tangent_shuffled": "Causal tangent EMA (**FM-10 shuffled**) w/ `complex_6b`",
        "complex6b_session_fp_spd": "Full-session SPD Fréchet mean w/ `complex_6b`",
    }

    for arm in ALL_ARMS:
        b14 = ds["bnci2014"]["level1"][arm]["mean"]
        b15 = ds["bnci2015"]["level1"][arm]["mean"]
        zh = ds["zhou2016"]["level1"][arm]["mean"]
        m17 = l1_17[arm]
        dr = ds["dreyer_sim2"]["level1"][arm]["mean"]
        if f"{arm}_vs_static_cal_ref" in c1:
            cmp = c1[f"{arm}_vs_static_cal_ref"]
            d_str = f"{cmp['mean_diff']:+.4f} [{cmp['ci95'][0]:+.4f}, {cmp['ci95'][1]:+.4f}]"
            bwt = f"{cmp['better']}/{cmp['worse']}/{cmp['tied']}"
        else:
            d_str = "—"
            bwt = "—"
        lines.append(
            f"| `{arm}` | {arm_desc[arm]} | {b14:.4f} | {b15:.4f} | {zh:.4f} | **{m17:.4f}** | {d_str} | {bwt} | {dr:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Level 2 — End-to-End Track 2 Integration (`EEGNet` + Adapted `RiemannExperts` via `C3`, Seeds 0–2)",
            "",
            "| Arm | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | $\\Delta$ vs `row 2` (95% CI) | B/W/T vs `row 2` | `dreyer_sim2` (14) | Adopt? |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    l2_names = ("row1_pooled_eegnet", "row2_current", "row3_oracle", *ALL_ARMS)
    for arm in l2_names:
        if arm == "static_cal_ref":
            continue
        b14 = ds["bnci2014"]["level2"][arm]["mean"]
        b15 = ds["bnci2015"]["level2"][arm]["mean"]
        zh = ds["zhou2016"]["level2"][arm]["mean"]
        m17 = l2_17[arm]
        dr = ds["dreyer_sim2"]["level2"][arm]["mean"]
        if f"{arm}_vs_row2" in c2:
            cr2 = c2[f"{arm}_vs_row2"]
            r2_str = f"{cr2['mean_diff']:+.4f} [{cr2['ci95'][0]:+.4f}, {cr2['ci95'][1]:+.4f}]"
            bwt = f"{cr2['better']}/{cr2['worse']}/{cr2['tied']}"
            adopt_str = "**YES**" if cr2["passes_adopt_rule"] else "no"
        else:
            r2_str = "—"
            bwt = "—"
            adopt_str = "ref"
        lines.append(
            f"| `{arm}` | {b14:.4f} | {b15:.4f} | {zh:.4f} | **{m17:.4f}** | {r2_str} | {bwt} | {dr:.4f} | {adopt_str} |"
        )

    lines.extend(
        [
            "",
            "## 3. Mandatory `FM-10` Window-Order Controls (Dataset Order vs Random Interleaving Across Subjects)",
            "",
            "| Order Comparison (17 Cross-Day Dev People) | Paired $\\Delta$ BA (Ordered − Shuffled) | 95% Bootstrap CI | Better / Worse / Tied |",
            "|---|---:|---:|---:|",
        ]
    )
    for k, label in [
        ("L1_ema_tangent_ordered_minus_shuffled", "Level 1: `ema_tangent_ordered − ema_tangent_shuffled`"),
        ("L1_ema_spd_ordered_minus_shuffled", "Level 1: `ema_spd_ordered − ema_spd_shuffled`"),
        ("L1_batch64_ordered_minus_shuffled", "Level 1: `batch64_ordered − batch64_shuffled`"),
        ("L2_ema_tangent_ordered_minus_shuffled", "Level 2: `ema_tangent_ordered − ema_tangent_shuffled`"),
        ("L2_ema_spd_ordered_minus_shuffled", "Level 2: `ema_spd_ordered − ema_spd_shuffled`"),
        ("L2_batch64_ordered_minus_shuffled", "Level 2: `batch64_ordered − batch64_shuffled`"),
    ]:
        cmp = fm10[k]
        lines.append(
            f"| {label} | {cmp['mean_diff']:+.4f} | [{cmp['ci95'][0]:+.4f}, {cmp['ci95'][1]:+.4f}] | "
            f"{cmp['better']} / {cmp['worse']} / {cmp['tied']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Per-Person Test Balanced Accuracy on 17 Dev People (Level 1 Standalone & Level 2 C3 Mixture)",
            "",
            "| Dataset | Person | L1 `static_cal_ref` | L1 `ema_tangent_shuffled` | L1 `session_fp_spd` | L1 `oracle_session_spd` | L2 `row2_current` | L2 `ema_tangent_shuffled` | L2 `session_fp_spd` | L2 `oracle_session_spd` | L2 `complex6b_session_fp_spd` |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in DEV_DATASETS:
        d_info = ds[name]
        for k, p_id in enumerate(d_info["people"]):
            l1_st = d_info["level1"]["static_cal_ref"]["per_person"][k]
            l1_em = d_info["level1"]["ema_tangent_shuffled"]["per_person"][k]
            l1_sp = d_info["level1"]["session_fp_spd"]["per_person"][k]
            l1_or = d_info["level1"]["oracle_session_spd"]["per_person"][k]
            l2_r2 = d_info["level2"]["row2_current"]["per_person"][k]
            l2_em = d_info["level2"]["ema_tangent_shuffled"]["per_person"][k]
            l2_sp = d_info["level2"]["session_fp_spd"]["per_person"][k]
            l2_or = d_info["level2"]["oracle_session_spd"]["per_person"][k]
            l2_c6 = d_info["level2"]["complex6b_session_fp_spd"]["per_person"][k]
            lines.append(
                f"| `{name}` | `{p_id}` | {l1_st:.4f} | {l1_em:.4f} | {l1_sp:.4f} | {l1_or:.4f} | "
                f"{l2_r2:.4f} | {l2_em:.4f} | {l2_sp:.4f} | {l2_or:.4f} | {l2_c6:.4f} |"
            )

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res = evaluate_all()
    json_path = RESULTS_DIR / "online_recenter_results.json"
    md_path = RESULTS_DIR / "online_recenter_results.md"
    json_path.write_text(json.dumps(res, indent=2))
    md_path.write_text(format_markdown_report(res))
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
