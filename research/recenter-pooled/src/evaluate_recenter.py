"""Pre-registered evaluation of Handoff §2a: Fingerprint-routed covariance re-centering.

Evaluates:
- Level 1: Standalone tangent-space task decoders (`ts_global_pooled`, `ts_recenter_routed`,
  `ts_recenter_whitener`, `ts_recenter_oracle`, `ts_recenter_fb_routed`, `ts_recenter_fb_oracle`,
  `ts_per_person_routed`, `ts_per_person_oracle`) on the 17 dev cross-day people (`bnci2014`,
  `bnci2015`, `zhou2016`) and 14 fresh `dreyer_sim2` people.
- Level 2: End-to-end Track 2 integration arms (`E1`, `E2`, `E3` and their oracle ceilings) against
  `row 2` (`current`) and `row 3` (`oracle`) across seeds 0–2.

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
import torch
from scipy.stats import beta as sp_beta

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
import metrics  # noqa: E402
import recenter  # noqa: E402

DEV_DATASETS = ("bnci2014", "bnci2015", "zhou2016")
SEEDS = (0, 1, 2)
BANK_DIR = REPO / "outputs" / "t2-int-harness"
RESULTS_DIR = EXP_ROOT / "results"

LEVEL1_ARMS = (
    "ts_global_pooled",
    "ts_recenter_routed",
    "ts_recenter_whitener",
    "ts_recenter_oracle",
    "ts_recenter_fb_routed",
    "ts_recenter_fb_oracle",
    "ts_per_person_routed",
    "ts_per_person_oracle",
)

LEVEL2_ARMS = (
    "row1_pooled_eegnet",
    "row2_current",
    "row3_oracle",
    "E1_fully_pooled_recenter",
    "E1_oracle",
    "E2_eeg_experts_plus_recenter",
    "E2_oracle",
    "E3_eeg_experts_plus_shrunk_riemann",
    "E3_oracle",
)


def clopper_pearson_ci(wins: int, total: int, alpha: float = 0.05) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    lo = 0.0 if wins == 0 else float(sp_beta.ppf(alpha / 2, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(sp_beta.ppf(1 - alpha / 2, wins + 1, total - wins))
    return [lo, hi]


def paired_summary(scores_a: dict[str, list[float]], scores_b: dict[str, list[float]]) -> dict:
    per_ds = {ds: (scores_a[ds], scores_b[ds]) for ds in scores_a}
    comp = metrics.pooled_comparison(per_ds)
    non_tied = comp["better"] + comp["worse"]
    comp["tied"] = int(comp["n"] - non_tied)
    comp["non_tied"] = int(non_tied)
    comp["mean_diff"] = float(comp["mean"])
    comp["ci95"] = [float(comp["ci"][0]), float(comp["ci"][1])]
    comp["win_ci95"] = clopper_pearson_ci(comp["better"], non_tied)
    return comp


def _load_bank(name: str, seed: int) -> dict:
    z = np.load(BANK_DIR / f"bank_{name}_seed{seed}.npz")
    return {k: z[k] for k in z.files}


def _evaluate_dataset(name: str) -> dict:
    if name == "dreyer_sim2":
        d = data.load_dreyer_sim2()
        raw_subj = np.load(REPO / "data" / "experiments" / "dreyer_windows.npz", allow_pickle=True)["subject"]
        tr_mask = d["pool"] | d["calib"]
        subj_train = raw_subj[tr_mask]
        eval_person_train = d["person"][tr_mask]
        seeds = (0,)
    else:
        d = data.load(name, part="dev")
        tr_mask = d["calib"]
        subj_train = d["person"][tr_mask]
        eval_person_train = d["person"][tr_mask]
        seeds = SEEDS

    X_tr = d["X"][tr_mask]
    y_tr = d["y"][tr_mask]
    unit_tr = d["unit"][tr_mask]
    X_te = d["X"][d["test"]]
    K_eval = len(d["people"])
    n_classes = d["n_classes"]
    sfreq = d["sfreq"]

    bank0 = _load_bank(name, 0)
    y_te = bank0["y"]
    p_te = bank0["person"]
    fp_logp = bank0["fp_logp"]
    idx = np.arange(len(y_te))

    # 1. Global uncentered pooled tangent-space LR
    pred_global = recenter.fit_global_pooled_8_30(X_tr, y_tr, n_classes, sfreq)
    lp_global = pred_global(X_te)

    # 2. Per-person reference covariance re-centering + 1 pooled LR (8–30 Hz)
    rec830 = recenter.fit_recenter_8_30(
        X_tr, y_tr, subj_train, eval_person_train, K_eval, n_classes, sfreq, unit_train=unit_tr
    )
    rec_k_logp = rec830["per_person_logp"](X_te)                 # (n_te, K, C)
    lp_rec_routed = metrics.route(fp_logp, rec_k_logp)           # (n_te, C)
    lp_rec_whitener = rec830["whitener_interp_logp"](X_te, fp_logp)
    lp_rec_oracle = rec_k_logp[idx, p_te]

    # 3. 5-band (4–45 Hz) per-person reference covariance re-centering + StandardScaler + 1 pooled LR
    rec_fb = recenter.fit_recenter_fb(X_tr, y_tr, subj_train, eval_person_train, K_eval, n_classes, sfreq)
    rec_fb_k_logp = rec_fb["per_person_logp"](X_te)
    lp_fb_routed = metrics.route(fp_logp, rec_fb_k_logp)
    lp_fb_oracle = rec_fb_k_logp[idx, p_te]

    # 4. Shipped per-person RiemannExperts (FM-2 per-person retraining comparator)
    rie_k_logp = bank0["riemann_logp"]
    lp_pp_routed = metrics.route(fp_logp, rie_k_logp)
    lp_pp_oracle = rie_k_logp[idx, p_te]

    l1_logps = {
        "ts_global_pooled": lp_global,
        "ts_recenter_routed": lp_rec_routed,
        "ts_recenter_whitener": lp_rec_whitener,
        "ts_recenter_oracle": lp_rec_oracle,
        "ts_recenter_fb_routed": lp_fb_routed,
        "ts_recenter_fb_oracle": lp_fb_oracle,
        "ts_per_person_routed": lp_pp_routed,
        "ts_per_person_oracle": lp_pp_oracle,
    }
    level1 = {}
    for arm, lp in l1_logps.items():
        scores = metrics.person_scores(y_te, p_te, lp, K_eval).tolist()
        level1[arm] = {
            "per_person": scores,
            "mean": float(np.mean(scores)),
        }

    # On dreyer_sim2, also record a calibration-only control (14 evaluation subjects' R1-R3 only)
    calib_only_control = None
    if name == "dreyer_sim2":
        cal_m = d["calib"]
        rec_cal_only = recenter.fit_recenter_8_30(
            d["X"][cal_m],
            d["y"][cal_m],
            d["person"][cal_m],
            d["person"][cal_m],
            K_eval,
            n_classes,
            sfreq,
        )
        cal_k_lp = rec_cal_only["per_person_logp"](X_te)
        s_routed = metrics.person_scores(y_te, p_te, metrics.route(fp_logp, cal_k_lp), K_eval).tolist()
        s_oracle = metrics.person_scores(y_te, p_te, cal_k_lp[idx, p_te], K_eval).tolist()
        calib_only_control = {
            "ts_recenter_routed_calib14_only": {"per_person": s_routed, "mean": float(np.mean(s_routed))},
            "ts_recenter_oracle_calib14_only": {"per_person": s_oracle, "mean": float(np.mean(s_oracle))},
        }

    # Level 2: End-to-end Track 2 mixture arms across seeds
    l2_seed_scores = {arm: [] for arm in LEVEL2_ARMS}
    for s in seeds:
        bank_s = _load_bank(name, s)
        ref_rows = metrics.rows(bank_s)
        l2_arms = recenter.level2_arms(bank_s, rec_k_logp)
        arm_logps = {
            "row1_pooled_eegnet": ref_rows["pooled"],
            "row2_current": ref_rows["current"],
            "row3_oracle": ref_rows["oracle"],
            **l2_arms,
        }
        for arm in LEVEL2_ARMS:
            l2_seed_scores[arm].append(metrics.person_scores(y_te, p_te, arm_logps[arm], K_eval))

    level2 = {}
    for arm in LEVEL2_ARMS:
        mean_per_person = np.mean(l2_seed_scores[arm], axis=0).tolist()
        level2[arm] = {
            "per_person": mean_per_person,
            "mean": float(np.mean(mean_per_person)),
        }

    return {
        "people": [str(p) for p in d["people"]],
        "n_classes": int(n_classes),
        "n_train_windows": int(len(X_tr)),
        "n_train_subjects": int(len(np.unique(subj_train))),
        "n_test_windows": int(len(X_te)),
        "level1": level1,
        "level2": level2,
        "calib_only_control": calib_only_control,
    }


def evaluate_all() -> dict:
    per_ds = {}
    for name in (*DEV_DATASETS, "dreyer_sim2"):
        print(f"Evaluating §2a on {name}...", flush=True)
        per_ds[name] = _evaluate_dataset(name)

    def _collect_l1(arm: str) -> dict[str, list[float]]:
        return {ds: per_ds[ds]["level1"][arm]["per_person"] for ds in DEV_DATASETS}

    def _collect_l2(arm: str) -> dict[str, list[float]]:
        return {ds: per_ds[ds]["level2"][arm]["per_person"] for ds in DEV_DATASETS}

    l1_pooled_17 = {}
    for arm in LEVEL1_ARMS:
        vals = [x for ds in DEV_DATASETS for x in per_ds[ds]["level1"][arm]["per_person"]]
        l1_pooled_17[arm] = float(np.mean(vals))

    l2_pooled_17 = {}
    for arm in LEVEL2_ARMS:
        vals = [x for ds in DEV_DATASETS for x in per_ds[ds]["level2"][arm]["per_person"]]
        l2_pooled_17[arm] = float(np.mean(vals))

    l1_comparisons = {
        "H2a1_recenter_routed_vs_global_pooled": paired_summary(
            _collect_l1("ts_recenter_routed"), _collect_l1("ts_global_pooled")
        ),
        "H2a1_recenter_oracle_vs_global_pooled": paired_summary(
            _collect_l1("ts_recenter_oracle"), _collect_l1("ts_global_pooled")
        ),
        "whitener_interp_vs_recenter_routed": paired_summary(
            _collect_l1("ts_recenter_whitener"), _collect_l1("ts_recenter_routed")
        ),
        "H2a2_recenter_routed_vs_per_person_routed": paired_summary(
            _collect_l1("ts_recenter_routed"), _collect_l1("ts_per_person_routed")
        ),
        "H2a2_recenter_oracle_vs_per_person_oracle": paired_summary(
            _collect_l1("ts_recenter_oracle"), _collect_l1("ts_per_person_oracle")
        ),
        "recenter_fb_routed_vs_recenter_routed": paired_summary(
            _collect_l1("ts_recenter_fb_routed"), _collect_l1("ts_recenter_routed")
        ),
        "recenter_fb_routed_vs_per_person_routed": paired_summary(
            _collect_l1("ts_recenter_fb_routed"), _collect_l1("ts_per_person_routed")
        ),
        "recenter_routing_headroom_oracle_minus_routed": paired_summary(
            _collect_l1("ts_recenter_oracle"), _collect_l1("ts_recenter_routed")
        ),
    }

    l2_comparisons = {}
    for arm in (
        "E1_fully_pooled_recenter",
        "E1_oracle",
        "E2_eeg_experts_plus_recenter",
        "E2_oracle",
        "E3_eeg_experts_plus_shrunk_riemann",
        "E3_oracle",
    ):
        comp_row2 = paired_summary(_collect_l2(arm), _collect_l2("row2_current"))
        comp_row1 = paired_summary(_collect_l2(arm), _collect_l2("row1_pooled_eegnet"))
        dreyer_delta = float(
            per_ds["dreyer_sim2"]["level2"][arm]["mean"] - per_ds["dreyer_sim2"]["level2"]["row2_current"]["mean"]
        )
        ok, reason = metrics.adopt(comp_row2, dreyer_delta)
        comp_row2["dreyer_sim2_delta_vs_row2"] = dreyer_delta
        comp_row2["passes_adopt_rule"] = bool(ok)
        comp_row2["adopt_reason"] = reason
        l2_comparisons[f"{arm}_vs_row2"] = comp_row2
        l2_comparisons[f"{arm}_vs_row1"] = comp_row1

    return {
        "datasets": per_ds,
        "level1_pooled_17_mean": l1_pooled_17,
        "level2_pooled_17_mean": l2_pooled_17,
        "level1_comparisons_17": l1_comparisons,
        "level2_comparisons_17": l2_comparisons,
    }


def format_markdown_report(res: dict) -> str:
    ds = res["datasets"]
    l1_17 = res["level1_pooled_17_mean"]
    l2_17 = res["level2_pooled_17_mean"]
    c1 = res["level1_comparisons_17"]
    c2 = res["level2_comparisons_17"]

    lines = [
        "# Handoff §2a Results: Fingerprint-Routed Covariance Re-Centering",
        "",
        "Generated automatically by `research/recenter-pooled/src/evaluate_recenter.py`.",
        "Evaluation uses the 17 cross-day dev people (`bnci2014` 6, `bnci2015` 8, `zhou2016` 3) and",
        "14 fresh `dreyer_sim2` people. The 8 locked holdout people were never loaded.",
        "",
        "## 1. Level 1 — Standalone Tangent-Space Task Decoding (Test Balanced Accuracy)",
        "",
        "| Arm | Reference Covariance | Task Classifier | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | `dreyer_sim2` (14) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    l1_meta = {
        "ts_global_pooled": ("1 global $\\bar{C}_{\\text{global}}$", "1 pooled LR (8–30 Hz)"),
        "ts_recenter_routed": ("Per-person $\\bar{C}_k$ (`fp_logp` mixture)", "1 pooled LR (8–30 Hz)"),
        "ts_recenter_whitener": ("Interpolated $\\bar{W}(x)=\\sum_k p_k \\bar{C}_k^{-1/2}$", "1 pooled LR (8–30 Hz)"),
        "ts_recenter_oracle": ("Oracle $\\bar{C}_{k_{\\text{true}}}$", "1 pooled LR (8–30 Hz)"),
        "ts_recenter_fb_routed": ("5-band per-person $\\bar{C}_{b,k}$ (`fp_logp`)", "1 pooled LR (4–45 Hz)"),
        "ts_recenter_fb_oracle": ("5-band oracle $\\bar{C}_{b,k_{\\text{true}}}$", "1 pooled LR (4–45 Hz)"),
        "ts_per_person_routed": ("Per-person $\\bar{C}_k$ (`fp_logp` mixture)", "$K$ per-person LRs (8–30 Hz)"),
        "ts_per_person_oracle": ("Oracle $\\bar{C}_{k_{\\text{true}}}$", "$K$ per-person LRs (8–30 Hz)"),
    }
    for arm in LEVEL1_ARMS:
        ref_desc, clf_desc = l1_meta[arm]
        b14 = ds["bnci2014"]["level1"][arm]["mean"]
        b15 = ds["bnci2015"]["level1"][arm]["mean"]
        zh = ds["zhou2016"]["level1"][arm]["mean"]
        m17 = l1_17[arm]
        dr = ds["dreyer_sim2"]["level1"][arm]["mean"]
        lines.append(
            f"| `{arm}` | {ref_desc} | {clf_desc} | {b14:.4f} | {b15:.4f} | {zh:.4f} | **{m17:.4f}** | {dr:.4f} |"
        )

    if ds["dreyer_sim2"]["calib_only_control"] is not None:
        ctrl = ds["dreyer_sim2"]["calib_only_control"]
        r_c = ctrl["ts_recenter_routed_calib14_only"]["mean"]
        o_c = ctrl["ts_recenter_oracle_calib14_only"]["mean"]
        lines.extend(
            [
                "",
                f"*Note on `dreyer_sim2` training pool:* In the main table above, `ts_recenter_*` fits its pooled LR on `pool | calib` (all 73 subjects re-centered by their own $\\bar{{C}}_s$, matching `pooled_eegnet` per FM-2). If fitted on only the 14 evaluation subjects' R1–R3 calibration windows (`d['calib']`), `ts_recenter_routed` scores **{r_c:.4f}** and `ts_recenter_oracle` scores **{o_c:.4f}**.",
            ]
        )

    lines.extend(
        [
            "",
            "### Level 1 Paired Comparisons on 17 Cross-Day Dev People (5,000 Bootstrap Resamples, Seed 0)",
            "",
            "| Comparison | Paired $\\Delta$ BA | 95% Bootstrap CI | Better / Worse / Tied | Non-Tied Win 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("H2a1_recenter_routed_vs_global_pooled", "`ts_recenter_routed − ts_global_pooled` (**H2a.1: Does re-centering help pooled TS?**)"),
        ("H2a1_recenter_oracle_vs_global_pooled", "`ts_recenter_oracle − ts_global_pooled` (**H2a.1 oracle ceiling**)"),
        ("whitener_interp_vs_recenter_routed", "`ts_recenter_whitener − ts_recenter_routed` (1-pass whitener vs tangent mixture)"),
        ("H2a2_recenter_routed_vs_per_person_routed", "`ts_recenter_routed − ts_per_person_routed` (**H2a.2: 1 pooled LR vs $K$ per-person LRs**)"),
        ("H2a2_recenter_oracle_vs_per_person_oracle", "`ts_recenter_oracle − ts_per_person_oracle` (**H2a.2 under oracle ID**)"),
        ("recenter_fb_routed_vs_recenter_routed", "`ts_recenter_fb_routed − ts_recenter_routed` (5-band vs single-band re-centered)"),
        ("recenter_fb_routed_vs_per_person_routed", "`ts_recenter_fb_routed − ts_per_person_routed` (5-band pooled vs per-person 8–30 Hz)"),
        ("recenter_routing_headroom_oracle_minus_routed", "`ts_recenter_oracle − ts_recenter_routed` (routing headroom of re-centered TS)"),
    ]:
        comp = c1[key]
        lo, hi = comp["ci95"]
        wlo, whi = comp["win_ci95"]
        lines.append(
            f"| {label} | {comp['mean_diff']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{comp['better']} / {comp['worse']} / {comp['tied']} | [{wlo:.3f}, {whi:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## 2. Level 2 — End-to-End Track 2 Integration Arms (Seeds 0–2 Mean)",
            "",
            "| Arm | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | $\\Delta$ vs `row 1` (95% CI) | $\\Delta$ vs `row 2` (95% CI) | B/W/T vs `row 2` | `dreyer_sim2` (14) | Adopt? |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for arm in LEVEL2_ARMS:
        b14 = ds["bnci2014"]["level2"][arm]["mean"]
        b15 = ds["bnci2015"]["level2"][arm]["mean"]
        zh = ds["zhou2016"]["level2"][arm]["mean"]
        m17 = l2_17[arm]
        dr = ds["dreyer_sim2"]["level2"][arm]["mean"]
        if f"{arm}_vs_row2" in c2:
            cr1 = c2[f"{arm}_vs_row1"]
            cr2 = c2[f"{arm}_vs_row2"]
            r1_str = f"{cr1['mean_diff']:+.4f} [{cr1['ci95'][0]:+.4f}, {cr1['ci95'][1]:+.4f}]"
            r2_str = f"{cr2['mean_diff']:+.4f} [{cr2['ci95'][0]:+.4f}, {cr2['ci95'][1]:+.4f}]"
            bwt = f"{cr2['better']}/{cr2['worse']}/{cr2['tied']}"
            adopt_str = "**YES**" if cr2["passes_adopt_rule"] else "no"
        else:
            r1_str = "—"
            r2_str = "—"
            bwt = "—"
            adopt_str = "ref"
        lines.append(
            f"| `{arm}` | {b14:.4f} | {b15:.4f} | {zh:.4f} | **{m17:.4f}** | "
            f"{r1_str} | {r2_str} | {bwt} | {dr:.4f} | {adopt_str} |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Person Test Balanced Accuracy (17 Dev People)",
            "",
            "| Dataset | Person | `ts_global_pooled` | `ts_recenter_routed` | `ts_recenter_oracle` | `ts_per_person_routed` | `row1_pooled_eegnet` | `row2_current` | `E1_fully_pooled_recenter` | `E2_eeg_experts_plus_recenter` | `E3_eeg_experts_plus_shrunk_riemann` |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in DEV_DATASETS:
        d_info = ds[name]
        for k, p_id in enumerate(d_info["people"]):
            g_pl = d_info["level1"]["ts_global_pooled"]["per_person"][k]
            r_rt = d_info["level1"]["ts_recenter_routed"]["per_person"][k]
            r_or = d_info["level1"]["ts_recenter_oracle"]["per_person"][k]
            pp_rt = d_info["level1"]["ts_per_person_routed"]["per_person"][k]
            r1 = d_info["level2"]["row1_pooled_eegnet"]["per_person"][k]
            r2 = d_info["level2"]["row2_current"]["per_person"][k]
            e1 = d_info["level2"]["E1_fully_pooled_recenter"]["per_person"][k]
            e2 = d_info["level2"]["E2_eeg_experts_plus_recenter"]["per_person"][k]
            e3 = d_info["level2"]["E3_eeg_experts_plus_shrunk_riemann"]["per_person"][k]
            lines.append(
                f"| `{name}` | `{p_id}` | {g_pl:.4f} | {r_rt:.4f} | {r_or:.4f} | {pp_rt:.4f} | "
                f"{r1:.4f} | {r2:.4f} | {e1:.4f} | {e2:.4f} | {e3:.4f} |"
            )

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res = evaluate_all()
    json_path = RESULTS_DIR / "recenter_results.json"
    md_path = RESULTS_DIR / "recenter_results.md"
    json_path.write_text(json.dumps(res, indent=2))
    md_path.write_text(format_markdown_report(res))
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
