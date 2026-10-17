"""Pre-registered evaluation of Protocol 02 (Handoff §2c): Per-person mu-peak band centering for
`RiemannExperts`.

Evaluates all 9 pre-registered arms from `experiments/02-protocol/protocol.md` on the 17 cross-day
dev people (`bnci2014`, `bnci2015`, `zhou2016`) and 14 fresh `dreyer_sim2` people, at both Level 1
(standalone `RiemannExperts`) and Level 2 (end-to-end Track 2 `C3` mixture across seeds 0–2).

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
import mu_band  # noqa: E402
import recenter  # noqa: E402

DEV_DATASETS = ("bnci2014", "bnci2015", "zhou2016")
SEEDS = (0, 1, 2)
BANK_DIR = REPO / "outputs" / "t2-int-harness"
RESULTS_DIR = EXP_ROOT / "results"


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

    res_mu = mu_band.evaluate_mu_arms_on_split(X_cal, y_cal, p_cal, X_te, fp_logp, K, n_classes, sfreq)
    arms_k_logp = res_mu["arms_logp"]

    level1 = {}
    for arm in mu_band.PROTOCOL02_ARMS:
        lp_routed = metrics.route(fp_logp, arms_k_logp[arm])
        scores = metrics.person_scores(y_te, p_te, lp_routed, K).tolist()
        level1[arm] = {"per_person": scores, "mean": float(np.mean(scores))}

    l2_names = ("row2_current", *mu_band.PROTOCOL02_ARMS)
    l2_seed_scores = {arm: [] for arm in l2_names}

    for s in seeds:
        bank_s = _load_bank(name, s)
        ref_rows = metrics.rows(bank_s)
        eeg_logp = bank_s["eeg_logp"]
        rel = bank_s["rel"]

        l2_seed_scores["row2_current"].append(metrics.person_scores(y_te, p_te, ref_rows["current"], K))
        for arm in mu_band.PROTOCOL02_ARMS:
            c3_k = recenter._c3_combine(eeg_logp, arms_k_logp[arm], rel)
            lp_routed = metrics.route(fp_logp, c3_k)
            l2_seed_scores[arm].append(metrics.person_scores(y_te, p_te, lp_routed, K))

    level2 = {}
    for arm in l2_names:
        mean_per_person = np.mean(l2_seed_scores[arm], axis=0).tolist()
        level2[arm] = {"per_person": mean_per_person, "mean": float(np.mean(mean_per_person))}

    return {
        "people": [str(p) for p in d["people"]],
        "mu_peaks_hz": [round(float(f), 3) for f in res_mu["mu_peaks"]],
        "bands_wide_hz": [[round(float(a), 2), round(float(b), 2)] for a, b in res_mu["bands_wide"]],
        "bands_lower_hz": [[round(float(a), 2), round(float(b), 2)] for a, b in res_mu["bands_lower"]],
        "level1": level1,
        "level2": level2,
    }


def evaluate_all() -> dict:
    per_ds = {}
    for name in (*DEV_DATASETS, "dreyer_sim2"):
        print(f"Evaluating Protocol 02 (§2c mu-band centering) on {name}...", flush=True)
        per_ds[name] = _evaluate_dataset(name)

    def _collect_l1(arm: str) -> dict[str, list[float]]:
        return {ds: per_ds[ds]["level1"][arm]["per_person"] for ds in DEV_DATASETS}

    def _collect_l2(arm: str) -> dict[str, list[float]]:
        return {ds: per_ds[ds]["level2"][arm]["per_person"] for ds in DEV_DATASETS}

    l1_pooled_17 = {
        arm: float(np.mean([x for ds in DEV_DATASETS for x in per_ds[ds]["level1"][arm]["per_person"]]))
        for arm in mu_band.PROTOCOL02_ARMS
    }
    l2_pooled_17 = {
        arm: float(np.mean([x for ds in DEV_DATASETS for x in per_ds[ds]["level2"][arm]["per_person"]]))
        for arm in ("row2_current", *mu_band.PROTOCOL02_ARMS)
    }

    l1_comparisons = {
        "mu_shift_wide_static_vs_fixed_static": evaluate_recenter.paired_summary(
            _collect_l1("mu_shift_wide_static"), _collect_l1("fixed_8_30_static")
        ),
        "mu_lower_edge_static_vs_fixed_static": evaluate_recenter.paired_summary(
            _collect_l1("mu_lower_edge_static"), _collect_l1("fixed_8_30_static")
        ),
        "mu_beta_2band_static_vs_fixed_static": evaluate_recenter.paired_summary(
            _collect_l1("mu_beta_2band_static"), _collect_l1("fixed_8_30_static")
        ),
        "mu_shift_wide_ema_shuf_vs_fixed_ema_shuf": evaluate_recenter.paired_summary(
            _collect_l1("mu_shift_wide_ema_shuf"), _collect_l1("fixed_8_30_ema_shuf")
        ),
        "mu_lower_edge_ema_shuf_vs_fixed_ema_shuf": evaluate_recenter.paired_summary(
            _collect_l1("mu_lower_edge_ema_shuf"), _collect_l1("fixed_8_30_ema_shuf")
        ),
        "mu_beta_2band_ema_shuf_vs_fixed_ema_shuf": evaluate_recenter.paired_summary(
            _collect_l1("mu_beta_2band_ema_shuf"), _collect_l1("fixed_8_30_ema_shuf")
        ),
    }

    l2_comparisons = {}
    for arm, ref_arm in (
        ("mu_shift_wide_static", "fixed_8_30_static"),
        ("mu_lower_edge_static", "fixed_8_30_static"),
        ("mu_beta_2band_static", "fixed_8_30_static"),
        ("mu_shift_wide_ema_shuf", "fixed_8_30_ema_shuf"),
        ("mu_lower_edge_ema_shuf", "fixed_8_30_ema_shuf"),
        ("mu_beta_2band_ema_shuf", "fixed_8_30_ema_shuf"),
        ("mu_shift_wide_sess_spd", "row2_current"),
    ):
        comp_ref = evaluate_recenter.paired_summary(_collect_l2(arm), _collect_l2(ref_arm))
        comp_row2 = evaluate_recenter.paired_summary(_collect_l2(arm), _collect_l2("row2_current"))
        dreyer_delta = float(
            per_ds["dreyer_sim2"]["level2"][arm]["mean"] - per_ds["dreyer_sim2"]["level2"][ref_arm]["mean"]
        )
        ok, reason = metrics.adopt(comp_ref, dreyer_delta)
        comp_ref["dreyer_sim2_delta_vs_ref"] = dreyer_delta
        comp_ref["passes_adopt_over_fixed_band"] = bool(ok)
        comp_ref["adopt_reason"] = reason
        l2_comparisons[f"{arm}_vs_{ref_arm}"] = comp_ref
        l2_comparisons[f"{arm}_vs_row2"] = comp_row2

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
        "# Protocol 02 Results: Per-Person $\\mu$-Peak Band Centering for `RiemannExperts` (Handoff §2c)",
        "",
        "Generated automatically by `research/recenter-pooled/src/evaluate_mu_band.py`.",
        "Evaluation uses the 17 cross-day dev people (`bnci2014` 6, `bnci2015` 8, `zhou2016` 3) and",
        "14 fresh `dreyer_sim2` people. The 8 locked holdout people were never loaded.",
        "",
        "## 1. Estimated Calibration $\\mu$-Peak Frequencies $f_{\\mu,k}$ (Hz)",
        "",
        "| Dataset | People & Calibration $f_{\\mu,k}$ (Hz) | Shifted Passbands $[\\text{lo}_k, \\text{hi}_k]$ (Hz) |",
        "|---|---|---|",
    ]
    for name in (*DEV_DATASETS, "dreyer_sim2"):
        d_info = ds[name]
        p_str = ", ".join(f"`{p}`: {f:.2f}" for p, f in zip(d_info["people"], d_info["mu_peaks_hz"]))
        b_str = ", ".join(f"`{p}`: [{lo:.1f}, {hi:.1f}]" for p, (lo, hi) in zip(d_info["people"], d_info["bands_wide_hz"]))
        lines.append(f"| `{name}` | {p_str} | {b_str} |")

    lines.extend(
        [
            "",
            "## 2. Level 1 — Standalone Per-Person `RiemannExperts` Task Decoding (Test Balanced Accuracy)",
            "",
            "| Arm | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | `dreyer_sim2` (14) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in mu_band.PROTOCOL02_ARMS:
        b14 = ds["bnci2014"]["level1"][arm]["mean"]
        b15 = ds["bnci2015"]["level1"][arm]["mean"]
        zh = ds["zhou2016"]["level1"][arm]["mean"]
        m17 = l1_17[arm]
        dr = ds["dreyer_sim2"]["level1"][arm]["mean"]
        lines.append(f"| `{arm}` | {b14:.4f} | {b15:.4f} | {zh:.4f} | **{m17:.4f}** | {dr:.4f} |")

    lines.extend(
        [
            "",
            "### Level 1 Paired Comparisons vs Fixed `8–30 Hz` Counterpart (17 Cross-Day Dev People)",
            "",
            "| Comparison | Paired $\\Delta$ BA | 95% Bootstrap CI | Better / Worse / Tied |",
            "|---|---:|---:|---:|",
        ]
    )
    for k, label in [
        ("mu_shift_wide_static_vs_fixed_static", "`mu_shift_wide_static − fixed_8_30_static`"),
        ("mu_lower_edge_static_vs_fixed_static", "`mu_lower_edge_static − fixed_8_30_static`"),
        ("mu_beta_2band_static_vs_fixed_static", "`mu_beta_2band_static − fixed_8_30_static`"),
        ("mu_shift_wide_ema_shuf_vs_fixed_ema_shuf", "`mu_shift_wide_ema_shuf − fixed_8_30_ema_shuf`"),
        ("mu_lower_edge_ema_shuf_vs_fixed_ema_shuf", "`mu_lower_edge_ema_shuf − fixed_8_30_ema_shuf`"),
        ("mu_beta_2band_ema_shuf_vs_fixed_ema_shuf", "`mu_beta_2band_ema_shuf − fixed_8_30_ema_shuf`"),
    ]:
        cmp = c1[k]
        lines.append(
            f"| {label} | {cmp['mean_diff']:+.4f} | [{cmp['ci95'][0]:+.4f}, {cmp['ci95'][1]:+.4f}] | "
            f"{cmp['better']} / {cmp['worse']} / {cmp['tied']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Level 2 — End-to-End Track 2 Integration (`C3` Mixture, Seeds 0–2 Mean)",
            "",
            "| Arm | `bnci2014` (6) | `bnci2015` (8) | `zhou2016` (3) | **17-Person Mean** | $\\Delta$ vs Fixed-Band Ref (95% CI) | $\\Delta$ vs `row 2` (95% CI) | `dreyer_sim2` (14) | Adopt $\\mu$-Band? |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    ref_map = {
        "mu_shift_wide_static": "fixed_8_30_static",
        "mu_lower_edge_static": "fixed_8_30_static",
        "mu_beta_2band_static": "fixed_8_30_static",
        "mu_shift_wide_ema_shuf": "fixed_8_30_ema_shuf",
        "mu_lower_edge_ema_shuf": "fixed_8_30_ema_shuf",
        "mu_beta_2band_ema_shuf": "fixed_8_30_ema_shuf",
        "mu_shift_wide_sess_spd": "row2_current",
    }
    for arm in mu_band.PROTOCOL02_ARMS:
        b14 = ds["bnci2014"]["level2"][arm]["mean"]
        b15 = ds["bnci2015"]["level2"][arm]["mean"]
        zh = ds["zhou2016"]["level2"][arm]["mean"]
        m17 = l2_17[arm]
        dr = ds["dreyer_sim2"]["level2"][arm]["mean"]
        if arm in ref_map:
            ref_a = ref_map[arm]
            cref = c2[f"{arm}_vs_{ref_a}"]
            cr2 = c2[f"{arm}_vs_row2"]
            ref_str = f"{cref['mean_diff']:+.4f} [{cref['ci95'][0]:+.4f}, {cref['ci95'][1]:+.4f}]"
            r2_str = f"{cr2['mean_diff']:+.4f} [{cr2['ci95'][0]:+.4f}, {cr2['ci95'][1]:+.4f}]"
            adopt_str = "**YES**" if cref["passes_adopt_over_fixed_band"] else "no"
        else:
            ref_str = "ref"
            r2_str = "—"
            adopt_str = "ref"
        lines.append(
            f"| `{arm}` | {b14:.4f} | {b15:.4f} | {zh:.4f} | **{m17:.4f}** | {ref_str} | {r2_str} | {dr:.4f} | {adopt_str} |"
        )

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res = evaluate_all()
    json_path = RESULTS_DIR / "mu_band_results.json"
    md_path = RESULTS_DIR / "mu_band_results.md"
    json_path.write_text(json.dumps(res, indent=2))
    md_path.write_text(format_markdown_report(res))
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
