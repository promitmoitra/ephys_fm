"""Run the pre-registered Protocol 00: 2 x 2 Complex Hermitian x 5/6-band fingerprint
on the Track 2 cross-day harness (17 dev people + dreyer_sim2 + zhou2016 single-day slices)."""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [
    str(REPO / "external" / "2026-competition"),
    str(REPO / "track2"),
    str(REPO / "research" / "integration-harness" / "src"),
    str(HERE),
]

import complex_fp  # noqa: E402
import data  # noqa: E402
import metrics  # noqa: E402
import riemann_parts  # noqa: E402
from submission import FBFingerprint  # noqa: E402

ARMS = ("real_6b", "real_5b", "complex_6b", "complex_5b", "herm_real_6b", "herm_real_5b")
CANDIDATE_ARMS = ("real_5b", "complex_6b", "complex_5b")
BANK_DIR = REPO / "outputs" / "t2-int-harness"


def id_person_metrics(person: np.ndarray, fp_logp: np.ndarray, K: int):
    """Per-person identification recall and mean negative log-likelihood on test windows."""
    pred = fp_logp.argmax(axis=-1)
    rec = np.zeros(K, dtype=np.float64)
    nll = np.zeros(K, dtype=np.float64)
    for k in range(K):
        m = person == k
        rec[k] = float((pred[m] == k).mean())
        nll[k] = float((-fp_logp[m, k]).mean())
    return rec, nll


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """Exact Clopper-Pearson binomial confidence interval (Rule FM-11)."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def paired_summary(opt_by_ds: dict, ref_by_ds: dict, seed: int = 0, n_boot: int = 5000) -> dict:
    """Pooled per-person paired difference summary with 5,000-resample bootstrap 95% CI
    and Clopper-Pearson interval on non-tied wins."""
    d = np.concatenate([np.asarray(opt_by_ds[k]) - np.asarray(ref_by_ds[k]) for k in opt_by_ds])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    better = int((d > 1e-12).sum())
    worse = int((d < -1e-12).sum())
    equal = int(len(d) - better - worse)
    n_non_tied = better + worse
    return {
        "mean": float(d.mean()),
        "ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
        "better": better,
        "worse": worse,
        "equal": equal,
        "n": int(len(d)),
        "win_ci_clopper_pearson": clopper_pearson_ci(better, n_non_tied),
        "per_person_deltas": [float(x) for x in d],
    }


def criterion_b_verdict(
    id_acc_cmp: dict,
    id_nll_gain_cmp: dict,
    task_cmp: dict,
    sim2_task_delta: float,
    max_export_diff: float,
):
    """Check Protocol 00 §5.2 Criterion B (fingerprint replacement bar)."""
    reasons = []
    id_acc_ok = id_acc_cmp["mean"] >= 0.010 and id_acc_cmp["ci"][0] > 0.0
    id_nll_ok = id_nll_gain_cmp["mean"] >= 0.020 and id_nll_gain_cmp["ci"][0] > 0.0
    if not (id_acc_ok or id_nll_ok):
        reasons.append(
            f"cross-day ID neither gained >=0.010 acc with CI>0 ({id_acc_cmp['mean']:+.4f} "
            f"[{id_acc_cmp['ci'][0]:+.4f}, {id_acc_cmp['ci'][1]:+.4f}]) nor reduced NLL by >=0.020 "
            f"with CI>0 ({id_nll_gain_cmp['mean']:+.4f} [{id_nll_gain_cmp['ci'][0]:+.4f}, "
            f"{id_nll_gain_cmp['ci'][1]:+.4f}])"
        )
    if task_cmp["mean"] < 0.000 or task_cmp["ci"][0] < -0.002:
        reasons.append(
            f"cross-day task delta {task_cmp['mean']:+.4f} "
            f"[{task_cmp['ci'][0]:+.4f}, {task_cmp['ci'][1]:+.4f}] failed non-degradation"
        )
    if sim2_task_delta < -0.002:
        reasons.append(f"dreyer_sim2 task delta {sim2_task_delta:+.4f} < -0.002")
    if max_export_diff >= 1e-4:
        reasons.append(f"max export probability diff {max_export_diff:.2e} >= 1e-4")
    return (len(reasons) == 0, reasons if reasons else ["passes Criterion B"])


def _state_bytes(state: dict) -> int:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.tell()


def _lsm(logits: torch.Tensor) -> np.ndarray:
    return torch.log_softmax(logits.double(), dim=-1).numpy()


def eval_dataset_arms(d: dict) -> dict:
    """Fit all 6 arms on d['calib'] and evaluate on d['test']."""
    X_cal, p_cal = d["X"][d["calib"]], d["person"][d["calib"]]
    X_te, p_te = d["X"][d["test"]], d["person"][d["test"]]
    K, sfreq = len(d["people"]), float(d["sfreq"])
    n_chans, n_times = X_cal.shape[1], X_cal.shape[2]
    Xt = torch.from_numpy(X_te)

    out = {}
    # 1. Real OAS arms (6b and 5b)
    for arm_name, bands in (("real_6b", complex_fp.BANDS_6), ("real_5b", complex_fp.BANDS_5)):
        t0 = time.time()
        st = riemann_parts.fit_fingerprint(X_cal, p_cal, sfreq, bands=bands)
        mod = FBFingerprint(len(bands), n_chans, n_times, K)
        mod.load_state_dict(st)
        t_inf0 = time.time()
        with torch.inference_mode():
            fp_logp = _lsm(mod(Xt))
        inf_s = time.time() - t_inf0
        rec, nll = id_person_metrics(p_te, fp_logp, K)
        out[arm_name] = {
            "fp_logp": fp_logp,
            "id_rec": rec,
            "id_nll": nll,
            "id_bal_acc": float(rec.mean()),
            "id_mean_nll": float(nll.mean()),
            "state_bytes": _state_bytes(st),
            "fit_sec": time.time() - t0,
            "infer_sec": inf_s,
            "max_export_prob_diff": 0.0,
        }

    # 2. Complex Hermitian arms (6b and 5b)
    for arm_name, bands in (("complex_6b", complex_fp.BANDS_6), ("complex_5b", complex_fp.BANDS_5)):
        t0 = time.time()
        st, ref_fn = complex_fp.fit_complex_fingerprint(X_cal, p_cal, sfreq, bands=bands)
        mod = complex_fp.ComplexFBFingerprint(len(bands), n_chans, n_times, K)
        mod.load_state_dict(st)
        t_inf0 = time.time()
        with torch.inference_mode():
            fp_logp = _lsm(mod(Xt))
        inf_s = time.time() - t_inf0
        ref_logp = ref_fn(X_te)
        max_diff = float(np.abs(np.exp(fp_logp) - np.exp(ref_logp)).max())
        top1_agree = float((fp_logp.argmax(-1) == ref_logp.argmax(-1)).mean())
        rec, nll = id_person_metrics(p_te, fp_logp, K)
        out[arm_name] = {
            "fp_logp": fp_logp,
            "id_rec": rec,
            "id_nll": nll,
            "id_bal_acc": float(rec.mean()),
            "id_mean_nll": float(nll.mean()),
            "state_bytes": _state_bytes(st),
            "fit_sec": time.time() - t0,
            "infer_sec": inf_s,
            "max_export_prob_diff": max_diff,
            "export_top1_agree": top1_agree,
        }

    # 3. Estimator-matched real Hermitian controls (6b and 5b)
    for arm_name, bands in (("herm_real_6b", complex_fp.BANDS_6), ("herm_real_5b", complex_fp.BANDS_5)):
        t0 = time.time()
        pred_fn = complex_fp.fit_herm_real_control(X_cal, p_cal, sfreq, bands=bands)
        t_inf0 = time.time()
        fp_logp = pred_fn(X_te)
        inf_s = time.time() - t_inf0
        rec, nll = id_person_metrics(p_te, fp_logp, K)
        out[arm_name] = {
            "fp_logp": fp_logp,
            "id_rec": rec,
            "id_nll": nll,
            "id_bal_acc": float(rec.mean()),
            "id_mean_nll": float(nll.mean()),
            "state_bytes": 0,
            "fit_sec": time.time() - t0,
            "infer_sec": inf_s,
            "max_export_prob_diff": 0.0,
        }

    return out


def _single_day_zhou_slices() -> dict:
    """Secondary diagnostic: Zhou 2016 dev people calibrated on session 0 alone or session 1 alone."""
    raw = np.load(data.PATHS["zhou2016"], allow_pickle=True)
    keep = ~np.isin(raw["subject"], data.holdout()["zhou2016"])
    X = raw["X"][keep].astype(np.float32)
    subj = raw["subject"][keep]
    ses = raw["session"][keep]
    people = np.unique(subj)
    person = np.searchsorted(people, subj)
    y = np.searchsorted(np.unique(raw["task"][keep]), raw["task"][keep])

    slices = {}
    for cal_s in (0, 1):
        d_sub = {
            "X": X,
            "y": y,
            "person": person,
            "calib": ses == cal_s,
            "test": ses == 2,
            "people": people,
            "n_classes": int(y.max()) + 1,
            "sfreq": data.SFREQ,
            "name": f"zhou2016_s{cal_s}_only",
        }
        arm_res = eval_dataset_arms(d_sub)
        slices[f"s{cal_s}_to_s2"] = {
            arm: {
                "id_bal_acc": arm_res[arm]["id_bal_acc"],
                "id_mean_nll": arm_res[arm]["id_mean_nll"],
                "id_rec": [float(x) for x in arm_res[arm]["id_rec"]],
                "id_nll": [float(x) for x in arm_res[arm]["id_nll"]],
            }
            for arm in ARMS
        }
    return slices


def _routing_split_acc(
    name: str,
    fp_logp_ref: np.ndarray,
    fp_logp_arm: np.ndarray,
    seeds=(0, 1, 2),
    bank_dir: Path = BANK_DIR,
) -> dict:
    """Task accuracy (plain window mean across seeds) on windows where real_6b ID was right vs wrong."""
    right_mask = None
    acc_right, acc_wrong = [], []
    for s in seeds:
        b = np.load(Path(bank_dir) / f"bank_{name}_seed{s}.npz")
        y, person = b["y"], b["person"]
        if right_mask is None:
            right_mask = fp_logp_ref.argmax(-1) == person
        lp = metrics.route(fp_logp_arm, metrics.c3_logp(b))
        pred = lp.argmax(-1)
        if right_mask.any():
            acc_right.append(float((pred[right_mask] == y[right_mask]).mean()))
        if (~right_mask).any():
            acc_wrong.append(float((pred[~right_mask] == y[~right_mask]).mean()))
    return {
        "n_right": int(right_mask.sum()),
        "n_wrong": int((~right_mask).sum()),
        "acc_on_ref_right": float(np.mean(acc_right)) if acc_right else None,
        "acc_on_ref_wrong": float(np.mean(acc_wrong)) if acc_wrong else None,
    }


def run_all(threads: int = 2) -> dict:
    torch.set_num_threads(threads)
    t_start = time.time()

    per_ds = {}
    for name in (*data.DATASETS, "dreyer_sim2"):
        print(f"[evaluate_2x2] Loading and fitting 6 arms on {name}...", flush=True)
        d = data.load_dreyer_sim2() if name == "dreyer_sim2" else data.load(name, part="dev")
        per_ds[name] = eval_dataset_arms(d)

        # Verify real_6b reproduces the banked fp_logp
        b0 = np.load(BANK_DIR / f"bank_{name}_seed0.npz")
        bank_diff = float(np.abs(np.exp(per_ds[name]["real_6b"]["fp_logp"]) - np.exp(b0["fp_logp"])).max())
        per_ds[name]["real_6b"]["bank_repro_max_prob_diff"] = bank_diff

        # Compute seed-averaged per-person task balanced accuracy for each arm
        seeds = (0,) if name == "dreyer_sim2" else (0, 1, 2)
        for arm in ARMS:
            fp_lp = per_ds[name][arm]["fp_logp"]
            task_scores = metrics.seed_mean_scores(
                name, lambda b, _fp=fp_lp: metrics.route(_fp, metrics.c3_logp(b)), seeds=seeds
            )
            per_ds[name][arm]["task_scores"] = task_scores
            per_ds[name][arm]["task_bal_acc"] = float(task_scores.mean())
            per_ds[name][arm]["routing_split"] = _routing_split_acc(
                name, per_ds[name]["real_6b"]["fp_logp"], fp_lp, seeds=seeds
            )

        # Also store oracle and pooled task scores
        oracle_scores = metrics.seed_mean_scores(
            name, lambda b: metrics.rows(b)["oracle"], seeds=seeds
        )
        pooled_scores = metrics.seed_mean_scores(
            name, lambda b: metrics.rows(b)["pooled"], seeds=seeds
        )
        per_ds[name]["_oracle_task_scores"] = oracle_scores
        per_ds[name]["_oracle_task_bal_acc"] = float(oracle_scores.mean())
        per_ds[name]["_pooled_task_bal_acc"] = float(pooled_scores.mean())

    # Pooled 17-dev-person comparisons against real_6b (row 2)
    comparisons = {}
    ref_id_rec = {n: per_ds[n]["real_6b"]["id_rec"] for n in data.DATASETS}
    ref_id_nll = {n: per_ds[n]["real_6b"]["id_nll"] for n in data.DATASETS}
    ref_task = {n: per_ds[n]["real_6b"]["task_scores"] for n in data.DATASETS}

    for arm in ARMS:
        arm_id_rec = {n: per_ds[n][arm]["id_rec"] for n in data.DATASETS}
        arm_id_nll = {n: per_ds[n][arm]["id_nll"] for n in data.DATASETS}
        # NLL gain = ref_nll - arm_nll (positive means lower/better NLL)
        id_acc_cmp = paired_summary(arm_id_rec, ref_id_rec)
        id_nll_gain_cmp = paired_summary(ref_id_nll, arm_id_nll)
        task_cmp = paired_summary({n: per_ds[n][arm]["task_scores"] for n in data.DATASETS}, ref_task)
        sim2_delta = float(
            per_ds["dreyer_sim2"][arm]["task_bal_acc"] - per_ds["dreyer_sim2"]["real_6b"]["task_bal_acc"]
        )
        max_exp_diff = max(per_ds[n][arm]["max_export_prob_diff"] for n in (*data.DATASETS, "dreyer_sim2"))

        crit_a_ok, crit_a_why = metrics.adopt(task_cmp, sim2_delta)
        crit_b_ok, crit_b_reasons = criterion_b_verdict(
            id_acc_cmp, id_nll_gain_cmp, task_cmp, sim2_delta, max_exp_diff
        )

        # Pooled 17-person means
        all_rec = np.concatenate([arm_id_rec[n] for n in data.DATASETS])
        all_nll = np.concatenate([arm_id_nll[n] for n in data.DATASETS])
        all_task = np.concatenate([per_ds[n][arm]["task_scores"] for n in data.DATASETS])

        comparisons[arm] = {
            "pooled_17_id_acc": float(all_rec.mean()),
            "pooled_17_id_nll": float(all_nll.mean()),
            "pooled_17_task_acc": float(all_task.mean()),
            "id_acc_vs_real_6b": id_acc_cmp,
            "id_nll_gain_vs_real_6b": id_nll_gain_cmp,
            "task_acc_vs_real_6b": task_cmp,
            "sim2_task_delta": sim2_delta,
            "max_export_prob_diff": max_exp_diff,
            "criterion_a_track2_adopt": {"passed": crit_a_ok, "reason": crit_a_why},
            "criterion_b_fp_replace": {"passed": crit_b_ok, "reasons": crit_b_reasons},
        }

    # Diagnostic comparisons: complex vs herm_real (phase-lag value-add)
    phase_lag_6b = paired_summary(
        {n: per_ds[n]["complex_6b"]["id_rec"] for n in data.DATASETS},
        {n: per_ds[n]["herm_real_6b"]["id_rec"] for n in data.DATASETS},
    )
    phase_lag_5b = paired_summary(
        {n: per_ds[n]["complex_5b"]["id_rec"] for n in data.DATASETS},
        {n: per_ds[n]["herm_real_5b"]["id_rec"] for n in data.DATASETS},
    )

    # Secondary single-day calibration slice on zhou2016
    zhou_single_day = _single_day_zhou_slices()

    # Projected 500 Hz size (T=2000, C=43, K=10)
    proj_500hz_bytes = {
        "real_6b_float32_MB": (6 * 2000 * 2000 * 4 + 6 * 43 * 43 * 4 + (6 * 43 * 44 // 2) * 10 * 4) / 1e6,
        "complex_6b_complex64_MB": (6 * 2000 * 2000 * 8 + 6 * 43 * 43 * 8 + (6 * 43 * 43) * 10 * 4) / 1e6,
        "complex_5b_complex64_MB": (5 * 2000 * 2000 * 8 + 5 * 43 * 43 * 8 + (5 * 43 * 43) * 10 * 4) / 1e6,
    }

    # Winner selection under Criterion B
    qualifying = [a for a in CANDIDATE_ARMS if comparisons[a]["criterion_b_fp_replace"]["passed"]]
    tie_order = {"real_5b": 0, "real_6b": 1, "complex_5b": 2, "complex_6b": 3}
    if qualifying:
        qualifying.sort(
            key=lambda a: (
                round(comparisons[a]["pooled_17_id_nll"], 3),
                -comparisons[a]["pooled_17_id_acc"],
                tie_order[a],
            )
        )
        selected = qualifying[0]
    else:
        selected = "real_6b"

    # Clean serializable per-dataset table
    ds_summary = {}
    for name in (*data.DATASETS, "dreyer_sim2"):
        ds_summary[name] = {
            "oracle_task_bal_acc": per_ds[name]["_oracle_task_bal_acc"],
            "pooled_task_bal_acc": per_ds[name]["_pooled_task_bal_acc"],
            "arms": {
                arm: {
                    "id_bal_acc": per_ds[name][arm]["id_bal_acc"],
                    "id_mean_nll": per_ds[name][arm]["id_mean_nll"],
                    "task_bal_acc": per_ds[name][arm]["task_bal_acc"],
                    "id_rec": [float(x) for x in per_ds[name][arm]["id_rec"]],
                    "id_nll": [float(x) for x in per_ds[name][arm]["id_nll"]],
                    "task_scores": [float(x) for x in per_ds[name][arm]["task_scores"]],
                    "state_bytes": per_ds[name][arm]["state_bytes"],
                    "fit_sec": per_ds[name][arm]["fit_sec"],
                    "infer_sec": per_ds[name][arm]["infer_sec"],
                    "max_export_prob_diff": per_ds[name][arm]["max_export_prob_diff"],
                    "routing_split": per_ds[name][arm]["routing_split"],
                }
                for arm in ARMS
            },
        }
        ds_summary[name]["arms"]["real_6b"]["bank_repro_max_prob_diff"] = per_ds[name]["real_6b"][
            "bank_repro_max_prob_diff"
        ]

    results = {
        "protocol": "research/complex-fingerprint/experiments/00-protocol/protocol.md",
        "elapsed_sec": time.time() - t_start,
        "selected_fingerprint": selected,
        "qualifying_candidates_criterion_b": qualifying,
        "datasets": ds_summary,
        "pooled_17_dev_comparisons": comparisons,
        "phase_lag_vs_herm_real": {"6b": phase_lag_6b, "5b": phase_lag_5b},
        "zhou2016_single_day_slices": zhou_single_day,
        "projected_500hz_size": proj_500hz_bytes,
    }
    return results


def format_markdown_report(res: dict) -> str:
    """Dynamically format markdown report from 2x2_results.json (Rule FM-11)."""
    lines = [
        "# Results: Protocol 00 — 2 × 2 Complex Hermitian × 5/6-Band Fingerprint",
        "",
        f"- **Protocol:** [`{res['protocol']}`](../experiments/00-protocol/protocol.md)",
        f"- **Selected fingerprint (Criterion B):** **`{res['selected_fingerprint']}`**",
        f"- **Qualifying candidates (Criterion B):** `{res['qualifying_candidates_criterion_b']}`",
        "",
        "## 1. Pooled 17 Cross-Day Dev People (`bnci2014` 6 + `bnci2015` 8 + `zhou2016` 3)",
        "",
        "| Arm | ID Bal Acc | Δ ID Acc vs `real_6b` [95% CI] (B/W/E) | ID NLL | −Δ NLL Gain vs `real_6b` [95% CI] (B/W) | Task Bal Acc | Δ Task vs `real_6b` [95% CI] (B/W/E) | `sim2` Δ Task | Max Export |Δp| | Criterion A | Criterion B |",
        "|---|---:|---|---:|---|---:|---|---:|---:|:---:|:---:|",
    ]
    for arm in ARMS:
        c = res["pooled_17_dev_comparisons"][arm]
        ia = c["id_acc_vs_real_6b"]
        inll = c["id_nll_gain_vs_real_6b"]
        ta = c["task_acc_vs_real_6b"]
        ca = "PASS" if c["criterion_a_track2_adopt"]["passed"] else "FAIL"
        cb = "**PASS**" if c["criterion_b_fp_replace"]["passed"] else "FAIL"
        lines.append(
            f"| `{arm}` | {c['pooled_17_id_acc']:.4f} | "
            f"{ia['mean']:+.4f} [{ia['ci'][0]:+.4f}, {ia['ci'][1]:+.4f}] ({ia['better']}/{ia['worse']}/{ia['equal']}) | "
            f"{c['pooled_17_id_nll']:.4f} | "
            f"{inll['mean']:+.4f} [{inll['ci'][0]:+.4f}, {inll['ci'][1]:+.4f}] ({inll['better']}/{inll['worse']}) | "
            f"{c['pooled_17_task_acc']:.4f} | "
            f"{ta['mean']:+.4f} [{ta['ci'][0]:+.4f}, {ta['ci'][1]:+.4f}] ({ta['better']}/{ta['worse']}/{ta['equal']}) | "
            f"{c['sim2_task_delta']:+.4f} | {c['max_export_prob_diff']:.2e} | {ca} | {cb} |"
        )

    lines.extend([
        "",
        "## 2. Per-Dataset Breakdown",
        "",
        "| Dataset | Arm | ID Bal Acc | ID Mean NLL | Task Bal Acc (Oracle / Pooled) | Ref-Right Task Acc | Ref-Wrong Task Acc | State Size (MB) |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ])
    for ds_name, ds in res["datasets"].items():
        ora = ds["oracle_task_bal_acc"]
        pld = ds["pooled_task_bal_acc"]
        for arm in ARMS:
            a = ds["arms"][arm]
            rs = a["routing_split"]
            r_right = f"{rs['acc_on_ref_right']:.4f}" if rs["acc_on_ref_right"] is not None else "—"
            r_wrong = f"{rs['acc_on_ref_wrong']:.4f}" if rs["acc_on_ref_wrong"] is not None else "—"
            mb = a["state_bytes"] / 1e6 if a["state_bytes"] else 0.0
            lines.append(
                f"| `{ds_name}` | `{arm}` | {a['id_bal_acc']:.4f} | {a['id_mean_nll']:.4f} | "
                f"{a['task_bal_acc']:.4f} ({ora:.4f} / {pld:.4f}) | {r_right} (n={rs['n_right']}) | "
                f"{r_wrong} (n={rs['n_wrong']}) | {mb:.2f} |"
            )

    pl6 = res["phase_lag_vs_herm_real"]["6b"]
    pl5 = res["phase_lag_vs_herm_real"]["5b"]
    lines.extend([
        "",
        "## 3. Estimator vs Phase-Lag Decomposition (`complex` vs `herm_real`)",
        "",
        f"- **6-band (`complex_6b − herm_real_6b`):** {pl6['mean']:+.4f} ID acc "
        f"[{pl6['ci'][0]:+.4f}, {pl6['ci'][1]:+.4f}], {pl6['better']} better / {pl6['worse']} worse / {pl6['equal']} equal.",
        f"- **5-band (`complex_5b − herm_real_5b`):** {pl5['mean']:+.4f} ID acc "
        f"[{pl5['ci'][0]:+.4f}, {pl5['ci'][1]:+.4f}], {pl5['better']} better / {pl5['worse']} worse / {pl5['equal']} equal.",
        "",
        "## 4. Secondary Single-Day Calibration Diagnostic on `zhou2016` (3 dev people)",
        "",
        "| Slice | `real_6b` Acc / NLL | `real_5b` Acc / NLL | `complex_6b` Acc / NLL | `complex_5b` Acc / NLL | `herm_real_6b` Acc / NLL | `herm_real_5b` Acc / NLL |",
        "|---|---|---|---|---|---|---|",
    ])
    for sl_name, sl in res["zhou2016_single_day_slices"].items():
        cells = [f"{sl[a]['id_bal_acc']:.4f} / {sl[a]['id_mean_nll']:.4f}" for a in ARMS]
        lines.append(f"| `{sl_name}` | " + " | ".join(cells) + " |")

    p500 = res["projected_500hz_size"]
    lines.extend([
        "",
        "## 5. Projected Buffer Sizes at Sealed-Phase 500 Hz (`T=2000, C=43, K=10`)",
        "",
        f"- `real_6b` (`float32`): **{p500['real_6b_float32_MB']:.1f} MB**",
        f"- `complex_6b` (`complex64`): **{p500['complex_6b_complex64_MB']:.1f} MB**",
        f"- `complex_5b` (`complex64`): **{p500['complex_5b_complex64_MB']:.1f} MB**",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    res_dir = HERE.parent / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_json = res_dir / "2x2_results.json"
    out_md = res_dir / "2x2_results.md"
    if args.report_only:
        res = json.loads(out_json.read_text())
    else:
        res = run_all(threads=args.threads)
        out_json.write_text(json.dumps(res, indent=2))
        print(f"Saved {out_json}")
    out_md.write_text(format_markdown_report(res))
    print(f"Saved {out_md}")
