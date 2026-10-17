"""Reference rows 1–3 per dataset (seeds 0–2) and on sim2; person scores averaged over seeds."""
import json, sys
from pathlib import Path
import numpy as np
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import data, metrics  # noqa: E402
OUT = SRC.parents[2] / "outputs" / "t2-int-harness"
RES = SRC.parent / "results"


def scores(name, seeds):
    per = {"pooled": [], "current": [], "oracle": []}
    for s in seeds:
        b = dict(np.load(OUT / f"bank_{name}_seed{s}.npz"))
        K = b["eeg_logp"].shape[1]
        for k, lp in metrics.rows(b).items():
            per[k].append(metrics.person_scores(b["y"], b["person"], lp, K))
        fp_right = b["fp_logp"].argmax(1) == b["person"]
        per.setdefault("fp_top1", []).append(float(fp_right.mean()))
    return {k: (np.mean(v, 0) if k != "fp_top1" else float(np.mean(v))) for k, v in per.items()}


def main():
    RES.mkdir(exist_ok=True)
    out = {name: scores(name, (0, 1, 2)) for name in data.DATASETS}
    out["dreyer_sim2"] = scores("dreyer_sim2", (0,))
    rows = ["| Dataset | People | Fingerprint top-1 | Pooled | Current (row 2) | Oracle (row 3) |",
            "|---|---|---|---|---|---|"]
    for name, r in out.items():
        rows.append(f"| {name} | {len(r['current'])} | {r['fp_top1']:.3f} | {r['pooled'].mean():.3f} | "
                    f"{r['current'].mean():.3f} | {r['oracle'].mean():.3f} |")
    cmp = metrics.pooled_comparison({n: (out[n]["current"], out[n]["pooled"]) for n in data.DATASETS})
    rows.append(f"\nCurrent − pooled, 17 dev cross-day people: {cmp['mean']:+.4f} "
                f"[{cmp['ci'][0]:+.4f}, {cmp['ci'][1]:+.4f}], {cmp['better']}/{cmp['worse']}")
    head = metrics.pooled_comparison({n: (out[n]["oracle"], out[n]["current"]) for n in data.DATASETS})
    rows.append(f"Oracle − current (routing headroom), 17 dev cross-day people: {head['mean']:+.4f} "
                f"[{head['ci'][0]:+.4f}, {head['ci'][1]:+.4f}], {head['better']}/{head['worse']}")
    (RES / "references.md").write_text("# Reference rows (balanced accuracy, person mean)\n\n" + "\n".join(rows) + "\n")
    (RES / "references.json").write_text(json.dumps(
        {n: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for n, r in out.items()}, indent=2))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
