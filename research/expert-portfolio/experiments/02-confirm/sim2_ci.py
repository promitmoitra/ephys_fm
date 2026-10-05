"""Person-bootstrap CIs for the sim2 differences in analysis.md (mean over seeds per person)."""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[4] / "track2" / "results"
rows = [json.loads((RES / f"dreyer_sim_seed{s}_fb_riemann_c3_sim2_shallow_portfolio.json").read_text())["scores"]
        for s in range(3)]
people = list(rows[0]["EEGNet experts, oracle"]["per_person"])


def pp(key):
    return np.array([[r[key]["per_person"][p] for p in people] for r in rows]).mean(0)


def ci(d, seed=0, n=5000):
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return (round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4), int((d > 0).sum()), int((d < 0).sum()))


E, ET, F = pp("EEGNet experts, oracle"), pp("E+T (C3), oracle"), pp("full combination, oracle")
for name, d in [("E+T+S - E+T", F - ET), ("E+T - EEGNet", ET - E), ("E+T+S - EEGNet", F - E)]:
    print(name, "(mean, CI lo, CI hi, better, worse):", ci(d))
