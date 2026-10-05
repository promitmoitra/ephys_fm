"""Report-only: E+T vs the full combination on R4–R6, seeds 0–2 (upper bounds)."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "research" / "expert-portfolio" / "src"))
from portfolio import LOOPB, OUT, PortfolioLogLinear, bal, dev_bank, nll, own  # noqa: E402

LAM = float(sys.argv[1]); STREAMS = sys.argv[2].split(",")       # e.g. 0.1 shallow,reve
dev = dev_bank(LAM)
c = np.load(LOOPB / "classical_xfit_ts_C0.1.npz")
res = {}
for seed, name in [(0, "bank_test_packaged.npz"), (1, "bank_test_seed1.npz"), (2, "bank_test_seed2.npz")]:
    t = dict(np.load(LOOPB / name, allow_pickle=True))
    t["ts"] = np.load(LOOPB / "classical_test_ts_C0.1.npy")
    t["rel"] = np.broadcast_to(c["rel_test"][None, :, None], t["eegnet"].shape[:2] + (1,)).copy()
    if "shallow" in STREAMS:
        t["shallow"] = np.load(OUT / f"bank_test_shallow_seed{seed}.npz")["shallow"]
    if "reve" in STREAMS:
        t["reve"] = np.load(OUT / f"bank_test_reve_lam{LAM}.npz")["reve"]
    row = {}
    for label, neural in {"E+T": ["eegnet"], "full": ["eegnet", *STREAMS]}.items():
        ex = [*neural, "ts", "rel"]
        lp = PortfolioLogLinear(ex).fit(own(dev, ex), dev["y"]).predict(own(t, ex))
        row[label] = {"oracle": bal(t["y"], lp), "nll": nll(lp, t["y"])}
    res[seed] = row
(Path(__file__).parent / "results_r4r6.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
