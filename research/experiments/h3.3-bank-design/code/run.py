"""H3.3: filter-bank design (see ../protocol.md).

    run.py select     # R12toR3, D1, D2 for all arms; applies the locked rule
    run.py heldout    # D3 (BNCI2015_001), D4 (Zhou2016) for fb6, fine18, selected
"""

import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import extra_data  # noqa: E402
import fp  # noqa: E402
import riemann  # noqa: E402

FB6 = riemann.BANDS["fb"]
ARMS = {
    "fb6": FB6,
    "fb5_nodelta": FB6[1:],
    "fine9_4hz": [(f, f + 4) for f in range(4, 40, 4)],
    "fine18": [(f, f + 2) for f in range(4, 40, 2)],
    "fine36": [(f, f + 1) for f in range(4, 40)],
    "fine11_8_30": [(f, f + 2) for f in range(8, 30, 2)],
}


def run_bench(bench, Xtr, ttr, Xte, tte, arms):
    bands = sorted({b for a in arms for b in ARMS[a]})
    feats = {b: riemann.tangent_features(Xtr, Xte, bands=[b]) for b in bands}
    out = {}
    for a in arms:
        F_tr = np.hstack([feats[b][0] for b in ARMS[a]])
        F_te = np.hstack([feats[b][1] for b in ARMS[a]])
        P = riemann.fit_predict_lr(F_tr, ttr, F_te, C=1.0)
        out[a] = fp.save_run(EXP, f"{bench}_{a}", P, tte, {"bench": bench, "arm": a})
    return out


def select():
    d = fp.load_eval()
    X, t, run = d["X"], d["person"], d["run"]
    b = fp.load_bnci()
    s1, s2 = b["session"] == 0, b["session"] == 1
    arms = list(ARMS)
    res = {
        "R12toR3": run_bench("R12toR3", X[run <= 1], t[run <= 1], X[run == 2], t[run == 2], arms),
        "D1_R1toR3": run_bench("D1_R1toR3", X[run == 0], t[run == 0], X[run == 2], t[run == 2], arms),
        "D2_bnci_s1tos2": run_bench("D2_bnci_s1tos2", b["X"][s1], b["person"][s1],
                                    b["X"][s2], b["person"][s2], arms),
    }
    ok = [a for a in arms
          if res["D1_R1toR3"][a]["bal_acc"] >= res["D1_R1toR3"]["fb6"]["bal_acc"] - 0.005]
    best = min(res["D2_bnci_s1tos2"][a]["nll"] for a in ok)
    tied = [a for a in ok if res["D2_bnci_s1tos2"][a]["nll"] <= best + 0.005]
    chosen = min(tied, key=lambda a: len(ARMS[a]))
    (EXP / "results" / "selection.json").write_text(json.dumps(
        {"eligible": ok, "tied": tied, "selected": chosen}, indent=2))
    fp.log(f"eligible {ok}; tied {tied}; SELECTED {chosen}")


def heldout():
    chosen = json.loads((EXP / "results" / "selection.json").read_text())["selected"]
    arms = sorted({"fb6", "fine18", chosen})
    for name in ("BNCI2015_001", "Zhou2016"):
        d = extra_data.load(name)
        tr, te = d["session"] == 0, d["session"] >= 1
        run_bench(f"heldout_{name}", d["X"][tr], d["person"][tr], d["X"][te], d["person"][te], arms)


if __name__ == "__main__":
    {"select": select, "heldout": heldout}[sys.argv[1]]()
