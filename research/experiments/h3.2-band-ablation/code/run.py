"""H3.2: band ablations of the tangent-space fingerprint (see ../protocol.md)."""

import sys
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import fp  # noqa: E402
import riemann  # noqa: E402

FB = riemann.BANDS["fb"]


def benchmarks():
    d = fp.load_eval()
    X, t, run = d["X"], d["person"], d["run"]
    yield "R12toR3", X[run <= 1], t[run <= 1], X[run == 2], t[run == 2]
    yield "D1_R1toR3", X[run == 0], t[run == 0], X[run == 2], t[run == 2]
    b = fp.load_bnci()
    s1, s2 = b["session"] == 0, b["session"] == 1
    yield "D2_bnci_s1tos2", b["X"][s1], b["person"][s1], b["X"][s2], b["person"][s2]


def main():
    bands_needed = FB + [None] + [(f, f + 2) for f in range(4, 40, 2)]
    arms = {f"single_{lo}-{hi}": [(lo, hi)] for lo, hi in FB}
    arms.update({f"drop_{lo}-{hi}": [b for b in FB if b != (lo, hi)] for lo, hi in FB})
    arms["fine18"] = [(f, f + 2) for f in range(4, 40, 2)]
    arms["fb_plus_broad"] = FB + [None]
    for bench, Xtr, ttr, Xte, tte in benchmarks():
        t0 = time.time()
        feats = {b: riemann.tangent_features(Xtr, Xte, bands=[b]) for b in bands_needed}
        fp.log(f"{bench}: features in {time.time() - t0:.0f} s")
        for arm, bands in arms.items():
            F_tr = np.hstack([feats[b][0] for b in bands])
            F_te = np.hstack([feats[b][1] for b in bands])
            P = riemann.fit_predict_lr(F_tr, ttr, F_te, C=1.0)
            fp.save_run(EXP, f"{bench}_{arm}", P, tte,
                        {"bench": bench, "arm": arm, "bands": [str(b) for b in bands]})


if __name__ == "__main__":
    main()
