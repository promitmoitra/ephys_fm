"""H3: Riemannian fingerprints, train R1-R2, test R3 (see ../protocol.md)."""

import sys
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import fp  # noqa: E402
import riemann  # noqa: E402


def main():
    d = fp.load_eval()
    X, t, run = d["X"], d["person"], d["run"]
    tr, te = run <= 1, run == 2
    for feat in ("broad", "fb"):
        t0 = time.time()
        F_tr, F_te = riemann.tangent_features(X[tr], X[te], bands=riemann.BANDS[feat])
        t_feat = time.time() - t0
        for C in (1.0, 0.1):
            t1 = time.time()
            P = riemann.fit_predict_lr(F_tr, t[tr], F_te, C=C)
            fp.save_run(EXP, f"ts_{feat}_C{C:g}", P, t[te],
                        {"features": feat, "C": C, "n_features": int(F_tr.shape[1]),
                         "wall_s": round(t_feat + time.time() - t1, 1)})


if __name__ == "__main__":
    main()
