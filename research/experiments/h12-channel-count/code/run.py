"""H12: fingerprint accuracy vs number of channels (see ../protocol.md)."""

import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import fp  # noqa: E402
import riemann  # noqa: E402

FB6 = riemann.BANDS["fb"]


def curve(name, Xtr, ttr, Xte, tte, ks, n_sub=10):
    rng = np.random.default_rng(0)
    C = Xtr.shape[1]
    rows = {}
    for k in ks:
        subsets = [np.arange(C)] if k == C else [np.sort(rng.choice(C, k, replace=False))
                                                  for _ in range(n_sub)]
        ms = []
        for ch in subsets:
            F_tr, F_te = riemann.tangent_features(Xtr[:, ch], Xte[:, ch], bands=FB6)
            m = fp.metrics(riemann.fit_predict_lr(F_tr, ttr, F_te, C=1.0), tte)
            ms.append({"bal_acc": m["bal_acc"], "nll": m["nll"], "channels": ch.tolist()})
        acc = [m["bal_acc"] for m in ms]
        nll = [m["nll"] for m in ms]
        rows[k] = {"bal_acc": [float(np.mean(acc)), float(np.std(acc))],
                   "nll": [float(np.mean(nll)), float(np.std(nll))], "runs": ms}
        fp.log(f"{name} k={k}: bal_acc {np.mean(acc):.3f} ± {np.std(acc):.3f}  "
               f"nll {np.mean(nll):.3f} ± {np.std(nll):.3f}")
    return rows


def main():
    out = {}
    b = fp.load_bnci()
    s1, s2 = b["session"] == 0, b["session"] == 1
    out["D2_bnci_s1tos2"] = curve("D2", b["X"][s1], b["person"][s1], b["X"][s2],
                                  b["person"][s2], [6, 9, 13, 17, 22])
    d = fp.load_eval()
    r = d["run"]
    out["D1_R1toR3"] = curve("D1", d["X"][r == 0], d["person"][r == 0], d["X"][r == 2],
                             d["person"][r == 2], [6, 9, 13, 20, 27])
    (EXP / "results").mkdir(exist_ok=True)
    (EXP / "results" / "curves.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
