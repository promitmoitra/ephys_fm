"""H11: multi-day calibration of the fingerprint (see ../protocol.md)."""

import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import extra_data  # noqa: E402
import fp  # noqa: E402
import riemann  # noqa: E402

BANKS = {"fb6": riemann.BANDS["fb"], "fine18": [(f, f + 2) for f in range(4, 40, 2)]}


def half_per_person(mask, person, rng):
    """Keep a random half of each person's windows inside `mask`."""
    keep = np.zeros_like(mask)
    for p in np.unique(person[mask]):
        idx = np.flatnonzero(mask & (person == p))
        keep[rng.choice(idx, len(idx) // 2, replace=False)] = True
    return keep


def main():
    rng = np.random.default_rng(0)
    for name in ("Zhou2016", "BNCI2015_001"):
        d = extra_data.load(name)
        X, t, s = d["X"], d["person"], d["session"]
        te = s == 2
        arms = {"S1": s == 0, "S2": s == 1, "S1+S2": s <= 1,
                "S1+S2_halfmatched": half_per_person(s == 0, t, rng) | half_per_person(s == 1, t, rng)}
        for bank, bands in BANKS.items():
            for arm, tr in arms.items():
                F_tr, F_te = riemann.tangent_features(X[tr], X[te], bands=bands)
                P = riemann.fit_predict_lr(F_tr, t[tr], F_te, C=1.0)
                # Scoring keeps all K people as candidate classes; only people with a
                # session 3 appear in the test set.
                keep = np.unique(t[te])
                fp.save_run(EXP, f"{name}_{bank}_{arm}", P, t[te],
                            {"dataset": name, "bank": bank, "arm": arm,
                             "n_train": int(tr.sum()), "test_people": keep.tolist()})


if __name__ == "__main__":
    main()
