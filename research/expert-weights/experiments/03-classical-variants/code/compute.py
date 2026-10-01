"""Protocol 03, step 1: cross-fitted predictions and reliability scores for every variant."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from classical import VARIANTS, _load, cross_fit, reliability  # noqa: E402
from combine import OUT  # noqa: E402

E, people = _load()
for v in VARIANTS:
    path = OUT / f"classical_xfit_{v}.npz"
    if path.exists():
        continue
    t0 = time.time()
    P = cross_fit(v, E, people)
    rel_dev, rel_test = reliability(v, E, people)
    np.savez(path, logp=P, rel_dev=rel_dev, rel_test=rel_test)
    print(f"{v}: {time.time() - t0:.0f} s", flush=True)
