"""P0: can REVE run offline with shipped positions, and how fast does it embed on CPU?"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "outputs" / "t2-portfolio"
POS_DIR = OUT / "reve_positions"


def main():
    torch.set_num_threads(4)
    POS_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. Prefetch (online, once): weights into the HF cache, positions into POS_DIR.
    os.environ["REVE_POSITIONS_PATH"] = str(POS_DIR)
    from braindecode.models import REVE
    t0 = time.time()
    # The plain call may fail if REVE's constructor needs the input geometry; then retry with it.
    kwargs = {}
    try:
        enc = REVE.from_pretrained("brain-bzh/reve-base")
    except Exception as e:                                  # record and retry
        report["plain_call_error"] = f"{type(e).__name__}: {e}"[:300]
        kwargs = dict(n_outputs=2, n_chans=27, n_times=800, sfreq=200)
        enc = REVE.from_pretrained("brain-bzh/reve-base", **kwargs)
    report["constructor_kwargs"] = kwargs
    report["download_s"] = round(time.time() - t0, 1)
    report["n_params"] = int(sum(p.numel() for p in enc.parameters()))
    report["positions_file"] = (POS_DIR / "reve_positions.json").exists()

    # 2. Offline reload: no network, positions from POS_DIR only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    enc = REVE.from_pretrained("brain-bzh/reve-base", **kwargs).eval()
    report["offline_load"] = "ok"

    # 3. Channel coverage for both datasets.
    bank = json.loads((POS_DIR / "reve_positions.json").read_text())
    for name, path in [("dreyer", "dreyer_windows.npz"), ("bnci", "tangermann_windows.npz")]:
        d = np.load(REPO / "data" / "experiments" / path, allow_pickle=True)
        chs = [str(c) for c in d["ch_names"]]
        report[f"{name}_missing_channels"] = [c for c in chs if c not in bank]

    # 4. Feature shape and timing on 1,000 Dreyer windows (zeros-safe standardised input).
    d = np.load(REPO / "data" / "experiments" / "dreyer_windows.npz", allow_pickle=True)
    chs = [str(c) for c in d["ch_names"]]
    pos = torch.tensor([bank[c] for c in chs], dtype=torch.float32)
    from scipy.signal import resample_poly
    X = resample_poly(d["X"][:1000], 5, 3, axis=-1).astype(np.float32)        # 120 -> 200 Hz
    X = (X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-6)
    X = np.clip(X, -15, 15)
    with torch.inference_mode():
        f = enc(torch.from_numpy(X[:2]), pos=pos.expand(2, -1, -1), return_features=True)
        report["features_shape"] = list(f["features"].shape)
        t0 = time.time()
        for i in range(0, 1000, 50):
            xb = torch.from_numpy(X[i:i + 50])
            enc(xb, pos=pos.expand(len(xb), -1, -1), return_features=True)
        report["embed_s_per_1000"] = round(time.time() - t0, 1)
    report["est_dreyer_all_min"] = round(report["embed_s_per_1000"] * 20.792 / 60, 1)
    # The constructor call travels with the positions file (read by load_reve_encoder).
    (POS_DIR / "reve_kwargs.json").write_text(json.dumps(kwargs))
    print(json.dumps(report, indent=2))
    (Path(__file__).parent / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
