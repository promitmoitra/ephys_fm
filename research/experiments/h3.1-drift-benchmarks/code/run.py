"""H3.1: fingerprints under drift. D1 Dreyer R1 -> R2 / R3; D2 BNCI session 1 -> 2."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import fp  # noqa: E402
import riemann  # noqa: E402


def benchmarks():
    d = fp.load_eval()
    X, t, run = d["X"], d["person"], d["run"]
    yield "D1_R1toR2", X[run == 0], t[run == 0], X[run == 1], t[run == 1]
    yield "D1_R1toR3", X[run == 0], t[run == 0], X[run == 2], t[run == 2]
    b = fp.load_bnci()
    s1, s2 = b["session"] == 0, b["session"] == 1
    yield "D2_bnci_s1tos2", b["X"][s1], b["person"][s1], b["X"][s2], b["person"][s2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ts_broad,ts_fb,eegnet")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    models = args.models.split(",")
    cache = {}
    for bench, Xtr, ttr, Xte, tte in benchmarks():
        K = int(ttr.max()) + 1
        for feat in ("broad", "fb"):
            if f"ts_{feat}" not in models:
                continue
            t0 = time.time()
            F_tr, F_te = riemann.tangent_features(Xtr, Xte, bands=riemann.BANDS[feat])
            P = riemann.fit_predict_lr(F_tr, ttr, F_te, C=1.0)
            fp.save_run(EXP, f"{bench}_ts_{feat}_C1", P, tte,
                        {"bench": bench, "wall_s": round(time.time() - t0, 1)})
        if "eegnet" not in models:
            continue
        # R1 -> R2 and R1 -> R3 share the training set: train once per seed.
        for seed in map(int, args.seeds.split(",")):
            key = (bench.split("to")[0], seed)
            t0 = time.time()
            if key not in cache:
                model = fp.make_eegnet(Xtr.shape[1], K, Xtr.shape[2])
                for k in [k for k in cache if k[0] != key[0]]:
                    del cache[k]
                cache[key], _ = fp.fit_fixed(model, Xtr, ttr, epochs=150, seed=seed,
                                             name=f"{bench} seed{seed}")
            P = fp.predict_proba(cache[key], Xte)
            fp.save_run(EXP, f"{bench}_eegnet_seed{seed}", P, tte,
                        {"bench": bench, "seed": seed, "epochs": 150,
                         "threads": args.threads, "wall_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
