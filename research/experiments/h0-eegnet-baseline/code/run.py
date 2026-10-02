"""H0: EEGNet fingerprint, train R1-R2, test R3, fixed 150 epochs (see ../protocol.md)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1] / "src"))
import fp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = fp.load_eval()
    X, t, run = d["X"], d["person"], d["run"]
    tr, te = run <= 1, run == 2
    K = len(d["people"])
    for seed in map(int, args.seeds.split(",")):
        t0 = time.time()
        model = fp.make_eegnet(X.shape[1], K, X.shape[2])

        def monitor(m, epoch):
            mm = fp.metrics(fp.predict_proba(m, X[te]), t[te])
            return {"r3_bal_acc": mm["bal_acc"], "r3_nll": mm["nll"]}

        model, curve = fp.fit_fixed(model, X[tr], t[tr], epochs=args.epochs, seed=seed,
                                    monitor=monitor, name=f"seed{seed}")
        P = fp.predict_proba(model, X[te])
        fp.save_run(EXP, f"eegnet_seed{seed}", P, t[te],
                    {"seed": seed, "epochs": args.epochs, "threads": args.threads,
                     "curve": curve, "wall_s": round(time.time() - t0, 1)})
        torch.save(model.state_dict(), fp.OUT / EXP.name / f"eegnet_seed{seed}.pt")


if __name__ == "__main__":
    main()
