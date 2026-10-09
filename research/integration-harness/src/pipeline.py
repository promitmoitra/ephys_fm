"""Train the shared parts once per dataset and seed, and cache per-window predictions."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
REPO = SRC.parents[2]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2")]
OUT = REPO / "outputs" / "t2-int-harness"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _lsm(z):
    return torch.log_softmax(torch.as_tensor(z, dtype=torch.float64), -1).numpy()


def run(d, seed, threads, epochs, ft_epochs=50, out_dir=OUT):
    import riemann_parts
    from models import make_model
    from submission import FBFingerprint, RiemannExperts
    from train_mixture import fit, logits
    torch.set_num_threads(threads)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X, y, person, unit = d["X"], d["y"], d["person"], d["unit"]
    cal, val, te = d["calib"], d["val"], d["test"]
    K, C, sf = len(d["people"]), d["n_classes"], d["sfreq"]
    n_chans, n_times = X.shape[1:]
    Xt = torch.from_numpy(X[te])
    bank = {"y": y[te], "person": person[te]}

    log(f"{d['name']} seed {seed}: fingerprint")
    st = riemann_parts.fit_fingerprint(X[cal], person[cal], sf)
    fp = FBFingerprint(len(riemann_parts.FP_BANDS), n_chans, n_times, K); fp.load_state_dict(st)
    with torch.inference_mode():
        bank["fp_logp"] = _lsm(fp(Xt).double())
        f_cal = fp.features(torch.from_numpy(X[cal])).double().numpy()
        f_te = fp.features(Xt).double().numpy()
    mu, sd = f_cal.mean(0), f_cal.std(0) + 1e-9
    z_cal, z_te = (f_cal - mu) / sd, (f_te - mu) / sd
    cents = np.stack([z_cal[person[cal] == k].mean(0) for k in range(K)])
    dist = lambda z: np.sqrt(((z[:, None] - cents[None]) ** 2).sum(-1)).min(1) / np.sqrt(z.shape[1])  # noqa: E731
    bank["fp_dist"], bank["fp_dist_ref"] = dist(z_te), np.median(dist(z_cal))
    bank["fp_feat_test"], bank["fp_feat_calib"] = f_te.astype(np.float32), f_cal.astype(np.float32)
    torch.save(st, out_dir / f"fp_{d['name']}_seed{seed}.pt")             # loop D

    log("fingerprint, leave-one-unit-out posteriors on calibration")
    cal_idx = np.where(cal)[0]
    cv = np.zeros((len(cal_idx), K))
    for u in np.unique(unit[cal]):
        tr, ho = cal & (unit != u), cal_idx[unit[cal] == u]
        st_u = riemann_parts.fit_fingerprint(X[tr], person[tr], sf)
        fp_u = FBFingerprint(len(riemann_parts.FP_BANDS), n_chans, n_times, K); fp_u.load_state_dict(st_u)
        with torch.inference_mode():
            cv[unit[cal] == u] = _lsm(fp_u(torch.from_numpy(X[ho])).double())
    bank |= {"calib_idx": cal_idx, "y_calib": y[cal], "person_calib": person[cal],
             "unit_calib": unit[cal], "fp_cv_logp_calib": cv}

    log("pooled EEGNet")
    pool = d.get("pool", np.zeros(len(y), bool)) | (cal & ~val)
    vmask = d.get("pool_val", val)
    pooled, info = fit(make_model("eegnet", n_chans, C, n_times, sf), X[pool], y[pool], lr=1e-3,
                       epochs=epochs, seed=seed, bs=64, X_val=X[vmask], y_val=y[vmask],
                       name="pooled", log_every=25)
    torch.save(pooled.state_dict(), out_dir / f"pooled_{d['name']}_seed{seed}.pt")
    bank["pooled_logp"] = _lsm(logits(pooled, X[te]).double())

    log(f"per-person EEGNet experts ({K})")
    eeg = np.zeros((int(te.sum()), K, C))
    for k in range(K):
        m = cal & (person == k)
        e, _ = fit(copy.deepcopy(pooled), X[m], y[m], lr=1e-4, epochs=ft_epochs, seed=seed, bs=32)
        eeg[:, k] = _lsm(logits(e, X[te]).double())
    bank["eeg_logp"] = eeg

    log("Riemannian experts")
    rs, cs, _, n_out = riemann_parts.fit_riemann(X[cal], y[cal], person[cal], unit[cal], K, C, sf)
    rm = RiemannExperts(K, n_chans, n_times, n_out, C); rm.load_state_dict(rs)
    with torch.inference_mode():
        bank["riemann_logp"] = torch.cat([rm(Xt[i:i + 256]) for i in range(0, len(Xt), 256)]).numpy()
    bank["rel"] = cs["rel"].numpy()
    path = out_dir / f"bank_{d['name']}_seed{seed}.npz"
    np.savez(path, **bank, pooled_info=str(info))
    log(f"saved {path}")
    return path


if __name__ == "__main__":
    import argparse
    import data
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=[*data.DATASETS, "dreyer_sim2"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    d = data.load_dreyer_sim2() if a.dataset == "dreyer_sim2" else data.load(a.dataset)
    for s in [int(x) for x in a.seeds.split(",")]:
        run(d, s, a.threads, epochs=100 if a.dataset == "dreyer_sim2" else 150)
