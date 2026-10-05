"""Train the page solvers on the warm-up train split (val for epochs) and write their folders.

    python track2/bench/train_bench.py --threads 4
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "external" / "2026-competition"), str(REPO / "track2"),
                str(REPO / "research" / "expert-portfolio" / "src")]
KIT_OUT = REPO / "external" / "2026-competition" / "tracks" / "bci_decoding" / "outputs"


def folder(name, ch_names, extra=None):
    d = KIT_OUT / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "track2" / "submission.py", d / "submission_lib.py")
    (d / "config.json").write_text(json.dumps({"ch_names": ch_names, **(extra or {})}))
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-reve", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from train_mixture import fit, load_windows
    from models import make_model
    import riemann_parts
    d = load_windows()
    X, y, split = d["X"], d["y"], d["split"]
    chs, sf = [str(c) for c in d["ch_names"]], float(d["sfreq"])
    tr, va = split == "train", split == "val"            # no evaluation-people labels
    C, (n_chans, n_times) = int(y.max()) + 1, X.shape[1:]
    net, _ = fit(make_model("shallow", n_chans, C, n_times, sf), X[tr], y[tr], lr=1e-3,
                 epochs=100, seed=0, bs=64, X_val=X[va], y_val=y[va], name="shallow", log_every=10)
    torch.save(net.state_dict(), folder("ShallowFBCSPNet-pooled", chs) / "model.pt")
    st = riemann_parts.fit_fingerprint(X[tr], y[tr], sf)
    torch.save(st, folder("Riemann-FB-pooled", chs, {"n_bands": len(riemann_parts.FP_BANDS)}) / "model.pt")
    if not a.skip_reve:
        import reve_parts
        from submission import load_reve_encoder
        pos_dir = REPO / "outputs" / "t2-portfolio" / "reve_positions"
        e = np.load(REPO / "outputs" / "t2-portfolio" / "reve_emb_dreyer.npz")["full"]
        mu, sd = e[tr].mean(0), e[tr].std(0) + 1e-6
        W, b = reve_parts.fit_head((e[tr] - mu) / sd, y[tr], C, 1e-3)
        st = reve_parts.export(reve_parts.resample_matrix(n_times, sf),
                               reve_parts.positions(chs, pos_dir / "reve_positions.json"),
                               mu, sd, W[None], b[None])
        dd = folder("REVE-probe-pooled", chs, {"n_out": 800})
        torch.save(st, dd / "model.pt")
        for f in ("reve_positions.json", "reve_kwargs.json"):
            shutil.copyfile(pos_dir / f, dd / f)


if __name__ == "__main__":
    main()
