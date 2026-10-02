"""Train and package the Track 2 fingerprint-mixture submission on Dreyer 2023.

Simulates the sealed phase's structure on the warm-up study. Sealed: 10
training participants (all sessions labeled), and 10 evaluation participants
whose sessions 1-3 are labeled calibration and whose sessions 4-6 are hidden.
Dreyer has one session of six runs per subject, so:

    training pool      the kit's train split (subjects 1-60, 82-87 minus val)
    validation         the kit's val split (pooled-model epoch selection)
    evaluation people  the kit's test subjects 61-81 (21 people)
      calibration      their runs R1-R3 (labeled, used for training)
      hidden test      their runs R4-R6 (scored here)

Pipeline (track2/README.md, "Current best model"):
  1. pooled EEGNet on training pool + all calibration runs
  2. fingerprint: filter-bank covariance model on all calibration runs
     (loop A; default), or the older EEGNet fingerprint (21-way, R1-R2,
     epoch picked on R3)
  3. one whole-network fine-tune of the pooled model per evaluation person on
     their R1-R3 (lr 1e-4, 50 epochs, last epoch)
  4. per-person Riemannian experts on R1-R3 and their calibration reliability,
     combined with the EEGNet experts in log space (loop B, combiner C3)
  5. score on R4-R6 through submission.py's own code path, and package

--reuse-eegnet PATH skips steps 1 and 3 and loads the EEGNet experts from an
existing mixture.pt (e.g. the seed-0 package), so a new fingerprint or
combiner can be packaged and scored against the same experts.

The packaged ZIP uses test subjects' labeled runs, so it must NOT be uploaded
to the warm-up leaderboard (inflated, leaky score). It exists to validate
the design and the submission contract.

Usage (repo root, venv active; Dreyer prepared under data/neural_compet):
    python track2/train_mixture.py [--seed 0] [--epochs 100]
    python track2/train_mixture.py --reuse-eegnet PATH/mixture.pt --out DIR
"""

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "external" / "2026-competition"
DATA = REPO / "data"
sys.path[:0] = [str(KIT), str(Path(__file__).resolve().parent)]

from submission import build_model  # noqa: E402  (the shipped code path)
import riemann_parts  # noqa: E402

CACHE = DATA / "experiments" / "dreyer_windows.npz"
OUT = REPO / "outputs" / "track2_dreyer_sim"
RESULTS = Path(__file__).resolve().parent / "results"
N_CALIB_RUNS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bal_acc(y, p):
    return float(balanced_accuracy_score(y, p))


# --------------------------------------------------------------------------
# Data: the kit's exact Dreyer pipeline, materialized once with run labels
# --------------------------------------------------------------------------

def load_windows():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    from torch.utils.data import DataLoader
    from benchmark_utils.nb_task import load_task

    loaders, meta = load_task(
        "eeg", "motor_imagery", data_dir=DATA / "neural_compet",
        dataset="dreyer2023", target_transform=lambda y: y.argmax(-1))
    parts = {k: [] for k in ("X", "y", "subject", "run", "split")}
    for split in ("train", "val", "test"):
        ds = loaders[split].dataset
        segs = ds.seg_ds.segments
        for X, y, _ in DataLoader(ds, batch_size=256, shuffle=False,
                                  collate_fn=None):
            parts["X"].append(X.numpy())
            parts["y"].append(y.numpy())
        # timeline "Dreyer2023Large:run=0R1acquisition,session=0,subject=61"
        tl = [dict(kv.split("=") for kv in s.timeline.split(":", 1)[1].split(","))
              for s in segs]
        parts["subject"].append([t["subject"] for t in tl])
        parts["run"].append([int(t["run"][0]) for t in tl])  # "0R1..." -> 0
        parts["split"].append([split] * len(segs))
        log(f"  {split}: {len(segs)} windows")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["ch_names"] = np.asarray(meta["ch_names"])
    out["sfreq"] = np.asarray(meta["sfreq"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def make_eegnet(n_chans, n_outputs, n_times):
    from models import make_model
    return make_model("eegnet", n_chans, n_outputs, n_times, sfreq=120.0)


def logits(model, X, bs=512):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(torch.from_numpy(X[i:i + bs]))
                          for i in range(0, len(X), bs)])


def fit(model, X, y, *, lr, epochs, seed, bs, X_val=None, y_val=None,
        name="", log_every=10):
    """AdamW + cross-entropy. With validation data, keep the best epoch;
    otherwise keep the last."""
    import copy
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    best, best_state, best_epoch = -1.0, None, epochs - 1
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(bs):
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        if X_val is not None:
            score = bal_acc(y_val, logits(model, X_val).argmax(1).numpy())
            if score > best:
                best, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
            if log_every and (epoch + 1) % log_every == 0:
                log(f"  {name} epoch {epoch + 1}/{epochs}: val {score:.3f} "
                    f"(best {best:.3f} @ {best_epoch + 1}), "
                    f"{(time.time() - t0) / (epoch + 1):.1f} s/epoch")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val": best, "best_epoch": best_epoch + 1}


def masks(d, eval_people=None):
    """Training / calibration / hidden masks.

    Default (the original simulation): the kit's test people are the evaluation people.
    sim2 (`eval_people` given): those people are the evaluation people, and every other
    train- or test-split person is an ordinary training person (all six runs)."""
    subj, run, split = d["subject"], d["run"], d["split"]
    if eval_people is None:
        is_eval = split == "test"
        train = split == "train"
    else:
        is_eval = np.isin(subj, list(eval_people))
        train = np.isin(split, ["train", "test"]) & ~is_eval
    calib = is_eval & (run < N_CALIB_RUNS)
    hidden = is_eval & (run >= N_CALIB_RUNS)
    people = sorted(np.unique(subj[is_eval]), key=int)
    return train, calib, hidden, people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100, help="pooled model")
    ap.add_argument("--fp-epochs", type=int, default=150, help="EEGNet fingerprint")
    ap.add_argument("--ft-epochs", type=int, default=50, help="per-person")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fingerprint", choices=["fb_riemann", "eegnet"], default="fb_riemann")
    ap.add_argument("--no-riemann-experts", action="store_true")
    ap.add_argument("--reuse-eegnet", type=Path, default=None,
                    help="mixture.pt whose EEGNet experts (and EEGNet fingerprint) to reuse")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--eval-people", default=None, help="JSON with a 'people' list (sim2)")
    ap.add_argument("--shallow-experts", action="store_true")
    ap.add_argument("--reve-probe", action="store_true")
    ap.add_argument("--reve-lam", type=float, default=0.1)
    ap.add_argument("--reve-emb-cache", default=None,
                    help="outputs/t2-portfolio/reve_emb_dreyer.npz")
    ap.add_argument("--combiner", choices=["C3", "portfolio"], default="C3")
    ap.add_argument("--portfolio-coef", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_start = time.time()

    log("loading windows")
    d = load_windows()
    X, y, subj, run, split = d["X"], d["y"], d["subject"], d["run"], d["split"]
    ch_names = [str(c) for c in d["ch_names"]]
    sfreq = float(d["sfreq"])
    n_chans, n_times = X.shape[1:]
    n_classes = int(y.max()) + 1
    eval_people = (json.loads(Path(args.eval_people).read_text())["people"]
                   if args.eval_people else None)
    if eval_people is not None and args.reuse_eegnet:
        raise SystemExit("--reuse-eegnet experts belong to the original evaluation people")
    if eval_people is not None and args.out == OUT:
        args.out = OUT / "sim2"          # never overwrite the original simulation's package
    train_mask, calib, hidden, people = masks(d, eval_people)
    K = len(people)
    s_idx = {s: i for i, s in enumerate(people)}
    lab = np.array([s_idx.get(s, -1) for s in subj])
    log(f"X {X.shape}, {n_classes} classes, {K} evaluation people; windows: "
        f"pool {int(train_mask.sum())} + calib {int(calib.sum())}, "
        f"val {int((split == 'val').sum())}, hidden test {int(hidden.sum())}")

    import copy
    pooled = control = info_pooled = reused = None
    if args.reuse_eegnet:
        reused = torch.load(args.reuse_eegnet, map_location="cpu", weights_only=True)
        experts = []
        for sd in reused["experts"]:
            e = make_eegnet(n_chans, n_classes, n_times)
            e.load_state_dict(sd)
            experts.append(e)
        log(f"reusing {len(experts)} EEGNet experts from {args.reuse_eegnet}")
    else:
        # 1. pooled model: training pool + calibration runs
        pool = train_mask | calib
        val = split == "val"
        log("pooled EEGNet")
        pooled, info_pooled = fit(
            make_eegnet(n_chans, n_classes, n_times), X[pool], y[pool], lr=1e-3,
            epochs=args.epochs, seed=args.seed, bs=64, X_val=X[val], y_val=y[val],
            name="pooled")
        # 3. per-person fine-tunes, plus the epoch-matched control
        log(f"per-person fine-tunes ({K}) and control")
        control, _ = fit(copy.deepcopy(pooled), X[calib], y[calib], lr=1e-4,
                         epochs=args.ft_epochs, seed=args.seed, bs=32)
        experts = []
        for s in people:
            m = calib & (subj == s)
            expert, _ = fit(copy.deepcopy(pooled), X[m], y[m], lr=1e-4,
                            epochs=args.ft_epochs, seed=args.seed, bs=32)
            experts.append(expert)

    # 2. fingerprint: who is this (among the evaluation people)?
    if args.fingerprint == "fb_riemann":
        log("fingerprint: filter-bank covariance model on all calibration runs")
        fp_state = riemann_parts.fit_fingerprint(X[calib], lab[calib], sfreq)
        info_fp = {"kind": "fb_riemann", "bands": riemann_parts.FP_BANDS}
    elif reused is not None:
        fp_state = reused["fingerprint"]
        info_fp = {"kind": "eegnet", "reused": str(args.reuse_eegnet)}
    else:
        log("fingerprint EEGNet")
        fp_tr = calib & (run < N_CALIB_RUNS - 1)
        fp_va = calib & (run == N_CALIB_RUNS - 1)
        fingerprint, info_fp = fit(
            make_eegnet(n_chans, K, n_times), X[fp_tr], lab[fp_tr], lr=1e-3,
            epochs=args.fp_epochs, seed=args.seed, bs=64, X_val=X[fp_va],
            y_val=lab[fp_va], name="fingerprint", log_every=25)
        fp_state = fingerprint.state_dict()
        info_fp["kind"] = "eegnet"

    # 4. per-person Riemannian experts + combiner
    state = {"fingerprint": fp_state, "experts": [e.state_dict() for e in experts]}
    config_extra = {"fingerprint": args.fingerprint,
                    "fingerprint_n_bands": len(riemann_parts.FP_BANDS),
                    "riemann_experts": not args.no_riemann_experts}
    rel = None
    if not args.no_riemann_experts:
        log("per-person Riemannian experts and calibration reliability")
        state["riemann"], state["combiner"], rel, n_out = riemann_parts.fit_riemann(
            X[calib], y[calib], lab[calib], run[calib], K, n_classes, sfreq)
        config_extra |= {"riemann_n_times_out": int(n_out), "combiner": "C3",
                         "combiner_coef": riemann_parts.C3}

    # 5. extra expert streams (loop C) and the portfolio combiner
    extra_files = []
    if args.shallow_experts:
        log("ShallowFBCSPNet pooled + per-person fine-tunes")
        sys.path.insert(0, str(REPO / "research" / "expert-portfolio" / "src"))
        from stream_bank import train_stream
        _, sh, _ = train_stream(X, y, train_mask | calib, split == "val",
                                [calib & (subj == s) for s in people], sfreq, args.seed,
                                args.threads, epochs=args.epochs, ft_epochs=args.ft_epochs)
        state["shallow_experts"] = [e.state_dict() for e in sh]
        config_extra["shallow_experts"] = True
    if args.reve_probe:
        log("REVE probe heads")
        import reve_parts
        from submission import load_reve_encoder
        pos_dir = REPO / "outputs" / "t2-portfolio" / "reve_positions"
        R = reve_parts.resample_matrix(n_times, sfreq)
        pos = reve_parts.positions(ch_names, pos_dir / "reve_positions.json")
        if args.reve_emb_cache:
            Z = np.load(args.reve_emb_cache)["full"]
            assert len(Z) == len(X), "embedding cache does not match the windows"
        else:
            Z = reve_parts.embed(load_reve_encoder(pos_dir), X, R, pos)
        pool = train_mask | calib
        mu, sd = Z[pool].mean(0), Z[pool].std(0) + 1e-6
        W0, b0 = reve_parts.fit_head((Z[pool] - mu) / sd, y[pool], n_classes, 1e-3)
        W, b = reve_parts.fit_person_heads((Z[calib] - mu) / sd, y[calib], lab[calib], K,
                                           n_classes, W0, b0, args.reve_lam)
        state["reve"] = reve_parts.export(R, pos, mu, sd, W, b)
        config_extra |= {"reve_probe": True, "reve_n_out": int(R.shape[1])}
        extra_files += [pos_dir / "reve_positions.json", pos_dir / "reve_kwargs.json"]
    if args.combiner == "portfolio":
        if args.no_riemann_experts:
            raise SystemExit("the portfolio combiner needs the Riemannian experts")
        coef = json.loads(Path(args.portfolio_coef).read_text())
        n_streams = 1 + int(args.shallow_experts) + int(args.reve_probe)
        assert len(coef["w"]) == n_streams, "coefficients don't match the enabled streams"
        b_full = [0.0, *coef["b"]] if n_classes == 2 else [0.0] * n_classes
        state["combiner"] = {"w": torch.tensor(coef["w"], dtype=torch.float64),
                             "c": torch.tensor(coef["c"], dtype=torch.float64),
                             "class_bias": torch.tensor(b_full, dtype=torch.float64),
                             "rel": state["combiner"]["rel"]}
        config_extra |= {"combiner": "portfolio", "combiner_coef": coef}

    # Package: the exact files a Codabench upload would contain
    sub_dir = args.out / "submission"
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    sub_dir.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parent / "submission.py",
                    sub_dir / "submission.py")
    torch.save(state, sub_dir / "mixture.pt")
    config = {"ch_names": ch_names, "n_times": int(n_times),
              "n_classes": n_classes, "sfreq": sfreq,
              "experts": [f"Dreyer2023Large/{s}" for s in people],
              **config_extra,
              "trained_on": "Dreyer 2023 sealed-phase simulation (NOT for warm-up upload)",
              "seed": args.seed,
              "eegnet_experts": str(args.reuse_eegnet) if args.reuse_eegnet else "trained"}
    (sub_dir / "config.json").write_text(json.dumps(config, indent=2))
    for f in extra_files:
        shutil.copyfile(f, sub_dir / f.name)
    zip_path = args.out / "track2_dreyer_sim.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(sub_dir.iterdir()):
            z.write(f, arcname=f.name)  # files at the ZIP root
    log(f"packaged {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")

    # Score on the hidden runs, reloading from the packaged files
    meta = {"ch_names": ch_names, "n_times": int(n_times), "n_classes": n_classes,
            "device": "cpu", "sfreq": sfreq, "submission_dir": sub_dir}
    state = torch.load(sub_dir / "mixture.pt", map_location="cpu", weights_only=True)
    mixture = build_model(meta, json.loads((sub_dir / "config.json").read_text()), state)
    Xh, yh, sh = X[hidden], y[hidden], subj[hidden]
    t0 = time.time()
    p_mix = torch.cat([mixture.predict(torch.from_numpy(Xh[i:i + 256]))
                       for i in range(0, len(Xh), 256)]).numpy()
    infer_s = time.time() - t0
    with torch.inference_mode():
        Xt = torch.from_numpy(Xh)
        p_subj = torch.softmax(mixture.fingerprint(Xt), 1).double()
        lp_eeg = torch.stack([torch.log_softmax(e(Xt), 1) for e in mixture.experts],
                             1).double()
        lp_all = torch.cat([mixture.expert_logp(Xt[i:i + 256])
                            for i in range(0, len(Xt), 256)]).double()
    n = torch.arange(len(Xh))
    true_idx = torch.as_tensor([s_idx[s] for s in sh])

    def soft(lp):
        return (p_subj[:, :, None] * lp.exp()).sum(1).argmax(1).numpy()

    preds = {}
    if pooled is not None:
        preds["pooled"] = logits(pooled, Xh).argmax(1).numpy()
        preds["control"] = logits(control, Xh).argmax(1).numpy()
    preds |= {"EEGNet experts, oracle": lp_eeg[n, true_idx].argmax(1).numpy(),
              "EEGNet experts, soft": soft(lp_eeg)}
    if args.combiner == "portfolio":
        from submission import LogLinearCombiner
        c3 = LogLinearCombiner(K, n_classes)
        c3.coef.copy_(torch.tensor([riemann_parts.C3["a"], riemann_parts.C3["c0"],
                                    riemann_parts.C3["c1"]], dtype=torch.float64))
        if n_classes == 2:
            c3.class_bias[1] = riemann_parts.C3["b"]
        c3.rel.copy_(mixture.combiner.rel)
        with torch.inference_mode():
            lp_c3 = torch.cat([c3(lp_eeg[i:i + 256], mixture.riemann(Xt[i:i + 256]))
                               for i in range(0, len(Xt), 256)]).double()
        preds |= {"E+T (C3), oracle": lp_c3[n, true_idx].argmax(1).numpy(),
                  "E+T (C3), soft": soft(lp_c3),
                  "full combination, oracle": lp_all[n, true_idx].argmax(1).numpy(),
                  "full combination, soft": soft(lp_all)}
    elif not args.no_riemann_experts:
        preds |= {"+ Riemannian (C3), oracle": lp_all[n, true_idx].argmax(1).numpy(),
                  "+ Riemannian (C3), soft": soft(lp_all)}
    preds["shipped mixture (predict)"] = p_mix
    fp_acc = bal_acc(true_idx.numpy(), p_subj.argmax(1).numpy())
    scores = {}
    for name, p in preds.items():
        per = [bal_acc(yh[sh == s], p[sh == s]) for s in people]
        scores[name] = {"bal_acc": bal_acc(yh, p),
                        "mean_over_people": float(np.mean(per)),
                        "per_person": dict(zip(people, map(float, per)))}
    result = {"args": {k: str(v) for k, v in vars(args).items()},
              "eval_people": people,
              "fingerprint_bal_acc": fp_acc,
              "pooled": info_pooled, "fingerprint": info_fp,
              "riemann_reliability": None if rel is None else dict(zip(people, rel.tolist())),
              "inference_s_hidden_windows": round(infer_s, 2),
              "n_hidden_windows": int(len(Xh)), "scores": scores,
              "zip_mb": round(zip_path.stat().st_size / 1e6, 3),
              "runtime_s": round(time.time() - t_start, 1)}
    RESULTS.mkdir(exist_ok=True)
    tag = (f"dreyer_sim_seed{args.seed}_{args.fingerprint}"
           f"{'' if args.no_riemann_experts else '_c3'}"
           f"{'_reused' if args.reuse_eegnet else ''}"
           f"{'_sim2' if args.eval_people else ''}{'_shallow' if args.shallow_experts else ''}"
           f"{'_reve' if args.reve_probe else ''}"
           f"{'_portfolio' if args.combiner == 'portfolio' else ''}")
    (RESULTS / f"{tag}.json").write_text(json.dumps(result, indent=2))
    (RESULTS / f"{tag}.md").write_text(render(result, K))
    print(render(result, K), flush=True)


def render(r, K):
    a = r["args"]
    experts = ("reused from `" + a["reuse_eegnet"] + "`" if a["reuse_eegnet"] != "None"
               else "trained")
    riemann = "off" if a["no_riemann_experts"] == "True" else f"on (combiner {a['combiner']})"
    streams = [x for x, f in (("ShallowFBCSPNet", "shallow_experts"), ("REVE probe", "reve_probe"))
               if a.get(f) == "True"]
    who = ("sim2: " + ", ".join(r["eval_people"]) if a.get("eval_people") not in (None, "None")
           else "subjects 61–81")
    lines = ["# Track 2: fingerprint mixture on Dreyer 2023 (sealed-phase simulation)\n",
             f"{K} evaluation people ({who}); calibration = runs R1–R3, "
             f"hidden test = R4–R6 ({r['n_hidden_windows']} windows). 2-class MI, "
             f"chance 0.5. Seed {a['seed']}. Fingerprint `{a['fingerprint']}`: "
             f"{r['fingerprint_bal_acc']:.3f} {K}-way on the hidden runs (chance "
             f"{1 / K:.3f}). Riemannian experts {riemann}. EEGNet experts {experts}. "
             f"Extra streams: {', '.join(streams) or 'none'}.\n",
             "| Model | Balanced acc (windows) | Mean over people |", "|---|---|---|"]
    for name, s in r["scores"].items():
        lines.append(f"| {name} | {s['bal_acc']:.3f} | {s['mean_over_people']:.3f} |")
    lines += ["", "`soft` rows mix the per-person experts by the packaged fingerprint's "
              "p(person | window); `oracle` rows use the true person. The last row runs "
              "the packaged `submission.py` + `mixture.pt` + `config.json` through "
              f"`predict`. Inference on {r['n_hidden_windows']} windows (CPU): "
              f"{r['inference_s_hidden_windows']} s. ZIP: {r['zip_mb']} MB. "
              f"Runtime {r['runtime_s']} s."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
