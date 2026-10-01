"""Experiment 05 (engineering): torch export of the per-person ts_C0.1 experts and the C3
combiner, checked against the research pipeline on R4–R6 (packaged seed-0 EEGNet experts)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from classical import _load  # noqa: E402
from combine import OUT, RelLogLinear, bal, load_bank, own  # noqa: E402
from torch_experts import C3Combiner, export  # noqa: E402

torch.set_num_threads(1)


def main():
    dev = load_bank("bank_xfit_seed0.npz")
    d = np.load(OUT / "classical_xfit_ts_C0.1.npz")
    dev["ts_C0.1"], dev["rel"] = d["logp"], d["rel_dev"][dev["fold"]][:, :, None]
    comb = RelLogLinear(["eegnet", "ts_C0.1", "rel"]).fit(own(dev, ["eegnet", "ts_C0.1", "rel"]),
                                                          dev["y"])
    w, b = comb.theta["w"].tolist(), float(comb.theta["b"][0])
    E, people = _load()
    ts_mod, _ = export(E, people, "ts_C0.1", [0, 1, 2], E["X"].shape[-1])
    c3 = C3Combiner(w[0], w[1], w[2], b, d["rel_test"])
    test = load_bank("bank_test_packaged.npz")
    X = torch.from_numpy(E["X"][E["run"] >= 3])
    with torch.no_grad():
        lp_ts = torch.cat([ts_mod(X[i:i + 256].double()) for i in range(0, len(X), 256)])
        lp_t = c3(torch.from_numpy(test["eegnet"]), lp_ts).numpy()            # (n, K, 2)
    test["ts_C0.1"] = np.load(OUT / "classical_test_ts_C0.1.npy")
    test["rel"] = np.broadcast_to(d["rel_test"][None, :, None], lp_t.shape[:2] + (1,)).copy()
    lp_r = comb.predict({k: test[k] for k in ["eegnet", "ts_C0.1", "rel"]})
    n, t, y = np.arange(len(test["y"])), test["true_idx"], test["y"]
    fp = np.exp(test["fp"])
    soft = lambda lp: (fp[:, :, None] * np.exp(lp)).sum(1)                    # noqa: E731
    res = {"coef": {"a": w[0], "c0": w[1], "c1": w[2], "b": b},
           "max_abs_dp_all_experts": float(np.abs(np.exp(lp_t) - np.exp(lp_r)).max()),
           "argmax_equal": float((lp_t.argmax(-1) == lp_r.argmax(-1)).mean()),
           "oracle_torch": bal(y, lp_t[n, t]), "oracle_research": bal(y, lp_r[n, t]),
           "soft_packaged_fp_torch": bal(y, np.log(soft(lp_t))),
           "ts_buffers_MB": sum(x.numel() * x.element_size() for x in ts_mod.buffers()) / 1e6}
    (HERE.parent / "results" / "export.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    torch.save({"ts": ts_mod.state_dict(), "c3": c3.state_dict()}, OUT / "c3_export_seed0.pt")


if __name__ == "__main__":
    main()
