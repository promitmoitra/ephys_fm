"""Protocol 03, step 2: score every classical variant alone and combined with EEGNet on the
cross-fitted bank (true ID, LOPO for the global combiners)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from classical import VARIANTS  # noqa: E402
from combine import (OUT, EqualLinear, LogLinearPool, RelLogLinear, bal,  # noqa: E402
                     eval_global, load_bank, nll, own)

torch.set_num_threads(1)


def main():
    bank = load_bank("bank_xfit_seed0.npz")
    y, t, fold = bank["y"], bank["true_idx"], bank["fold"]
    K = len(bank["people"])
    base_pp = np.array([bal(y[t == k], own(bank, ["eegnet"])["eegnet"][t == k])
                        for k in range(K)])
    res = {"R0": {"bal_acc": bal(y, own(bank, ["eegnet"])["eegnet"]),
                  "nll": nll(own(bank, ["eegnet"])["eegnet"], y)}}
    for v in VARIANTS:
        d = np.load(OUT / f"classical_xfit_{v}.npz")
        bank[v] = d["logp"]
        assert np.allclose(bank["ts"], bank[v]) if v == "ts" else True, "ts mismatch"
        rel = d["rel_dev"][fold]                                    # (n, K)
        bank[f"rel_{v}"] = rel[:, :, None]
        alone = own(bank, [v])[v]
        row = {"alone": {"bal_acc": bal(y, alone), "nll": nll(alone, y),
                         "conf": float(np.exp(alone).max(1).mean())}}
        for name, make, ex in [("R1", EqualLinear, ["eegnet", v]),
                               ("H3", lambda e: LogLinearPool(e, bias=True), ["eegnet", v]),
                               ("H8", RelLogLinear, ["eegnet", v, f"rel_{v}"])]:
            r = eval_global(bank, make, ex)
            d_pp = r["per_person"] - base_pp
            row[name] = {"bal_acc": r["bal_acc"], "nll": r["nll"],
                         "vs_R0": [float(d_pp.mean()), int((d_pp > 0).sum()),
                                   int((d_pp < 0).sum())],
                         "per_fold": [bal(y[fold == f], r["oof"][fold == f]) for f in range(3)],
                         "params": make(ex).fit(own(bank, ex), y).describe()}
        own_rel = rel[np.arange(len(y)), t]
        acc_pp = np.array([bal(y[t == k], alone[t == k]) for k in range(K)])
        row["rel_corr"] = float(np.corrcoef(
            [own_rel[t == k].mean() for k in range(K)], acc_pp)[0, 1])
        res[v] = row
    lines = ["# Protocol 03 results: classical variants on the cross-fitted bank (true ID)\n",
             f"EEGNet alone (R0): {res['R0']['bal_acc']:.3f} / NLL {res['R0']['nll']:.3f}. "
             "Combined rows: LOPO; Δ vs R0 = mean per-person difference, people better / worse.\n",
             "| Variant | Alone acc / NLL / mean conf | R1 equal avg | H3 log-linear+bias | "
             "H8 reliability-weighted | corr(rel, held-out acc) |",
             "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        r = res[v]

        def cell(k):
            c = r[k]
            return (f"{c['bal_acc']:.3f} / {c['nll']:.3f} ({c['vs_R0'][0]:+.3f}, "
                    f"{c['vs_R0'][1]}/{c['vs_R0'][2]})")
        a = r["alone"]
        lines.append(f"| {v} | {a['bal_acc']:.3f} / {a['nll']:.3f} / {a['conf']:.3f} | "
                     f"{cell('R1')} | {cell('H3')} | {cell('H8')} | {r['rel_corr']:.2f} |")
    lines.append("\nParams (full fit): " + "; ".join(
        f"{v}: H3 `{json.dumps(res[v]['H3']['params'])}`, H8 `{json.dumps(res[v]['H8']['params'])}`"
        for v in VARIANTS))
    out = HERE.parent / "results"
    (out / "variants.json").write_text(json.dumps(res, indent=2))
    (out / "variants.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
