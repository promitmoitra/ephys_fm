"""Build the loop B progress report (research/to_human/report-NNN.html) from result files."""

import json
import sys
from html import escape
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from combine import OUT, EqualLinear, LogLinearPool, RelLogLinear, bal, eval_global, load_bank, own  # noqa: E402

R = SRC.parent
EXP = R / "experiments"


def boot(d, rng):
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def dev_effects():
    """Gain over EEGNet alone on the cross-fitted dev bank, per person, with bootstrap CI."""
    b = load_bank("bank_xfit_seed0.npz")
    y, t, fold = b["y"], b["true_idx"], b["fold"]
    for v in ["ts", "ts_C0.1"]:
        d = np.load(OUT / f"classical_xfit_{v}.npz")
        b[v] = d["logp"]
        b[f"rel_{v}"] = d["rel_dev"][fold][:, :, None]
    e = own(b, ["eegnet"])["eegnet"]
    r0 = np.array([bal(y[t == k], e[t == k]) for k in range(21)])
    rng = np.random.default_rng(0)
    out = {}
    for name, mk, ex in [("Plain average", EqualLinear, ["eegnet", "ts"]),
                         ("C1 log-linear", LogLinearPool, ["eegnet", "ts"]),
                         ("C3 reliability-weighted", RelLogLinear,
                          ["eegnet", "ts_C0.1", "rel_ts_C0.1"])]:
        out[name] = boot(eval_global(b, mk, ex)["per_person"] - r0, rng)
    # per-person reliability vs held-out ts_C0.1 accuracy
    d = np.load(OUT / "classical_xfit_ts_C0.1.npz")
    ts = own({"ts": d["logp"], "true_idx": t, "y": y}, ["ts"])["ts"]
    acc = [bal(y[t == k], ts[t == k]) for k in range(21)]
    rel = d["rel_dev"].mean(0)
    return out, list(zip(rel.tolist(), acc, r0.tolist()))


def test_effects():
    c = json.loads((EXP / "checkpoint-2" / "results" / "confirm.json").read_text())
    m = {"Plain average": "R1", "C1 log-linear": "C1", "C3 reliability-weighted": "C3"}
    return {k: (c[v]["vs_R0_oracle"]["mean"], *c[v]["vs_R0_oracle"]["ci95"]) for k, v in m.items()}, c


def forest_svg(dev, test):
    """Dot + CI whisker per rule, two rows per rule (dev, test), one shared x axis."""
    rules = list(dev)
    W, left, right, row, top = 640, 190, 24, 26, 30
    lo, hi = -0.035, 0.035
    H = top + len(rules) * (2 * row + 14) + 34

    def x(v):
        return left + (v - lo) / (hi - lo) * (W - left - right)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f1t" class="chart">',
         '<title id="f1t">Gain over EEGNet alone, dev vs test, with 95% CIs</title>']
    for v in np.arange(-0.03, 0.031, 0.01):
        s.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 8}" y2="{H - 28}" '
                 f'class="{"zero" if abs(v) < 1e-9 else "grid"}"/>')
        s.append(f'<text x="{x(v):.1f}" y="{H - 10}" class="tick" text-anchor="middle">'
                 f'{v:+.2f}</text>')
    y = top
    for r in rules:
        s.append(f'<text x="0" y="{y + row - 4}" class="lab">{escape(r)}</text>')
        for i, (nm, vals, cls) in enumerate([("dev (cross-fitted R1–R3)", dev[r], "s1"),
                                              ("test (R4–R6)", test[r], "s2")]):
            yy = y + 8 + i * row
            m, a, b = vals
            s.append(f'<g class="pt"><title>{escape(r)} — {nm}: {m:+.3f} '
                     f'[{a:+.3f}, {b:+.3f}]</title>'
                     f'<rect x="{left}" y="{yy - 11}" width="{W - left - right}" height="22" '
                     f'fill="transparent"/>'
                     f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{yy}" y2="{yy}" class="{cls} ci"/>'
                     f'<circle cx="{x(m):.1f}" cy="{yy}" r="5" class="{cls} dot"/></g>')
        y += 2 * row + 14
    s.append("</svg>")
    return "\n".join(s)


def scatter_svg(pts):
    W, H, l, b_, t_, r_ = 640, 330, 52, 44, 16, 16

    def sx(v):
        return l + (v - 0.4) / 0.6 * (W - l - r_)

    def sy(v):
        return H - b_ - (v - 0.4) / 0.6 * (H - b_ - t_)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="f2t" class="chart">',
         '<title id="f2t">Per-person calibration reliability vs held-out ts accuracy</title>']
    for v in np.arange(0.4, 1.01, 0.1):
        s.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{t_}" y2="{H - b_}" class="grid"/>')
        s.append(f'<line x1="{l}" x2="{W - r_}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>')
        s.append(f'<text x="{sx(v):.1f}" y="{H - b_ + 16}" class="tick" text-anchor="middle">{v:.1f}</text>')
        s.append(f'<text x="{l - 8}" y="{sy(v) + 4:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    s.append(f'<line x1="{sx(0.4)}" y1="{sy(0.4)}" x2="{sx(1)}" y2="{sy(1)}" class="diag"/>')
    s.append(f'<text x="{(l + W - r_) / 2}" y="{H - 6}" class="axis" text-anchor="middle">'
             'reliability: run-to-run ts accuracy on calibration data</text>')
    s.append(f'<text transform="translate(14 {(H - b_) / 2}) rotate(-90)" class="axis" '
             'text-anchor="middle">held-out ts accuracy</text>')
    for i, (rel, acc, eeg) in enumerate(pts):
        s.append(f'<g class="pt"><title>person {i + 1}: reliability {rel:.2f}, held-out ts '
                 f'{acc:.2f}, EEGNet {eeg:.2f}</title><circle cx="{sx(rel):.1f}" '
                 f'cy="{sy(acc):.1f}" r="10" fill="transparent"/><circle cx="{sx(rel):.1f}" '
                 f'cy="{sy(acc):.1f}" r="5" class="s1 dot"/></g>')
    s.append("</svg>")
    return "\n".join(s)


CSS = """
:root{--surface:#fcfcfb;--text:#0b0b0b;--muted:#52514e;--grid:#e4e3df;--zero:#9a9893;
--s1:#2a78d6;--s2:#eb6834;--card:#ffffff;--border:#e4e3df}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--text:#fff;
--muted:#c3c2b7;--grid:#383835;--zero:#77766f;--s1:#3987e5;--s2:#d95926;--card:#222221;--border:#383835}}
:root[data-theme="dark"]{--surface:#1a1a19;--text:#fff;--muted:#c3c2b7;--grid:#383835;--zero:#77766f;
--s1:#3987e5;--s2:#d95926;--card:#222221;--border:#383835}
body{margin:0;background:var(--surface);color:var(--text);font:15px/1.55 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.15rem;margin:2em 0 .4em}
.sub{color:var(--muted)}.card{background:var(--card);border:1px solid var(--border);
border-radius:8px;padding:14px 16px;margin:12px 0}
.tiles{display:flex;gap:12px;flex-wrap:wrap}.tile{flex:1 1 150px;background:var(--card);
border:1px solid var(--border);border-radius:8px;padding:12px}
.tile b{display:block;font-size:1.6rem}.tile span{color:var(--muted);font-size:.85rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;
padding:5px 8px;border-bottom:1px solid var(--border)}td.n,th.n{text-align:right;
font-variant-numeric:tabular-nums}
.chart{width:100%;height:auto}.chart text{fill:var(--text)}.tick,.axis{fill:var(--muted)!important;
font-size:11px}.lab{font-size:13px}.grid{stroke:var(--grid);stroke-width:1}
.zero{stroke:var(--zero);stroke-width:1.5}.diag{stroke:var(--zero);stroke-dasharray:4 4}
.ci{stroke-width:2;stroke-linecap:round}.s1.ci{stroke:var(--s1)}.s2.ci{stroke:var(--s2)}
.s1.dot{fill:var(--s1);stroke:var(--surface);stroke-width:2}.s2.dot{fill:var(--s2);
stroke:var(--surface);stroke-width:2}.pt:hover .dot{r:7}
.legend{display:flex;gap:16px;font-size:.85rem;color:var(--muted)}.sw{display:inline-block;
width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
details{margin:6px 0}code{font-size:.85em}
"""


def main():
    dev, pts = dev_effects()
    test, c2 = test_effects()
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{dev[k][0]:+.3f} [{dev[k][1]:+.3f}, {dev[k][2]:+.3f}]</td>"
        f"<td class=n>{test[k][0]:+.3f} [{test[k][1]:+.3f}, {test[k][2]:+.3f}]</td></tr>"
        for k in dev)
    srows = "".join(f"<tr><td>{i + 1}</td><td class=n>{r:.2f}</td><td class=n>{a:.2f}</td>"
                    f"<td class=n>{e:.2f}</td></tr>" for i, (r, a, e) in enumerate(pts))
    ctab = "".join(
        f"<tr><td>{cid}</td><td>{escape(r['desc'])}</td><td class=n>{r['oracle']['bal_acc']:.3f}</td>"
        f"<td class=n>{r['oracle']['nll']:.3f}</td><td class=n>"
        f"{r.get('soft_fp_loopA', r['soft_fp_packaged'])['bal_acc']:.3f}</td></tr>"
        for cid, r in c2.items())
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loop B Progress</title><style>{CSS}</style></head><body><main>
<p class="sub">Track 2 · autoresearch loop B · report 001 · 2026-09-30</p>
<h1>Learned weights for per-person experts</h1>
<p>Each evaluation person has a fine-tuned EEGNet expert and a Riemannian tangent-space (ts)
expert. The question: which combination rule, learned on calibration data only, is best on the
hidden runs? All numbers are on the Dreyer sealed-phase simulation with oracle person ID,
unless marked soft-routed.</p>
<div class="tiles">
<div class="tile"><span>EEGNet alone, R4–R6</span><b>0.908</b><span>NLL 0.235</span></div>
<div class="tile"><span>Adopted rule C3, R4–R6</span><b>0.921</b><span>NLL 0.209</span></div>
<div class="tile"><span>C3 soft-routed with loop A's fingerprint</span><b>{c2['C3'].get('soft_fp_loopA', c2['C3']['soft_fp_packaged'])['bal_acc']:.3f}</b><span>EEGNet alone 0.907</span></div>
</div>

<h2>1 · The plain average was the wrong rule; log-space pooling is robust</h2>
<p>Gain over EEGNet alone (mean per-person balanced accuracy, person-bootstrap 95% CI). The
dev bank has every calibration run held out in turn (2,520 windows). The plain average swings
with the split. The log-space rules have a non-negative point estimate on every split, and on the test runs their CI clears zero.</p>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>dev (cross-fitted R1–R3)</span>
<span><i class="sw" style="background:var(--s2)"></i>test (R4–R6)</span></div>
<div class="card">{forest_svg(dev, test)}</div>
<details><summary>Table view</summary><table><tr><th>Rule</th><th class=n>Dev Δ [CI]</th>
<th class=n>Test Δ [CI]</th></tr>{rows}</table></details>

<h2>2 · Why: the ts expert is over-confident, and whether it works is a person trait</h2>
<p>ts is near-perfect for about 6 of 21 people and at chance for about 9. It is equally
confident either way (mean confidence 0.755 at 0.680 accuracy; EEGNet 0.883 at 0.887). A plain
average therefore lets chance-level ts votes overrule EEGNet. A person's ts accuracy from one
calibration run to another predicts their held-out ts accuracy almost perfectly. The adopted
rule weights ts by that reliability: weight ≈ 0.1 at chance, ≈ 1.0 at 0.9.</p>
<div class="card">{scatter_svg(pts)}</div>
<details><summary>Table view (per person)</summary><table><tr><th>Person</th>
<th class=n>Reliability</th><th class=n>Held-out ts</th><th class=n>EEGNet</th></tr>{srows}</table></details>

<h2>3 · Checkpoint 2 on the hidden runs</h2>
<p>Combiners fitted on the dev bank, applied unchanged. The candidates and the adoption rule
were committed before this ran.</p>
<table><tr><th>ID</th><th>Rule</th><th class=n>Oracle acc</th><th class=n>NLL</th>
<th class=n>Soft (loop A fp)</th></tr>{ctab}</table>

<h2>What I learned about the method</h2>
<ul><li>A 40-window-per-person dev set (R3 only) gave the <em>wrong sign</em> for the ts effect.
Cross-fitting all calibration runs tripled the dev data; even then, rules differ by less than
the ±0.011 CI. The consistent signal is the NLL gain of log-space pooling.</li>
<li>Filter banks, other windows and CSP don't make a better classical expert. Regularising it
(C = 0.1) does, because it fixes calibration.</li></ul>

<h2>Next</h2>
<ul><li><b>Running:</b> seed robustness. EEGNet experts for training seeds 1 and 2, with
the same combiners applied unchanged (protocol 04).</li>
<li>Then: an integration plan for <code>submission.py</code> (per-person tangent-space + LR as
torch tensors, like loop A's fingerprint export), and a PR.</li></ul>
<p class="sub">Source: <code>research/</code> in worktree <code>t2-expert-weights</code>, branch
<code>exp/expert-weights</code>. Protocols were committed before their results.</p>
</main></body></html>"""
    out = R / "to_human" / "report-001.html"
    out.write_text(html)
    print(out)


if __name__ == "__main__":
    main()
