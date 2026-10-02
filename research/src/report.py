"""Build research/to_human/progress-<n>.html from the saved result JSONs.

    python research/src/report.py 001
"""

import html
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
EXPS = R / "experiments"

SERIES = [("eegnet", "EEGNet (shipped recipe, last epoch)"),
          ("ts_broad", "Riemann, broadband"),
          ("ts_fb", "Riemann, 6-band filter bank")]


def res(exp, name):
    p = EXPS / exp / "results" / f"{name}.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def agg(exp, names):
    ms = [m for m in (res(exp, n) for n in names) if m]
    if not ms:
        return None
    return {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])), len(ms))
            for k in ("bal_acc", "nll", "p_true", "ece")}


def dev_rows():
    """(benchmark label, {series: agg})"""
    seeds = lambda pre: [f"{pre}_seed{s}" for s in (0, 1, 2)]
    return [
        ("Dreyer R1–R2 → R3 (locked inner loop)", {
            "eegnet": agg("h0-eegnet-baseline", seeds("eegnet")),
            "ts_broad": agg("h3-riemann-fingerprint", ["ts_broad_C1"]),
            "ts_fb": agg("h3-riemann-fingerprint", ["ts_fb_C1"])}),
        ("Dreyer R1 → R2 (one training run)", {
            "eegnet": agg("h3.1-drift-benchmarks", seeds("D1_R1toR2_eegnet")),
            "ts_broad": agg("h3.1-drift-benchmarks", ["D1_R1toR2_ts_broad_C1"]),
            "ts_fb": agg("h3.1-drift-benchmarks", ["D1_R1toR2_ts_fb_C1"])}),
        ("Dreyer R1 → R3 (two runs later)", {
            "eegnet": agg("h3.1-drift-benchmarks", seeds("D1_R1toR3_eegnet")),
            "ts_broad": agg("h3.1-drift-benchmarks", ["D1_R1toR3_ts_broad_C1"]),
            "ts_fb": agg("h3.1-drift-benchmarks", ["D1_R1toR3_ts_fb_C1"])}),
        ("BNCI day 1 → day 2 (cross-day, 9 people)", {
            "eegnet": agg("h3.1-drift-benchmarks", seeds("D2_bnci_s1tos2_eegnet")),
            "ts_broad": agg("h3.1-drift-benchmarks", ["D2_bnci_s1tos2_ts_broad_C1"]),
            "ts_fb": agg("h3.1-drift-benchmarks", ["D2_bnci_s1tos2_ts_fb_C1"])}),
    ]


def dot_plot(rows, key, lo, hi, label):
    """Cleveland dot plot: one row per benchmark, one dot per series."""
    W, row_h, left, right, top = 720, 44, 290, 30, 30
    H = top + row_h * len(rows) + 30
    x = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(label)}" class="chart">']
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 8}" y2="{H - 26}" class="grid"/>'
                   f'<text x="{x(v):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{v:.2f}</text>')
    for r, (bench, vals) in enumerate(rows):
        y = top + r * row_h + row_h / 2
        out.append(f'<text x="{left - 12}" y="{y + 4:.1f}" class="rowlab" text-anchor="end">'
                   f'{html.escape(bench)}</text>')
        for s, (sid, sname) in enumerate(SERIES):
            a = vals.get(sid)
            if not a:
                continue
            m, sd, n = a[key]
            v = min(max(m, lo), hi)
            dy = (s - 1) * 9
            if n > 1 and sd > 0:
                out.append(f'<line x1="{x(max(m - sd, lo)):.1f}" x2="{x(min(m + sd, hi)):.1f}" '
                           f'y1="{y + dy:.1f}" y2="{y + dy:.1f}" class="err s{s + 1}"/>')
            tip = f"{sname} · {bench}: {key} {m:.3f}" + (f" ± {sd:.3f} (n={n})" if n > 1 else "")
            out.append(f'<circle cx="{x(v):.1f}" cy="{y + dy:.1f}" r="5" class="dot s{s + 1}">'
                       f'<title>{html.escape(tip)}</title></circle>')
    out.append("</svg>")
    return "\n".join(out)


def table(rows):
    head = "".join(f"<th>{html.escape(n)}</th>" for _, n in SERIES)
    body = []
    for bench, vals in rows:
        cells = []
        for sid, _ in SERIES:
            a = vals.get(sid)
            if not a:
                cells.append("<td class='muted'>running</td>")
                continue
            ba, nll = a["bal_acc"], a["nll"]
            sd = f" ± {ba[1]:.3f}" if ba[2] > 1 else ""
            cells.append(f"<td>{ba[0]:.3f}{sd}<br><span class='muted'>NLL {nll[0]:.3f}</span></td>")
        body.append(f"<tr><th scope='row'>{html.escape(bench)}</th>{''.join(cells)}</tr>")
    return f"<table><thead><tr><th>Benchmark</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def confirm_table():
    rows = []
    for f in sorted((EXPS / "confirm-1" / "results").glob("summary_*.json")):
        for name, s in json.loads(f.read_text()).items():
            rows.append(f"<tr><th scope='row'>{html.escape(name)}</th><td>{s['fp_bal_acc']:.3f}</td>"
                        f"<td>{s['fp_nll']:.3f}</td><td>{s['mix_soft']:.3f}</td>"
                        f"<td>{s['mix_hard']:.3f}</td><td>{s['mix_oracle']:.3f}</td></tr>")
    if not rows:
        return "<p class='muted'>Not run yet.</p>"
    return ("<table><thead><tr><th>Fingerprint</th><th>21-way bal. acc</th><th>NLL</th>"
            "<th>Mixture, soft</th><th>hard</th><th>oracle ID</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def band_table():
    d = EXPS / "h3.2-band-ablation" / "results"
    if not d.exists():
        return "<p class='muted'>Running.</p>"
    benches = ["R12toR3", "D1_R1toR3", "D2_bnci_s1tos2"]
    found = {p.stem.split("_", 1)[1] for p in d.glob("R12toR3_*.json")}
    lo = lambda a: int(a.split("_")[1].split("-")[0])
    arms = (sorted([a for a in found if a.startswith("single_")], key=lo)
            + sorted([a for a in found if a.startswith("drop_")], key=lo)
            + sorted(a for a in found if not a.startswith(("single_", "drop_"))))
    rows = []
    for arm in arms:
        cells = []
        for b in benches:
            m = res("h3.2-band-ablation", f"{b}_{arm}")
            cells.append(f"<td>{m['bal_acc']:.3f} <span class='muted'>/ {m['nll']:.3f}</span></td>"
                         if m else "<td class='muted'>–</td>")
        rows.append(f"<tr><th scope='row'>{arm}</th>{''.join(cells)}</tr>")
    return ("<table><thead><tr><th>Bands</th><th>R1–R2 → R3</th><th>R1 → R3</th>"
            "<th>BNCI cross-day</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def heldout_table():
    arms = ["fb6", "fine18", "fine11_8_30"]
    rows = []
    for ds in ("BNCI2015_001", "Zhou2016"):
        cells = []
        for a in arms:
            m = res("h3.3-bank-design", f"heldout_{ds}_{a}")
            cells.append(f"<td>{m['bal_acc']:.3f} <span class='muted'>/ {m['nll']:.3f}</span></td>"
                         if m else "<td class='muted'>–</td>")
        rows.append(f"<tr><th scope='row'>{ds}</th>{''.join(cells)}</tr>")
    return ("<table><thead><tr><th>Held-out dataset (day 1 → later days)</th>"
            + "".join(f"<th>{a}</th>" for a in arms) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def multiday_table():
    arms = ["S1", "S2", "S1+S2", "S1+S2_halfmatched"]
    labels = ["day 1", "day 2", "days 1 + 2", "days 1 + 2, data-matched"]
    rows = []
    for ds in ("BNCI2015_001", "Zhou2016"):
        cells = []
        for a in arms:
            m = res("h11-multiday-calibration", f"{ds}_fb6_{a}")
            cells.append(f"<td>{m['bal_acc']:.3f} <span class='muted'>/ {m['nll']:.3f}</span></td>"
                         if m else "<td class='muted'>–</td>")
        rows.append(f"<tr><th scope='row'>{ds} → day 3</th>{''.join(cells)}</tr>")
    return ("<table><thead><tr><th>Trained on</th>" + "".join(f"<th>{l}</th>" for l in labels)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def channel_plot():
    p = EXPS / "h12-channel-count" / "results" / "curves.json"
    if not p.exists():
        return "<p class='muted'>Running.</p>"
    cur = json.loads(p.read_text())
    W, H, left, right, top, bottom = 720, 300, 56, 200, 16, 40
    x = lambda k: left + (k - 4) / (28 - 4) * (W - left - right)
    y = lambda v: top + (1.0 - v) / (1.0 - 0.8) * (H - top - bottom)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Fingerprint accuracy vs channels" class="chart">']
    for v in (0.8, 0.85, 0.9, 0.95, 1.0):
        out.append(f'<line x1="{left}" x2="{W - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{left - 8}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    for k in (6, 9, 13, 17, 20, 22, 27):
        out.append(f'<text x="{x(k):.1f}" y="{H - 18}" class="tick" text-anchor="middle">{k}</text>')
    out.append(f'<text x="{(left + W - right) / 2:.0f}" y="{H - 2}" class="tick" text-anchor="middle">channels (random subsets)</text>')
    for s, (key, lab) in enumerate((("D2_bnci_s1tos2", "BNCI day 1 → 2 (9 people)"),
                                    ("D1_R1toR3", "Dreyer R1 → R3 (21 people)"))):
        pts = sorted((int(k), v["bal_acc"][0], v["bal_acc"][1]) for k, v in cur.get(key, {}).items())
        if not pts:
            continue
        cls = f"s{s + 1}"
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(k):.1f},{y(max(m, 0.8)):.1f}" for i, (k, m, _) in enumerate(pts))
        out.append(f'<path d="{path}" class="line {cls}"/>')
        for k, m, sd in pts:
            out.append(f'<circle cx="{x(k):.1f}" cy="{y(max(m, 0.8)):.1f}" r="4.5" class="dot {cls}">'
                       f'<title>{html.escape(lab)}, {k} channels: {m:.3f} ± {sd:.3f}</title></circle>')
        k, m, _ = pts[-1]
        out.append(f'<text x="{x(k) + 10:.1f}" y="{y(max(m, 0.8)) + 4 + s * 14:.1f}" class="rowlab">{html.escape(lab)}</text>')
    out.append("</svg>")
    return "\n".join(out)


CSS = """
:root{--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;--grid:#e6e5e0;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--card:#ffffff;--border:#e6e5e0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--text:#fff;
--text2:#c3c2b7;--muted:#8f8e86;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--card:#222220;--border:#33332f}}
:root[data-theme="dark"]{--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;--muted:#8f8e86;--grid:#33332f;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--card:#222220;--border:#33332f}
body{background:var(--surface);color:var(--text);font:15px/1.55 system-ui,sans-serif;margin:0}
main{max-width:860px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px}
p,li{color:var(--text2)} .muted{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;margin:12px 0;overflow-x:auto}
.chart{width:100%;height:auto}.grid{stroke:var(--grid);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}.rowlab{fill:var(--text2);font-size:12px}
.dot{stroke:var(--card);stroke-width:2}.s1{fill:var(--s1);stroke:var(--s1)}.s2{fill:var(--s2);stroke:var(--s2)}
.s3{fill:var(--s3);stroke:var(--s3)}.line{fill:none;stroke-width:2}.line.s1{stroke:var(--s1)}.line.s2{stroke:var(--s2)}.dot.s1,.dot.s2,.dot.s3{stroke:var(--card)}.err{stroke-width:2;opacity:.5}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--text2);margin:4px 0 8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:top}
thead th{color:var(--muted);font-weight:600}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi b{display:block;font-size:26px;color:var(--text)}.kpi span{color:var(--muted);font-size:13px}
"""


def main(n, narrative):
    rows = dev_rows()
    legend = "".join(f"<span><i style='background:var(--s{i + 1})'></i>{html.escape(nm)}</span>"
                     for i, (_, nm) in enumerate(SERIES))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Fingerprint loop progress</title>
<style>{CSS}</style></head><body><main>
<h1>Track 2 loop A: fingerprint model</h1>
<p class="muted">Progress report {n} · branch exp/fingerprint-model · dev decisions on R3 and drift proxies only</p>
{narrative}
<h2>Person-ID balanced accuracy on the dev benchmarks</h2>
<div class="card"><div class="legend">{legend}</div>
{dot_plot(rows, "bal_acc", 0.5, 1.0, "Balanced accuracy by benchmark and model")}
<p class="muted">Dots: mean over seeds (EEGNet) or the deterministic value (Riemann); bars: ±1 SD over 3 seeds.
Chance: 1/21 = 0.048 (Dreyer), 1/9 = 0.111 (BNCI).</p></div>
<div class="card">{table(rows)}</div>
<h2>Across days: the risk for the sealed phase</h2>
<p class="muted">Held-out cross-day datasets (H3.3): the finer banks chosen on BNCI2014 do not generalize.</p>
<div class="card">{heldout_table()}</div>
<p class="muted">Multi-day calibration (H11), fb6, balanced accuracy / NLL on day 3.</p>
<div class="card">{multiday_table()}</div>
<p class="muted">Channel count (H12): fb6 balanced accuracy, mean ± SD over 10 random channel subsets.</p>
<div class="card">{channel_plot()}</div>
<h2>Band ablation (H3.2): balanced accuracy / NLL</h2>
<div class="card">{band_table()}</div>
<h2>Confirmation on the hidden runs R4–R6 (pre-registered, confirm-1)</h2>
<p class="muted">Soft mixture uses the fixed EEGNet experts of the shipped seed-0 mixture. Shipped: fingerprint 0.744, soft 0.901, oracle 0.908.</p>
<div class="card">{confirm_table()}</div>
</main></body></html>"""
    out = R / "to_human" / f"progress-{n}.html"
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    n = sys.argv[1] if len(sys.argv) > 1 else "001"
    narr = Path(sys.argv[2]).read_text() if len(sys.argv) > 2 else ""
    main(n, narr)
