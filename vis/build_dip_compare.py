#!/usr/bin/env python3
"""A self-contained 2D HTML page: why DIP barely bends -- the default calibration vs `rev2`.

Panels (one y-axis each, two series, inline SVG + hover tooltips):
  1. objective (mm^2) on the 64 most-flexed free_motion frames with DIP HELD at each value, the other 8 index
     numbers re-fitted (scripts/dip_profile.py) -- where the minimum sits is the DIP the markers prefer;
  2. middle floating-marker RMS (mm) over the same sweep -- the term that blocks DIP in the default;
  3. tracked PIP vs DIP per frame (small multiples, one panel per calibration, 1500 frames each);
  4. a summary table (fit error, flexion range, marker placement).

    ~/miniforge3/envs/far_grasp/bin/python -m vis.build_dip_compare
Writes notes/figures/dip_compare.html. Inputs: notes/2026-10-27_dip_profile{,_rev2}.json,
outputs/free_motion/mano_track{,_rev2}.npz, notes/2026-10-22_calibration_eval{,_rev2}.json.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import ROOT                                   # noqa: E402
from config import load_config, strip_overrides                      # noqa: E402

VARIANTS = [("", "default (hand held at design + pivot corrections)", "--series-1"),
            ("_rev2", "rev2 (hand free, no pivot corrections)", "--series-2")]
W, H, PAD_L, PAD_B, PAD_T, PAD_R = 460, 280, 52, 40, 16, 16

CSS = """
.viz-root { color-scheme: light; --surface-1:#fcfcfb; --surface-2:#f3f2ef; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --text-muted:#7c7b76; --grid:#e4e3df; --series-1:#2a78d6; --series-2:#eb6834; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232321; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#8e8d86;
  --grid:#33332f; --series-1:#3987e5; --series-2:#d95926; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark; --surface-1:#1a1a19; --surface-2:#232321;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#8e8d86; --grid:#33332f; --series-1:#3987e5; --series-2:#d95926; }
body { margin:0; }
.viz-root { background:var(--surface-1); color:var(--text-primary); font:13px/1.4 system-ui,sans-serif; padding:18px 22px; min-height:100vh; box-sizing:border-box; }
h1 { font-size:17px; margin:0 0 4px; } h2 { font-size:14px; margin:0 0 2px; } .sub { color:var(--text-secondary); margin:0 0 14px; max-width:980px; }
.row { display:flex; flex-wrap:wrap; gap:22px; margin-bottom:22px; } .card { background:var(--surface-2); border-radius:8px; padding:12px 14px; }
.cap { color:var(--text-secondary); font-size:12px; margin:0 0 6px; max-width:440px; }
.legend { display:flex; gap:16px; margin:0 0 14px; color:var(--text-secondary); } .legend i { display:inline-block; width:12px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }
svg text { fill:var(--text-secondary); font-size:11px; } svg .axis { stroke:var(--text-muted); stroke-width:1; } svg .grid { stroke:var(--grid); stroke-width:1; }
table { border-collapse:collapse; font-size:12.5px; } th, td { padding:4px 10px; border-bottom:1px solid var(--grid); text-align:right; } th:first-child, td:first-child { text-align:left; }
th { color:var(--text-secondary); font-weight:600; } td.win { font-weight:600; }
#tip { position:fixed; pointer-events:none; background:var(--surface-1); color:var(--text-primary); border:1px solid var(--grid); border-radius:6px; padding:5px 8px; font-size:12px; display:none; box-shadow:0 2px 8px rgba(0,0,0,.15); white-space:pre; }
"""

JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => { tip.textContent = el.dataset.tip; tip.style.display = 'block';
    tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; });
  el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
"""


def scale(v, lo, hi, a, b):
    return a + (np.asarray(v, float) - lo) / (hi - lo) * (b - a)


def axes(xlo, xhi, ylo, yhi, xt, yt, xl, yl) -> str:
    x0, x1, y0, y1 = PAD_L, W - PAD_R, H - PAD_B, PAD_T
    s = []
    for v in yt:
        y = scale(v, ylo, yhi, y0, y1)
        s.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/><text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{v:g}</text>')
    for v in xt:
        x = scale(v, xlo, xhi, x0, x1)
        s.append(f'<text x="{x:.1f}" y="{y0 + 16}" text-anchor="middle">{v:g}</text>')
    s.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>')
    s.append(f'<text x="{(x0 + x1) / 2}" y="{H - 6}" text-anchor="middle">{html.escape(xl)}</text>')
    s.append(f'<text transform="translate(13,{(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">{html.escape(yl)}</text>')
    return "".join(s)


def line_chart(series: list, key: str, ylabel: str, ymax: float) -> str:
    """series: [(label, colour var, rows)]; rows from the dip profile."""
    out = [f'<svg width="{W}" height="{H}" role="img" aria-label="{html.escape(ylabel)} vs held DIP">',
           axes(0, 90, 0, ymax, range(0, 91, 10), np.linspace(0, ymax, 5).round(1), "DIP held at (deg)", ylabel)]
    for label, col, rows in series:
        xs = scale([r["dip_deg"] for r in rows], 0, 90, PAD_L, W - PAD_R)
        ys = scale([min(r[key], ymax) for r in rows], 0, ymax, H - PAD_B, PAD_T)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="var({col})" stroke-width="2" stroke-linejoin="round"/>')
        best = int(np.argmin([r[key] for r in rows]))
        for i, (x, y, r) in enumerate(zip(xs, ys, rows)):
            t = (f"{label}\nDIP held {r['dip_deg']:.0f} deg\n{ylabel}: {r[key]:.2f}\nPIP {r['pip_mean_deg']:.0f} deg, "
                 f"middle marker {r['float_mid_rms_mm']:.2f} mm, tip cap {r['tip_rms_mm']:.2f} mm")
            rad = 5 if i == best else 4
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad}" fill="var({col})" stroke="var(--surface-2)" stroke-width="2"/>'
                       f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="transparent" data-tip="{html.escape(t)}"/>')
        bx, by = xs[best], ys[best]
        out.append(f'<text x="{bx:.1f}" y="{by - 10:.1f}" text-anchor="middle" style="fill:var(--text-primary)">min @ {rows[best]["dip_deg"]:.0f}°</text>')
    out.append("</svg>")
    return "".join(out)


def scatter(label: str, col: str, pip: np.ndarray, dip: np.ndarray) -> str:
    out = [f'<svg width="{W}" height="{H}" role="img" aria-label="tracked PIP vs DIP, {html.escape(label)}">',
           axes(-20, 120, -20, 100, range(-20, 121, 20), range(-20, 101, 20), "PIP flexion (deg)", "DIP flexion (deg)")]
    xs, ys = scale(pip, -20, 120, PAD_L, W - PAD_R), scale(dip, -20, 100, H - PAD_B, PAD_T)
    for x, y, p, d in zip(xs, ys, pip, dip):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="var({col})" fill-opacity="0.45" data-tip="PIP {p:.0f} deg, DIP {d:.0f} deg"/>')
    # reference: DIP = 2/3 PIP, the usual human coupling
    a, b = scale([0, 120], -20, 120, PAD_L, W - PAD_R), scale([0, 80], -20, 100, H - PAD_B, PAD_T)
    out.append(f'<line x1="{a[0]:.1f}" y1="{b[0]:.1f}" x2="{a[1]:.1f}" y2="{b[1]:.1f}" stroke="var(--text-muted)" stroke-dasharray="4 4" stroke-width="1"/>'
               f'<text x="{a[1] - 4:.1f}" y="{b[1] - 6:.1f}" text-anchor="end">DIP = ⅔ PIP</text>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", type=int, default=1500)
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    prof, track, ev, cal = {}, {}, {}, {}
    for tag, _l, _c in VARIANTS:
        prof[tag] = json.loads((ROOT / "notes" / f"2026-10-27_dip_profile{tag}.json").read_text())
        z = np.load(ROOT / cfg.data.out_root / "free_motion" / f"mano_track{tag}.npz")
        th = np.degrees(z["index_pose"][z["valid"]])
        sel = np.linspace(0, len(th) - 1, args.points).astype(int)
        track[tag] = (th[sel, 5], th[sel, 8])
        ev[tag] = json.loads((ROOT / "notes" / f"2026-10-22_calibration_eval{tag}.json").read_text())["takes"]
        cal[tag] = json.loads((ROOT / cfg.data.out_root / "calibration" / f"hand_calib{tag}.json").read_text())
    series = [(lab, col, prof[tag]["profile"]) for tag, lab, col in VARIANTS]
    legend = "".join(f'<span><i style="background:var({c})"></i>{html.escape(l)}</span>' for _t, l, c in VARIANTS)

    # summary table: lower is better for errors; the better value per row is bold
    def row(name, vals, fmt, better="low"):
        b = int(np.argmin(vals) if better == "low" else np.argmax(vals)) if better else -1
        return "<tr><td>" + name + "</td>" + "".join(
            f'<td class="{"win" if i == b else ""}">{fmt(v)}</td>' for i, v in enumerate(vals)) + "</tr>"

    f = lambda tag, sp, k: ev[tag]["free_motion"][sp][k]["rms_mm"]                       # noqa: E731
    rows = [
        row("free motion RMS, tip cap (mm)", [f(t, "all", "tip_cap") for t, *_ in VARIANTS], "{:.2f}".format),
        row("free motion RMS, floating proximal (mm)", [f(t, "all", "float_proximal") for t, *_ in VARIANTS], "{:.2f}".format),
        row("free motion RMS, floating middle (mm)", [f(t, "all", "float_middle") for t, *_ in VARIANTS], "{:.2f}".format),
        row("held-out RMS, tip / prox / mid (mm)", [sum(f(t, "heldout", k) for k in ("tip_cap", "float_proximal", "float_middle"))
                                                    for t, *_ in VARIANTS],
            lambda v: f"{v:.2f} (sum)"),
        row("distal-direction error (deg)", [f(t, "all", "distal_dir_deg") for t, *_ in VARIANTS], "{:.2f}".format),
        row("DIP range p1-p99 (deg)", [ev[t]["free_motion"]["index_flexion_range_deg_p1_p99"]["dip"][1] for t, *_ in VARIANTS],
            lambda v: f"up to {v:.0f}", better=None),
        row("PIP range p1-p99 (deg)", [ev[t]["free_motion"]["index_flexion_range_deg_p1_p99"]["pip"][1] for t, *_ in VARIANTS],
            lambda v: f"up to {v:.0f}", better=None),
        row("DIP the markers prefer, most-flexed frames (deg)",
            [prof[t]["profile"][int(np.argmin([r["objective_mm2"] for r in prof[t]["profile"]]))]["dip_deg"] for t, *_ in VARIANTS],
            "{:.0f}".format, better=None),
        row("middle marker along its phalanx", [prof[t]["floating_placement"]["middle"]["along_bone_frac"] for t, *_ in VARIANTS],
            lambda v: f"{100 * v:.0f}%", better=None),
        row("hand vs designed position (mm)", [cal[t]["fixture"]["hand_vs_design"]["dt_mm"] for t, *_ in VARIANTS],
            "{:.1f}".format, better=None),
        row("MCP pivot correction (mm)", [float(np.linalg.norm(cal[t].get("index_joint_offset_mm", {}).get("mcp", [0, 0, 0])))
                                         for t, *_ in VARIANTS], "{:.1f}".format, better=None),
    ]
    table = ("<table><tr><th></th>" + "".join(f"<th>{html.escape(l.split(' (')[0])}</th>" for _t, l, _c in VARIANTS)
             + "</tr>" + "".join(rows) + "</table>")
    body = f"""<div class="viz-root">
<h1>Why DIP barely bends: default calibration vs rev2</h1>
<p class="sub">On the 64 most-flexed free-motion frames, DIP is held at each value and the other 8 index joint numbers are
re-fitted to the markers (scripts/dip_profile.py). Where the objective is lowest is the DIP the markers prefer. In the
default, the middle floating marker sits near PIP and blocks any real DIP bend; in rev2 it sits mid-phalanx and the
markers prefer a physiological DIP.</p>
<div class="legend">{legend}</div>
<div class="row">
 <div class="card"><h2>Objective vs held DIP</h2><p class="cap">Mean Stage-E objective (mm², tip cap + floating markers + distal direction). Lower is better; the label marks each minimum.</p>{line_chart(series, "objective_mm2", "objective (mm²)", 50)}</div>
 <div class="card"><h2>Middle floating-marker error vs held DIP</h2><p class="cap">RMS (mm) of the middle-phalanx marker. In the default it rises steeply past 40°: the marker vetoes DIP.</p>{line_chart(series, "float_mid_rms_mm", "middle marker RMS (mm)", 40)}</div>
</div>
<div class="row">
 <div class="card"><h2>Tracked PIP vs DIP — default</h2><p class="cap">{args.points} free-motion frames. Dashed: the usual human coupling DIP ≈ ⅔ PIP.</p>{scatter(VARIANTS[0][1], VARIANTS[0][2], *track[""])}</div>
 <div class="card"><h2>Tracked PIP vs DIP — rev2</h2><p class="cap">Same frames and axes.</p>{scatter(VARIANTS[1][1], VARIANTS[1][2], *track["_rev2"])}</div>
</div>
<div class="row"><div class="card"><h2>Summary</h2><p class="cap">Bold = lower error. Sources: notes/2026-10-22_calibration_eval{{,_rev2}}.json, notes/2026-10-27_dip_profile{{,_rev2}}.json.</p>{table}</div></div>
<div id="tip"></div></div>"""
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>DIP: default vs rev2</title><style>{CSS}</style></head>"
            f"<body>{body}<script>{JS}</script></body></html>")
    out = ROOT / "notes" / "figures" / "dip_compare.html"
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
