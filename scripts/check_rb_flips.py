#!/usr/bin/env python3
"""Diagnostic: is Motive's own rigid-body solve usable for the caps, and does the MoSca re-solve fix it?

Two measurements per take, both from `outputs/<take>/mosca.npz` (Stage B):

  marker residual   |pose x template - marker| over the markers MoSca accepted for the body, with the pose
                    either Motive's (CSV) or MoSca's re-solve, the template the take's refined one. A flip
                    shows up as a residual of the cap's own size (~10-17 mm), not as noise.
  orientation       the tip cap's rotation relative to the knuckle cap: deviation from its mean (on the
                    calibration take the hand is rigid in the fixture, so that deviation IS the error) and the
                    frame-to-frame step at 240 Hz (> 20 deg in one frame is not a finger).

    ~/miniforge3/envs/far_grasp/bin/python -m scripts.check_rb_flips
Writes notes/2026-10-22_rb_flips.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import ROOT, rot_deg, rot_mean, write_json        # noqa: E402
from config import load_config, strip_overrides                         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(strip_overrides())
    cfg = load_config()
    B = cfg.data.bodies
    out = {"source": "scripts/check_rb_flips.py", "takes": {}}
    print("| take | body | Motive marker resid p50 / p90 / p99 mm | MoSca marker resid p50 / p90 / p99 mm |")
    print("|---|---|---|---|")
    for take in ("calibration", "free_motion"):
        z = np.load(ROOT / cfg.data.out_root / take / "mosca.npz")
        names = list(z["body_names"])
        frame, p, body, row = z["frame"], z["p_world_mm"].astype(float), z["body"], z["row"]
        res = {"bodies": {}}
        for n in (B.fixture, B.knuckle, B.tip):
            k = names.index(n)
            tk = z[f"{n}__template_mm"]
            on = body == k
            f, r, y = frame[on], row[on], p[on]
            row_res = {}
            for src, R, t, v in (("motive", z[f"{n}__motive_R_world_from_body_seq"], z[f"{n}__motive_t_world_from_body_mm_seq"],
                                  z[f"{n}__motive_valid"]),
                                 ("mosca", z[f"{n}__R_world_from_body_seq"], z[f"{n}__t_world_from_body_mm_seq"], z[f"{n}__valid"])):
                ok = v[f]
                e = np.linalg.norm(np.einsum("nij,nj->ni", R[f[ok]].astype(float), tk[r[ok]]) + t[f[ok]] - y[ok], axis=1)
                row_res[src] = {"p50_mm": round(float(np.median(e)), 3), "p90_mm": round(float(np.percentile(e, 90)), 3),
                                "p99_mm": round(float(np.percentile(e, 99)), 3), "over_5mm_frac": round(float((e > 5).mean()), 4)}
            res["bodies"][n] = row_res
            print(f"| {take} | {n} | " + " | ".join(
                f"{row_res[s]['p50_mm']:.2f} / {row_res[s]['p90_mm']:.2f} / {row_res[s]['p99_mm']:.2f}" for s in ("motive", "mosca")) + " |")
        for src, pre in (("motive", "motive_"), ("mosca", "")):
            Rk, Rt = z[f"{B.knuckle}__{pre}R_world_from_body_seq"].astype(float), z[f"{B.tip}__{pre}R_world_from_body_seq"].astype(float)
            ok = z[f"{B.knuckle}__{pre}valid"] & z[f"{B.tip}__{pre}valid"]
            Rrel = np.einsum("tji,tjk->tik", Rk[ok], Rt[ok])
            dev = rot_deg(Rrel, rot_mean(Rrel)[None])
            step = rot_deg(Rrel[1:], Rrel[:-1])
            res[f"tip_rel_knuckle_{src}"] = {"dev_from_mean_deg_p50_p90_p99": np.percentile(dev, [50, 90, 99]).round(2),
                                             "step_deg_p99": round(float(np.percentile(step, 99)), 2),
                                             "steps_over_20deg": int((step > 20).sum()), "frames": int(ok.sum())}
        out["takes"][take] = res
    print()
    for take, res in out["takes"].items():
        for src in ("motive", "mosca"):
            r = res[f"tip_rel_knuckle_{src}"]
            print(f"{take:12s} {src:6s} tip-vs-knuckle rotation: dev from mean p50/p90/p99 "
                  f"{r['dev_from_mean_deg_p50_p90_p99']} deg, steps > 20 deg: {r['steps_over_20deg']}")
    write_json(ROOT / "notes" / "2026-10-22_rb_flips.json", out)


if __name__ == "__main__":
    main()
