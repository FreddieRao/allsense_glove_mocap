#!/usr/bin/env python3
"""Rigid-body smoothing, (1) zero-phase Butterworth vs (2) Kalman / RTS, applied to the re-solved POSE or to the RAW
MARKER tracks (`markers:*`, then the pose re-solved from the smoothed markers), against the raw MoSca poses.

Two numbers per body and take, because smoothing trades one for the other:
  jitter     pose minus its 50 ms moving average (scripts/rb_jitter.py:jitter), RMS / p50 / p95 -- lower is smoother
  fidelity   |R x_row + t - marker| over every marker MoSca accepted for the body (p50 / p95 mm). The raw pose
             is the per-frame least-squares fit, so smoothing can only RAISE this; a smoother that raises it
             by more than the noise it removed is cutting real motion.
MANO-level effects (fit error, output smoothness) come from running the pipeline with `smooth.method=...`
and `run.tag=...` and are compared by evaluation/fit_error.py.

    ~/miniforge3/envs/far_grasp/bin/python -m scripts.compare_smoothing
Writes notes/2026-10-22_smoothing_rb.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import ROOT, write_json            # noqa: E402
from config import load_config, strip_overrides           # noqa: E402
from mosca.smooth import resolve_from_markers, smooth_body, smooth_marker_tracks   # noqa: E402
from scripts.rb_jitter import jitter                      # noqa: E402


def fidelity(z, k: int, name: str, R: np.ndarray, t_mm: np.ndarray, valid: np.ndarray) -> list[float]:
    on = z["body"] == k
    f, r, y = z["frame"][on], z["row"][on], z["p_world_mm"][on].astype(float)
    ok = valid[f]
    tk = z[f"{name}__template_mm"]
    e = np.linalg.norm(np.einsum("nij,nj->ni", R[f[ok]], tk[r[ok]]) + t_mm[f[ok]] - y[ok], axis=1)
    return [round(float(np.median(e)), 3), round(float(np.percentile(e, 95)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    out = {"source": "scripts/compare_smoothing.py", "smooth": cfg.smooth.to_dict(), "takes": {}}
    print(f"{'take':12s} {'body':22s} {'method':15s} {'jitter mm rms/p50/p95':>24s} {'jitter deg rms/p50/p95':>24s} "
          f"{'fidelity mm p50/p95':>20s}")
    for take in ("calibration", "free_motion"):
        z = np.load(ROOT / cfg.data.out_root / take / "mosca.npz")
        rate = float(z["rate_hz"])
        half = max(1, int(round(0.050 * rate / 2)))
        res = {}
        for k, n in enumerate(z["body_names"]):
            R0, t0 = z[f"{n}__R_world_from_body_seq"].astype(float), z[f"{n}__t_world_from_body_mm_seq"].astype(float)
            v = z[f"{n}__valid"]
            res[n] = {}
            on = z["body"] == k
            M = np.full((len(v), len(z[f"{n}__template_mm"]), 3), np.nan)
            M[z["frame"][on], z["row"][on]] = z["p_world_mm"][on]
            for m in ("none", "butter", "rts", "markers:butter", "markers:rts"):
                vv = v
                if m.startswith("markers:"):
                    # option B: smooth each raw marker track, then re-solve the pose from the smoothed markers
                    Ms = smooth_marker_tracks(M, rate, m.split(":")[1], cfg.smooth)
                    R, t, vv = resolve_from_markers(z[f"{n}__template_mm"], Ms, v)
                else:
                    R, t = smooth_body(m, R0, t0, v, z[f"{n}__rmse_mm"].astype(float), z[f"{n}__n"].astype(int),
                                       z[f"{n}__template_mm"], rate, cfg.smooth)
                j = jitter(R, t, vv, half)
                fd = fidelity(z, k, n, R, t, vv)
                res[n][m] = {"jitter": j, "fidelity_mm_p50_p95": fd}
                f = lambda a: "/".join(f"{x:.3f}" for x in a)                          # noqa: E731
                print(f"{take:12s} {n:22s} {m:15s} {f(j['trans_mm_rms_p50_p95']):>24s} {f(j['rot_deg_rms_p50_p95']):>24s} "
                      f"{f(fd):>20s}", flush=True)
        out["takes"][take] = res
    write_json(ROOT / "notes" / "2026-10-22_smoothing_rb.json", out)


if __name__ == "__main__":
    main()
