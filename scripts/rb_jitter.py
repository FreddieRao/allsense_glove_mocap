#!/usr/bin/env python3
"""Rigid-body jitter, Motive's own solve vs the MoSca re-solve, for every body of both takes.

JITTER = the high-frequency part of the pose: the pose minus a smoothed copy of itself. The smoothing is a
centred moving window of `--window-ms` (default 50 ms = 12 frames at 240 Hz) over the valid frames, applied to
the translation and to the rotation (as rotation vectors relative to the window centre, so it is wrap-free).
A real hand or fixture motion is smooth on 50 ms; what is left is noise -- and flips, which appear as huge
outliers, so the median and p95 are reported next to the RMS.

Also reported: the frame-to-frame step (240 Hz) in mm and deg, and for the static fixture in free_motion the
plain standard deviation of its pose over the take (it does not move, so all of it is jitter).

    ~/miniforge3/envs/far_grasp/bin/python -m scripts.rb_jitter
Writes notes/2026-10-22_rb_jitter.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import ROOT, write_json            # noqa: E402
from config import load_config, strip_overrides           # noqa: E402


def jitter(R: np.ndarray, t_mm: np.ndarray, valid: np.ndarray, half: int) -> dict:
    """Translation (mm) and rotation (deg) jitter about a centred moving window of 2*half+1 frames."""
    idx = np.flatnonzero(valid & np.isfinite(R[:, 0, 0]) & np.isfinite(t_mm).all(1))
    # only frames whose whole window is valid and contiguous
    ok = np.zeros(len(R), bool)
    ok[idx] = True
    cs = np.r_[0, np.cumsum(ok)]
    full = np.array([i for i in idx if i - half >= 0 and i + half < len(R) and cs[i + half + 1] - cs[i - half] == 2 * half + 1])
    Rv = Rotation.from_matrix(R[idx])
    Rf = R.copy()
    Rf[~ok] = np.eye(3)                                   # never read: only full windows are evaluated
    rot_all = Rotation.from_matrix(Rf)
    k = np.ones(2 * half + 1) / (2 * half + 1)
    t_s = np.stack([np.convolve(np.nan_to_num(t_mm[:, a]), k, "same") for a in range(3)], 1)
    dt = np.linalg.norm(t_mm[full] - t_s[full], axis=1)
    dr = np.empty(len(full))
    for n, i in enumerate(full):
        rel = (rot_all[i].inv() * rot_all[i - half:i + half + 1]).as_rotvec()   # window relative to its centre
        dr[n] = np.degrees(np.linalg.norm(rel.mean(0)))                          # centre vs window mean
    st = np.linalg.norm(np.diff(t_mm[idx], axis=0), axis=1)[np.diff(idx) == 1]
    sr = np.degrees((Rv[1:] * Rv[:-1].inv()).magnitude())[np.diff(idx) == 1]
    q = lambda a: [round(float(x), 3) for x in (np.sqrt((a ** 2).mean()), np.median(a), np.percentile(a, 95))]  # noqa: E731
    return {"frames": int(len(full)), "trans_mm_rms_p50_p95": q(dt), "rot_deg_rms_p50_p95": q(dr),
            "step_mm_p50_p99": [round(float(np.median(st)), 3), round(float(np.percentile(st, 99)), 3)],
            "step_deg_p50_p99": [round(float(np.median(sr)), 3), round(float(np.percentile(sr, 99)), 3)]}


def static_std(R: np.ndarray, t_mm: np.ndarray, valid: np.ndarray) -> dict:
    ok = valid & np.isfinite(R[:, 0, 0])
    Rv = Rotation.from_matrix(R[ok])
    dev = np.degrees((Rv.mean().inv() * Rv).magnitude())
    return {"trans_std_mm": np.round(t_mm[ok].std(0), 3).tolist(), "rot_dev_deg_rms": round(float(np.sqrt((dev ** 2).mean())), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window-ms", type=float, default=50.0)
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    out = {"source": "scripts/rb_jitter.py", "window_ms": args.window_ms, "takes": {}}
    print(f"jitter = pose minus its {args.window_ms:.0f} ms moving average; RMS / p50 / p95")
    print(f"{'take':12s} {'body':22s} {'src':7s} {'trans mm':>22s} {'rot deg':>22s} {'step mm p50/p99':>17s} {'step deg p50/p99':>17s}")
    for take in ("calibration", "free_motion"):
        z = np.load(ROOT / cfg.data.out_root / take / "mosca.npz")
        half = max(1, int(round(args.window_ms / 1000.0 * float(z["rate_hz"]) / 2)))
        res = {}
        for n in z["body_names"]:
            res[n] = {}
            for src, pre in (("motive", "motive_"), ("mosca", "")):
                R = z[f"{n}__{pre}R_world_from_body_seq"].astype(float)
                t = z[f"{n}__{pre}t_world_from_body_mm_seq"].astype(float)
                v = z[f"{n}__{pre}valid"]
                j = jitter(R, t, v, half)
                if take == "free_motion" and n == cfg.data.bodies.fixture:
                    j["static"] = static_std(R, t, v)
                res[n][src] = j
                f = lambda a: "/".join(f"{x:.3f}" for x in a)                     # noqa: E731
                print(f"{take:12s} {n:22s} {src:7s} {f(j['trans_mm_rms_p50_p95']):>22s} {f(j['rot_deg_rms_p50_p95']):>22s} "
                      f"{f(j['step_mm_p50_p99']):>17s} {f(j['step_deg_p50_p99']):>17s}"
                      + (f"   static std mm {j['static']['trans_std_mm']} rot {j['static']['rot_dev_deg_rms']} deg" if "static" in j else ""))
        out["takes"][take] = res
    write_json(ROOT / "notes" / "2026-10-22_rb_jitter.json", out)


if __name__ == "__main__":
    main()
