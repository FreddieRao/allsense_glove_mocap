#!/usr/bin/env python3
"""Compare the hand-scan MANO fit (data/hand_scan) with the mocap-calibrated MANO (outputs/calibration/hand_calib.json).

The parameters are DIRECTLY comparable: the scan fit uses smplx.MANO right with use_pca=False and
flat_hand_mean=True, and our `ManoIndex` (mano_clean) reproduces the scan's `rest_vertices_mm` from its betas to
8e-6 mm (checked 2026-10-22). So the comparison below is in one model and one canonical frame.

WHAT IS COMPARED
  betas            the 10 numbers, side by side, for: custom_pose.json (our init), our mocap fit, the scan fit
  shape at rest    each hand's flat zero-pose mesh in the canonical frame: per-finger bone lengths (wrist-MCP,
                   proximal, middle, distal-to-tip vertex), palm width (index MCP - pinky MCP), hand length
                   (wrist - middle tip), and the vertex distance between hands (all / index-finger vertices),
                   raw and after a rigid alignment on the palm (betas move the wrist too)
  index pose       the scan's index joint angles (open hand on the scanner) against our calibration-take
                   posture (open hand in the fixture)
  mocap check      Stage D re-run with the scan's betas as start + prior (`scan_init`) and HELD FIXED
                   (`scan_frozen`) -- does the mocap data accept the scan's shape? Fit-frame RMS per marker group.
                   Produced by:
                     python -m calibration.fit_hand --tag scan_frozen fit.betas_init=data/hand_scan/SCAN2_MANO_PARAMETERS.npz fit.freeze_betas=true
                     python -m calibration.fit_hand --tag scan_init   fit.betas_init=data/hand_scan/SCAN2_MANO_PARAMETERS.npz

    ~/miniforge3/envs/far_grasp/bin/python -m scripts.compare_scan_mano
Writes notes/2026-10-22_scan_vs_mocap_mano.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import ROOT, load_custom_pose, procrustes, write_json      # noqa: E402
from calibration.mano_index import ManoIndex                                      # noqa: E402
from config import load_config, strip_overrides                                   # noqa: E402

TIP_VERTEX = {"thumb": 745, "index": 320, "middle": 445, "ring": 556, "pinky": 673}
FINGER_JOINTS = {"index": (1, 2, 3), "middle": (4, 5, 6), "pinky": (7, 8, 9), "ring": (10, 11, 12), "thumb": (13, 14, 15)}


def rest(mano: ManoIndex, betas) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (778,3) mm, joints (16,3) mm) at zero pose, canonical frame."""
    with torch.no_grad():
        P = mano.forward(torch.as_tensor(np.asarray(betas, np.float32).reshape(1, 10)), torch.zeros(1, 45))
    return P.vertices[0].numpy() * 1000.0, P.T_canonical_from_joint[0, :, :3, 3].numpy() * 1000.0


def measures(V: np.ndarray, J: np.ndarray) -> dict:
    out = {}
    for f, (a, b, c) in FINGER_JOINTS.items():
        tip = V[TIP_VERTEX[f]]
        out[f] = {"wrist_mcp": np.linalg.norm(J[a] - J[0]), "proximal": np.linalg.norm(J[b] - J[a]),
                  "middle": np.linalg.norm(J[c] - J[b]), "distal_to_tip": np.linalg.norm(tip - J[c])}
        out[f] = {k: round(float(v), 2) for k, v in out[f].items()}
    out["palm_width_index_pinky_mcp"] = round(float(np.linalg.norm(J[1] - J[7])), 2)
    out["hand_length_wrist_middle_tip"] = round(float(np.linalg.norm(V[TIP_VERTEX["middle"]] - J[0])), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(strip_overrides())
    cfg = load_config()
    mano = ManoIndex(str(cfg.mano.model_dir), device="cpu")
    scan2 = np.load(ROOT / "data/hand_scan/SCAN2_MANO_PARAMETERS.npz")
    scan1 = np.load(ROOT / "data/hand_scan/SCAN1_MANO_PARAMETERS.npz")
    ours = json.loads((ROOT / cfg.data.out_root / "calibration" / "hand_calib.json").read_text())
    hands = {"custom_pose (init)": np.asarray(load_custom_pose(cfg)["final_betas"]),
             "mocap fit (ours)": np.asarray(ours["betas"]),
             "scan fit": scan2["betas"].reshape(-1)}
    assert np.allclose(scan1["betas"], scan2["betas"]), "the two half-scans share one betas vector"
    V, J, M = {}, {}, {}
    for k, b in hands.items():
        V[k], J[k] = rest(mano, b)
        M[k] = measures(V[k], J[k])

    print("betas")
    print(f"{'':20s}" + "".join(f"{f'b{i}':>8s}" for i in range(10)) + "     |b|")
    for k, b in hands.items():
        print(f"{k:20s}" + "".join(f"{x:8.3f}" for x in b) + f"  {np.linalg.norm(b):6.3f}")

    print("\nrest-pose measures (mm)")
    rows = [("index", "wrist_mcp"), ("index", "proximal"), ("index", "middle"), ("index", "distal_to_tip"),
            ("middle", "proximal"), ("middle", "middle"), ("middle", "distal_to_tip"),
            ("thumb", "proximal"), ("thumb", "middle"), ("thumb", "distal_to_tip"),
            ("palm_width_index_pinky_mcp", None), ("hand_length_wrist_middle_tip", None)]
    print(f"{'':34s}" + "".join(f"{k:>22s}" for k in hands) + f"{'scan - ours':>14s}")
    for f, s in rows:
        vals = [M[k][f] if s is None else M[k][f][s] for k in hands]
        name = f if s is None else f"{f} {s}"
        print(f"{name:34s}" + "".join(f"{v:22.2f}" for v in vals) + f"{vals[2] - vals[1]:14.2f}")

    # vertex distance between our hand and the scan hand at rest
    dom = mano.weights.argmax(1).numpy()
    index_v = np.isin(dom, [1, 2, 3])
    palm_v = dom == 0
    a, b = V["mocap fit (ours)"], V["scan fit"]
    R, t = procrustes(a[palm_v], b[palm_v])
    a_al = a @ R.T + t
    d_raw, d_al = np.linalg.norm(a - b, axis=1), np.linalg.norm(a_al - b, axis=1)
    dist = {"raw_all_p50_p95_max_mm": np.percentile(d_raw, [50, 95, 100]).round(2).tolist(),
            "raw_index_p50_p95_max_mm": np.percentile(d_raw[index_v], [50, 95, 100]).round(2).tolist(),
            "palm_aligned_all_p50_p95_max_mm": np.percentile(d_al, [50, 95, 100]).round(2).tolist(),
            "palm_aligned_index_p50_p95_max_mm": np.percentile(d_al[index_v], [50, 95, 100]).round(2).tolist()}
    print("\nvertex distance ours vs scan at rest (mm, p50/p95/max):")
    for k, v in dist.items():
        print(f"  {k:36s} {v}")

    # index pose: scan (open hand on the scanner) vs our calibration-take posture
    pose = {"scan1_index_deg": np.degrees(scan1["hand_pose"][:3]).round(1).tolist(),
            "scan2_index_deg": np.degrees(scan2["hand_pose"][:3]).round(1).tolist(),
            "mocap_calibration_take_index_deg_mean": np.round(ours["theta_cal_index_deg_mean"], 1).tolist()}
    print("\nindex joints, axis-angle components in deg (rows MCP/PIP/DIP, cols x twist / y abduction / z flexion)")
    for k, v in pose.items():
        print(f"  {k:40s} {v}")

    # mocap re-fits with the scan betas
    refits = {}
    for tag in ("", "_scan_init", "_scan_frozen"):
        p = ROOT / cfg.data.out_root / "calibration" / f"hand_calib{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        refits[tag.strip("_") or "ours"] = {
            "betas_frozen": d.get("betas_frozen", False),
            "fit_rms_mm": {t: {g: round(e["rms_mm"], 3) for g, e in v.items()} for t, v in d["fit_error"].items()},
            "index_bone_lengths_mm": {k: round(v, 2) for k, v in d["index_bone_lengths_mm"].items()},
            "float_height_mm": {k: round(v["height_mm"], 2) for k, v in d["floating"].items()},
            "betas_final": np.round(d["betas"], 3).tolist()}
    if refits:
        print("\nStage D fit-frame RMS (mm) with each shape")
        print(f"{'':14s}{'cal tip':>9s}{'cal prox':>9s}{'cal mid':>9s}{'free tip':>9s}{'free prox':>10s}{'free mid':>9s}"
              f"{'h prox':>8s}{'h mid':>7s}")
        for k, r in refits.items():
            c, f = r["fit_rms_mm"]["calibration"], r["fit_rms_mm"]["free_motion"]
            print(f"{k:14s}{c['tip_cap']:9.2f}{c['float_proximal']:9.2f}{c['float_middle']:9.2f}{f['tip_cap']:9.2f}"
                  f"{f['float_proximal']:10.2f}{f['float_middle']:9.2f}{r['float_height_mm']['proximal']:8.2f}"
                  f"{r['float_height_mm']['middle']:7.2f}")
    out = {"source": "scripts/compare_scan_mano.py", "scan_dir": "data/hand_scan",
           "betas": {k: np.round(b, 4).tolist() for k, b in hands.items()}, "measures_mm": M,
           "vertex_distance_ours_vs_scan": dist, "index_pose": pose, "mocap_refits": refits}
    write_json(ROOT / "notes" / "2026-10-22_scan_vs_mocap_mano.json", out)


if __name__ == "__main__":
    main()
