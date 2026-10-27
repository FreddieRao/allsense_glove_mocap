#!/usr/bin/env python3
"""Why does the tracked DIP barely bend? Profile the fit over a FIXED DIP flexion on strongly flexed frames.

For K free_motion frames with the largest tracked total flexion (PIP + DIP), the DIP flexion (about canonical z) is
held at each value of a sweep while the other 8 index-pose numbers are re-optimised against the same objective
Stage E uses (tip cap + floating markers + joint ranges + distal direction). If the residual barely changes across
the sweep, the markers do not determine how a bend splits between PIP and DIP and the split is decided by priors;
if it rises steeply away from the tracked value, the data really prefers that DIP.

Also reports where each floating marker sits on its phalanx (fraction along the bone, skin-weight split), because
a middle-phalanx marker sitting near the PIP joint or skinned partly to the distal bone cannot see DIP.

    ~/miniforge3/envs/far_grasp/bin/python -m scripts.dip_profile [run.calib_tag=...]
Writes notes/2026-10-27_dip_profile.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import ROOT, calib_sfx, load_rig, sfx, write_json                 # noqa: E402
from calibration.hand_model import distal_dir_deg2, joint_penalty, make_frames, robust_mm  # noqa: E402
from calibration.mano_index import ManoIndex, skin_point                               # noqa: E402
from calibration.track import load_calib                                              # noqa: E402
from config import load_config, strip_overrides                                      # noqa: E402


def objective(calib, fr, theta, w_dir: float) -> torch.Tensor:
    """Per-frame Stage E objective (F,): tip + floating + joint range + distal direction."""
    p = calib.predict(fr, theta)
    d_tip = ((p["tip"] - fr.tip_obs).norm(dim=-1) * 1000.0)
    d_fl = ((p["float"] - fr.float_obs).norm(dim=-1) * 1000.0)
    gm = lambda d: d ** 2 / (1 + (d / 5.0) ** 2)                                         # noqa: E731
    tip = (gm(d_tip) * fr.tip_mask).sum(1) / fr.tip_mask.sum(1).clamp_min(1)
    fl = (gm(d_fl) * fr.float_mask).sum(1) / fr.float_mask.sum(1).clamp_min(1)
    d_meas = torch.einsum("fij,j->fi", fr.R_world_from_tipcap, calib.distal_dir_tipcap)
    ang = torch.rad2deg(torch.arccos((p["distal_dir_world"] * d_meas).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)))
    return tip + fl + w_dir * ang ** 2, d_tip, d_fl, ang


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    rig = load_rig("free_motion", cfg)
    mano = ManoIndex(str(cfg.mano.model_dir))
    calib = load_calib(cfg, mano, load_rig("calibration", cfg))
    dev = mano.device
    tr = np.load(ROOT / cfg.data.out_root / "free_motion" / f"mano_track{sfx(cfg)}.npz")
    th = tr["index_pose"]
    ok = np.flatnonzero(tr["valid"])
    total = np.degrees(th[ok, 5] + th[ok, 8])
    sel = ok[np.argsort(-total)[: args.frames]]
    fr = make_frames(rig, sel, dev)
    theta0 = torch.as_tensor(th[sel], device=dev)
    w_dir = float(cfg.track.distal_dir_weight)

    # ---- where the floating markers sit
    with torch.no_grad():
        P = mano.forward(calib.betas, calib.full_pose(torch.zeros(1, 9, device=dev)), calib.joint_offset_full())
        fl = skin_point(P, mano.faces, calib.float_face, calib.float_bary, calib.float_h)[0].cpu().numpy()
        J = P.T_canonical_from_joint[0, :, :3, 3].cpu().numpy()
        W = mano.weights.cpu().numpy()
        F = mano.faces.cpu().numpy()
    place = {}
    for m, (L, b) in enumerate((("proximal", 1), ("middle", 2))):
        seg = J[b + 1] - J[b]
        tri = F[int(calib.float_face[m])]
        wb = (calib.float_bary[m].detach().cpu().numpy()[:, None] * W[tri]).sum(0)
        place[L] = {"along_bone_frac": round(float((fl[m] - J[b]) @ seg / (seg @ seg)), 3),
                    "dist_from_proximal_joint_mm": round(float(np.linalg.norm(fl[m] - J[b]) * 1000), 2),
                    "skin_weight_on_bones": {f"j{j}": round(float(wb[j]), 3) for j in np.argsort(-wb)[:3]}}

    # ---- DIP profile
    tracked_dip = np.degrees(th[sel, 8])
    sweep = np.arange(0.0, 91.0, 10.0)
    rows = []
    for dip in sweep:
        free = torch.nn.Parameter(theta0.clone())
        opt = torch.optim.Adam([free], lr=0.01)
        mask = torch.ones(9, device=dev)
        mask[8] = 0.0
        fixed = torch.zeros(9, device=dev)
        fixed[8] = float(np.radians(dip))
        for _ in range(args.iters):
            opt.zero_grad()
            t = free * mask + fixed
            obj, *_ = objective(calib, fr, t, w_dir)
            (obj.mean() + joint_penalty(t)).backward()
            opt.step()
        with torch.no_grad():
            t = free * mask + fixed
            obj, d_tip, d_fl, ang = objective(calib, fr, t, w_dir)
            rows.append({"dip_deg": float(dip), "objective_mm2": round(float(obj.mean()), 3),
                         "tip_rms_mm": round(float(torch.sqrt((d_tip[fr.tip_mask] ** 2).mean())), 3),
                         "float_prox_rms_mm": round(float(torch.sqrt((d_fl[:, 0][fr.float_mask[:, 0]] ** 2).mean())), 3),
                         "float_mid_rms_mm": round(float(torch.sqrt((d_fl[:, 1][fr.float_mask[:, 1]] ** 2).mean())), 3),
                         "distal_dir_rms_deg": round(float(torch.sqrt((ang ** 2).mean())), 3),
                         "pip_mean_deg": round(float(torch.rad2deg(t[:, 5]).mean()), 1),
                         "mcp_mean_deg": round(float(torch.rad2deg(t[:, 2]).mean()), 1)})
        print(rows[-1], flush=True)
    out = {"source": "scripts/dip_profile.py", "calibration": f"hand_calib{calib_sfx(cfg)}.json",
           "frames": len(sel), "tracked_total_flexion_deg_mean": round(float(total[np.argsort(-total)[: args.frames]].mean()), 1),
           "tracked_dip_deg_mean": round(float(tracked_dip.mean()), 1),
           "tracked_pip_deg_mean": round(float(np.degrees(th[sel, 5]).mean()), 1),
           "floating_placement": place, "index_joint_offset_mm": calib.index_joint_offset.detach().cpu().numpy() * 1000.0,
           "profile": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "profile"}, indent=1, default=lambda o: np.round(o, 2).tolist()))
    write_json(ROOT / "notes" / f"2026-10-27_dip_profile{calib_sfx(cfg)}.json", out)


if __name__ == "__main__":
    main()
