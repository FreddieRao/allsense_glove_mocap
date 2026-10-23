#!/usr/bin/env python3
"""Stage D: the joint fit -- MANO betas, both cap bindings and both floating-marker skin points, from both takes.

THE MODEL (calibration/hand_model.py), per frame i of either take:

    T_world_from_canonical[i] = T_world_from_knucklecap[i] @ T_knucklecap_from_canonical
    tip cap row r  = T_world_from_canonical[i] @ T_canonical_from_joint[3](beta, theta_i) @ T_distal_from_tipcap @ x_r
    floating m     = T_world_from_canonical[i] @ skin(face_m, bary_m, h_m; beta, theta_i)

SHARED UNKNOWNS: beta (10), T_knucklecap_from_canonical (6), T_distal_from_tipcap (6), per floating marker a
skin point (face + 2 barycentric) and its height h above the skin, and T_fixture_from_cad (6, prior only).
PER-POSE UNKNOWNS: theta (9, the index joints) for each of the calibration frames -- all pulled toward the
flat hand, since the hand rests open in the fixture -- and for each of the K free_motion frames chosen by farthest-point sampling on the tip
cap's pose in the knuckle cap's frame, so the fit sees the whole range of motion rather than its mode.

WHY BOTH TAKES. Free motion alone has a degeneracy that TableMocap's finger-bend notes measured on the glove:
a cap's mounting rotation on the distal phalanx and the PIP/DIP flexion trade off, since only their product
reaches the cap. Here two things break it: the floating markers on the proximal and middle phalanges see
those bones directly, and the calibration take holds the index at a KNOWN posture (open palm, theta_cal pulled
to the flat hand by `w.cal_pose`) while the same bindings must explain it.

THE TERMS (mm^2 units, Geman-McClure at `robust_mm`):
    tip      every tip-cap marker, both takes
    float    every floating-marker observation, both takes
    joint    index joint ranges (hand_model.JOINT_RANGE_DEG)
    cal_pose theta_cal toward the flat hand
    betas    toward custom_pose.json's betas: only the index and palm are observed, so the other fingers'
             shape must be held by this prior
    height   each floating marker's height inside [h_min, h_max] (a 3 mm-radius marker on its base)
    fixture  where T_knucklecap_from_canonical puts the hand in the fixture vs the designed
             `hand_aligned_mm.obj` placement through T_fixture_from_cad. The registration is CERTAIN since
             Stage C fits it plane-constrained (0.20 mm rms, next basin 0.73 mm) and is held fixed
             (`fit.freeze_fixture_cad`); the designed hand is both the START of every binding (init_rig) and this
             soft prior (`fixture_sigma_*`), which leaves the hand free to sit a few mm off the design.

THE SCHEDULE (as TableMocap solve_plate_binding._one_start: betas and the bindings are never free together):
poses -> bindings+floats+poses -> betas+poses -> bindings -> betas -> polish (everything, small steps).
The floating markers are re-bound to their nearest face every `rebind_every` iterations.

    ~/miniforge3/envs/far_grasp/bin/python -m calibration.fit_hand
Writes outputs/calibration/hand_calib.json and outputs/calibration/hand_calib.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import (ROOT, T_cad_from_canonical, T_from_json, T_json, apply, inv_se3,       # noqa: E402
                                load_rig, rot_deg, write_json)
from calibration.hand_model import (FLOAT_LABELS, Calib, Delta, init_index_pose, joint_penalty,         # noqa: E402
                                    make_frames, robust_mm, valid_frames)
from calibration.mano_index import INDEX_TIP_VERTEX, ManoIndex                                        # noqa: E402
from config import load_config, strip_overrides                                                      # noqa: E402


def holdout_mask(n: int, fc) -> np.ndarray:
    """(n,) True on the held-out time blocks: `n_blocks` contiguous blocks, `holdout_blocks` of them out."""
    blk = np.minimum((np.arange(n) * int(fc.n_blocks)) // n, int(fc.n_blocks) - 1)
    return np.isin(blk, list(fc.holdout_blocks))


def fps_frames(rig, cand: np.ndarray, K: int, lever_mm: float) -> np.ndarray:
    """Farthest-point sampling of K frames on the tip cap's pose in the knuckle frame (mm; rotation as a
    rotation vector times `lever_mm`)."""
    from scipy.spatial.transform import Rotation
    kn, tp = rig.bodies["knuckle"], rig.bodies["tip"]
    R = np.einsum("tji,tjk->tik", kn.R_world_from_body_seq[cand], tp.R_world_from_body_seq[cand])
    t = np.einsum("tji,tj->ti", kn.R_world_from_body_seq[cand], tp.t_world_from_body_seq[cand] - kn.t_world_from_body_seq[cand])
    X = np.c_[t * 1000.0, Rotation.from_matrix(R).as_rotvec() * lever_mm]
    sel = [int(np.argmin(np.linalg.norm(X - np.median(X, 0), axis=1)))]
    d = np.linalg.norm(X - X[sel[0]], axis=1)
    for _ in range(min(K, len(cand)) - 1):
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return np.sort(cand[np.array(sel)])


class FixturePrior:
    """Soft placement of the hand in the fixture through the (ambiguous) fixture->CAD registration."""

    def __init__(self, cfg, init: dict, rig_cal, device):
        from calibration.init_rig import cad_surface
        self.device = device
        T_cad_can, _ = T_cad_from_canonical(cfg)
        self.T_cad_can = torch.as_tensor(T_cad_can, dtype=torch.float32, device=device)
        self.T_fix_kn = torch.as_tensor(T_from_json(init["T_fixture_from_knucklecap"]), dtype=torch.float32,
                                        device=device)
        self.fix_from_cad = Delta(T_from_json(init["T_fixture_from_cad"]), device)
        from calibration.common import fixture_markers_for_cad
        self.X_fix, self.radius, _src = fixture_markers_for_cad(cfg, rig_cal)
        self.pts, self.tree = cad_surface(cfg, int(cfg.init.cad_samples))
        self.target_cad = None

    def refresh(self) -> float:
        """Recompute each fixture marker's offset-surface target (ICP correspondence step). rms mm."""
        T = self.fix_from_cad.numpy()
        Y = apply(inv_se3(T), self.X_fix)
        d, j = self.tree.query(Y)
        s = self.pts[j]
        u = (Y - s) / np.maximum(d, 1e-9)[:, None]
        self.target_cad = torch.as_tensor(s + u * self.radius[:, None], dtype=torch.float32, device=self.device)
        return float(np.sqrt(((d - self.radius) ** 2).mean())) * 1000.0

    def loss(self, T_kn_can: torch.Tensor, sig_t_mm: float, sig_r_deg: float) -> tuple[torch.Tensor, dict]:
        T_fc = self.fix_from_cad()
        T_meas = self.T_fix_kn @ T_kn_can                    # fixture <- canonical, through the knuckle cap
        T_des = T_fc @ self.T_cad_can                        # fixture <- canonical, as designed in the CAD
        dt = (T_meas[:3, 3] - T_des[:3, 3]).norm() * 1000.0
        c = ((T_meas[:3, :3].T @ T_des[:3, :3]).diagonal().sum() - 1.0) / 2.0
        dr = torch.rad2deg(torch.arccos(c.clamp(-1 + 1e-7, 1 - 1e-7)))
        Xc = torch.as_tensor(self.X_fix, dtype=torch.float32, device=self.device)
        pc = (T_fc[:3, :3].T @ (Xc - T_fc[:3, 3]).T).T       # fixture markers in the CAD frame
        e_icp = ((pc - self.target_cad).norm(dim=1) * 1000.0) ** 2
        return (dt / sig_t_mm) ** 2 + (dr / sig_r_deg) ** 2 + e_icp.mean(), {"dt_mm": float(dt), "dr_deg": float(dr)}


def fit(cfg, tag: str = "") -> dict:
    fc = cfg.fit
    w = fc.weights
    torch.manual_seed(0)
    init = json.loads((ROOT / cfg.data.out_root / "calibration" / "init_rig.json").read_text())
    init_np = {**init, "T_knucklecap_from_canonical": T_from_json(init["T_knucklecap_from_canonical"]),
               "T_distal_from_tipcap": T_from_json(init["T_distal_from_tipcap"])}
    if fc.betas_init:
        # another shape as the start and prior (e.g. the hand-scan MANO fit), optionally held fixed
        init_np["betas"] = np.load(ROOT / fc.betas_init)["betas"].reshape(-1).tolist()
        print(f"betas from {fc.betas_init}{' (frozen)' if fc.freeze_betas else ''}", flush=True)
    rig_cal, rig_free = load_rig("calibration", cfg), load_rig("free_motion", cfg)
    mano = ManoIndex(str(cfg.mano.model_dir))
    dev = mano.device
    base_pose = np.asarray(init["hand_pose"], float)
    init_np["joint_offset_radius_m"] = float(fc.joint_offset_mm) / 1000.0
    calib = Calib(mano, init_np, rig_cal.bodies["tip"].template_body, base_pose).to(dev)

    # ---- frames
    cal_ok = valid_frames(rig_cal)
    cal_idx = cal_ok[np.linspace(0, len(cal_ok) - 1, int(fc.n_cal_frames)).astype(int)]
    free_ok = valid_frames(rig_free)
    held = holdout_mask(rig_free.n_frames, fc)
    free_idx = fps_frames(rig_free, free_ok[~held[free_ok]], int(fc.n_free_frames), float(fc.fps_lever_mm))
    F_cal, F_free = make_frames(rig_cal, cal_idx, dev), make_frames(rig_free, free_idx, dev)
    print(f"frames: calibration {len(cal_idx)}, free_motion {len(free_idx)} (FPS over {int((~held[free_ok]).sum())} "
          f"non-held-out valid frames)", flush=True)

    # ---- poses: theta_cal from the init (flat index), free frames from the cap-to-cap flexion
    from calibration.common import procrustes
    kn, tp = rig_cal.bodies["knuckle"], rig_cal.bodies["tip"]
    R_kt = np.einsum("tji,tjk->tik", kn.R_world_from_body_seq[cal_idx], tp.R_world_from_body_seq[cal_idx])
    from calibration.common import rot_mean
    R_kn_tip_ref = rot_mean(R_kt)
    # one index pose PER calibration frame, each pulled to the flat hand: the finger is not perfectly still in
    # the fixture (Stage C: tip cap slip p50 1.2 deg / p95 2.8 deg relative to the fixture), and one shared
    # pose turned that slip into a 2.25 mm tip-cap RMS on the first fit
    theta_cal = torch.nn.Parameter(torch.as_tensor(base_pose[:9], dtype=torch.float32, device=dev)[None]
                                   .repeat(len(cal_idx), 1) + init_index_pose(calib, F_cal, R_kn_tip_ref))
    theta_free = torch.nn.Parameter(init_index_pose(calib, F_free, R_kn_tip_ref) + theta_cal.detach().mean(0))
    fix = FixturePrior(cfg, init, rig_cal, dev) if float(w.fixture) > 0 else None

    def objective(log: bool = False):
        pc = calib.predict(F_cal, theta_cal)
        pf = calib.predict(F_free, theta_free)
        s = float(fc.robust_scale_mm)
        terms = {
            "tip": 0.5 * (robust_mm(pc["tip"] - F_cal.tip_obs, F_cal.tip_mask, s)
                          + robust_mm(pf["tip"] - F_free.tip_obs, F_free.tip_mask, s)),
            "float": 0.5 * (robust_mm(pc["float"] - F_cal.float_obs, F_cal.float_mask, s)
                            + robust_mm(pf["float"] - F_free.float_obs, F_free.float_mask, s)),
            "joint": joint_penalty(torch.cat([theta_cal, theta_free])),
            "cal_pose": (torch.rad2deg(theta_cal) ** 2).mean(),
            "betas": ((calib.betas - calib.betas0) ** 2).sum(),
            "joint_offset": ((calib.index_joint_offset * 1000.0) ** 2).sum(),
            "height": ((torch.relu(float(fc.h_min_mm) - calib.float_h * 1000.0) ** 2)
                       + (torch.relu(calib.float_h * 1000.0 - float(fc.h_max_mm)) ** 2)).sum(),
        }
        extra = {}
        if fix is not None:
            terms["fixture"], extra = fix.loss(calib.T_knucklecap_from_canonical(), float(fc.fixture_sigma_t_mm),
                                               float(fc.fixture_sigma_r_deg))
        total = sum(float(getattr(w, k)) * v for k, v in terms.items())
        return total, terms, extra

    groups = {
        "pose": [theta_cal, theta_free],
        "binding_r": [calib.T_knucklecap_from_canonical.w, calib.T_distal_from_tipcap.w],
        "binding_t": [calib.T_knucklecap_from_canonical.v, calib.T_distal_from_tipcap.v],
        "float_uv": [calib.float_uv], "float_h": [calib.float_h],
        "betas": [] if fc.freeze_betas else [calib.betas],
        "joint_offset": [calib.joint_offset_u] if float(fc.joint_offset_mm) > 0 else [],
        # the plane-constrained registration (init_rig) is certain to 0.2 mm, so it is held unless asked otherwise
        "fixture": list(fix.fix_from_cad.parameters()) if fix is not None and not fc.freeze_fixture_cad else [],
    }
    t0 = time.perf_counter()
    history = []
    best = None                                   # (total, calib state, theta_cal, theta_free) at a phase end

    def rebind(opt) -> None:
        # Re-attaching a floating marker to another face keeps its POINT but changes what float_uv / float_h
        # mean, so their Adam moments are stale: left in place they kicked the float term 0.53 -> 1.08 mm^2 in
        # the polish phase on 2026-10-22. Drop them.
        calib.rebind_floats(theta_cal.detach().mean(0, keepdim=True))
        for prm in (calib.float_uv, calib.float_h):
            opt.state.pop(prm, None)

    for ph in fc.schedule:
        ph = ph.to_dict() if hasattr(ph, "to_dict") else dict(ph)
        lr = ph["lr"]
        opt = torch.optim.Adam([{"params": groups[g], "lr": float(v)} for g, v in lr.items() if groups[g]])
        for it in range(int(ph["iters"])):
            if fix is not None and it % 50 == 0:
                fix.refresh()
            if it and it % int(fc.rebind_every) == 0 and "float_uv" in lr:
                rebind(opt)
            opt.zero_grad()
            total, terms, extra = objective()
            total.backward()
            opt.step()
        if "float_uv" in lr:
            rebind(opt)
        with torch.no_grad():
            total, terms, extra = objective()
        row = {"phase": ph["name"], "total": round(float(total), 4),
               **{k: round(float(v), 4) for k, v in terms.items()}, **{k: round(v, 2) for k, v in extra.items()}}
        history.append(row)
        print(f"[{time.perf_counter() - t0:6.1f}s] {ph['name']:10s} " +
              " ".join(f"{k}={v}" for k, v in row.items() if k != "phase"), flush=True)
        if best is None or float(total) < best[0]:
            best = (float(total), {k: v.clone() for k, v in calib.state_dict().items()},
                    theta_cal.detach().clone(), theta_free.detach().clone(), ph["name"])
    # keep the best phase end: a later phase that raised the objective is not the answer
    calib.load_state_dict(best[1])
    with torch.no_grad():
        theta_cal.copy_(best[2])
        theta_free.copy_(best[3])
    print(f"kept phase '{best[4]}' (total {best[0]:.4f})", flush=True)
    history.append({"phase": "kept", "kept": best[4], "total": round(best[0], 4)})

    # ---- report on the fit frames
    with torch.no_grad():
        pc = calib.predict(F_cal, theta_cal)
        pf = calib.predict(F_free, theta_free)

    def rms(pred, obs, mask):
        e = ((pred - obs).norm(dim=-1) * 1000.0)[mask].cpu().numpy()
        return {"rms_mm": float(np.sqrt((e ** 2).mean())), "p50_mm": float(np.median(e)),
                "p90_mm": float(np.percentile(e, 90)), "n": int(len(e))}

    fit_err = {}
    for name, p, F in (("calibration", pc, F_cal), ("free_motion", pf, F_free)):
        fit_err[name] = {"tip_cap": rms(p["tip"], F.tip_obs, F.tip_mask)}
        for m, L in enumerate(FLOAT_LABELS):
            fit_err[name][f"float_{L}"] = rms(p["float"][:, m], F.float_obs[:, m], F.float_mask[:, m])
    T_kn_can = calib.T_knucklecap_from_canonical.numpy()
    T_dist_tip = calib.T_distal_from_tipcap.numpy()
    betas = calib.betas.detach().cpu().numpy()[0]
    floats = {}
    for m, L in enumerate(FLOAT_LABELS):
        floats[L] = {"face": int(calib.float_face[m]), "bary": calib.float_bary[m].detach().cpu().numpy(),
                     "vertex_nearest": int(mano.faces[calib.float_face[m]][int(torch.argmax(calib.float_bary[m]))]),
                     "height_mm": float(calib.float_h[m]) * 1000.0, "bone": L}
    with torch.no_grad():
        z9 = torch.zeros(1, 9, device=dev)
        P = mano.forward(calib.betas, calib.full_pose(z9))
        from calibration.mano_index import skin_point
        fl_can = skin_point(P, mano.faces, calib.float_face, calib.float_bary, calib.float_h)[0].cpu().numpy()
        J = P.T_canonical_from_joint[0, :, :3, 3].cpu().numpy()
        for m, L in enumerate(FLOAT_LABELS):
            b = 1 if L == "proximal" else 2
            seg = J[b + 1] - J[b]
            floats[L]["along_bone_frac"] = float((fl_can[m] - J[b]) @ seg / (seg @ seg))
            floats[L]["p_canonical_flat_mm"] = fl_can[m] * 1000.0
    fixture_rep = None
    if fix is not None:
        with torch.no_grad():
            _t, _terms, fx_extra = objective()
        fixture_rep = {"T_fixture_from_cad": T_json(fix.fix_from_cad.numpy()), "icp_rms_mm": fix.refresh(),
                       "frozen": bool(fc.freeze_fixture_cad), "hand_vs_design": fx_extra}
    T_init_kn = init_np["T_knucklecap_from_canonical"]
    T_init_dt = init_np["T_distal_from_tipcap"]
    out = {
        "stage": "D fit_hand",
        "model": ["T_world_from_canonical = T_world_from_knucklecap @ T_knucklecap_from_canonical",
                  "tip_cap_marker = T_world_from_canonical @ T_canonical_from_joint[3](betas, theta) @ "
                  "T_distal_from_tipcap @ x_tipcap",
                  "floating_marker = T_world_from_canonical @ (sum_i bary_i v_i(betas, theta) + height * n)",
                  "MANO: smplx right, mano_clean, flat_hand_mean=True, hand_pose 45-D axis-angle, metres"],
        "index_joint_offset_mm": {j: (calib.index_joint_offset[k].detach().cpu().numpy() * 1000.0).round(3)
                                  for k, j in enumerate(("mcp", "pip", "dip"))},
        "index_joint_offset_m": calib.index_joint_offset.detach().cpu().numpy(),
        "joint_offset_radius_mm": float(fc.joint_offset_mm),
        "betas": betas, "betas_init": init_np["betas"], "betas_init_source": fc.betas_init or cfg.data.mano_pose_json,
        "betas_frozen": bool(fc.freeze_betas),
        "index_bone_lengths_mm": mano.index_bone_lengths_mm(calib.betas.detach()),
        "index_bone_lengths_mm_init": mano.index_bone_lengths_mm(calib.betas0),
        "T_knucklecap_from_canonical": T_json(T_kn_can),
        "T_distal_from_tipcap": T_json(T_dist_tip),
        "T_canonical_from_knucklecap": T_json(inv_se3(T_kn_can)),
        "T_tipcap_from_distal": T_json(inv_se3(T_dist_tip)),
        "binding_moved_from_init": {
            "knuckle_mm": float(np.linalg.norm(T_kn_can[:3, 3] - T_init_kn[:3, 3]) * 1000),
            "knuckle_deg": float(rot_deg(T_kn_can[:3, :3], T_init_kn[:3, :3])),
            "tip_mm": float(np.linalg.norm(T_dist_tip[:3, 3] - T_init_dt[:3, 3]) * 1000),
            "tip_deg": float(rot_deg(T_dist_tip[:3, :3], T_init_dt[:3, :3]))},
        "floating": floats,
        "theta_cal_index_deg_mean": np.degrees(theta_cal.detach().cpu().numpy().mean(0)).reshape(3, 3),
        "theta_cal_index_deg_std": np.degrees(theta_cal.detach().cpu().numpy().std(0)).reshape(3, 3),
        "base_hand_pose": base_pose,
        "fixture": fixture_rep,
        "fit_frames": {"calibration": len(cal_idx), "free_motion": len(free_idx),
                       "free_motion_holdout_blocks": list(fc.holdout_blocks), "n_blocks": int(fc.n_blocks)},
        "fit_error": fit_err, "history": history,
        "tip_vertex": INDEX_TIP_VERTEX,
    }
    out_dir = ROOT / cfg.data.out_root / "calibration"
    stem = f"hand_calib{'_' + tag if tag else ''}"
    write_json(out_dir / f"{stem}.json", out)
    np.savez_compressed(out_dir / f"{stem}.npz", cal_idx=cal_idx, free_idx=free_idx,
                        theta_cal=theta_cal.detach().cpu().numpy(), theta_free=theta_free.detach().cpu().numpy())
    print(f"wrote {out_dir / (stem + '.npz')}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="", help="write hand_calib_<tag>.* (ablations); default run.tag")
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    from calibration.common import sfx
    out = fit(cfg, args.tag or sfx(cfg).lstrip("_"))
    print(json.dumps({"fit_error": out["fit_error"], "betas": out["betas"], "bones": out["index_bone_lengths_mm"],
                      "floating": {k: {kk: v[kk] for kk in ("face", "height_mm", "along_bone_frac")}
                                   for k, v in out["floating"].items()},
                      "moved": out["binding_moved_from_init"]},
                     indent=1, default=lambda o: np.round(o, 4).tolist()), flush=True)


if __name__ == "__main__":
    main()
