#!/usr/bin/env python3
"""Stage E: every frame of a take tracked with the FROZEN calibration -- the index pose per frame, nothing else.

With betas, T_knucklecap_from_canonical, T_distal_from_tipcap and the floating skin points fixed by Stage D,
each frame has 9 unknowns (MANO index joints 1-3) and up to 5 tip-cap markers + 2 floating markers to explain
(hand_model.py). Frames are solved in chunks with a temporal smoothness term between consecutive frames.

TWO PASSES, BECAUSE THE FLOATING LABELS ARE ONLY AS GOOD AS THEIR DISTANCE WINDOWS. Stage B named the two
phalanx markers by distance windows to the caps (mosca/floating.py), which says "on this finger" and nothing
sharper. Pass 1 solves the pose from the tip cap alone, which predicts where each floating marker must be;
the labels are then re-claimed per frame by Hungarian matching within `reclaim_gate_mm` of that prediction
(the scaffold's `recover`, with MANO as the template) among the observations no rigid body owns; pass 2
solves again with tip cap + re-claimed floating markers. The report counts how many labels changed.

    ~/miniforge3/envs/far_grasp/bin/python -m calibration.track --take calibration free_motion
Writes outputs/<take>/mano_track[_<run.tag>].npz.
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

from calibration.common import ROOT, T_from_json, calib_sfx, load_rig, rot_mean, sfx, write_json                  # noqa: E402
from calibration.hand_model import (Calib, init_index_pose, joint_penalty, make_frames, robust_mm,  # noqa: E402
                                    valid_frames)
from calibration.mano_index import INDEX_TIP_VERTEX, ManoIndex                                    # noqa: E402
from config import load_config, strip_overrides                                                  # noqa: E402
from mosca import floating                                                                       # noqa: E402


def load_calib(cfg, mano: ManoIndex, rig_cal) -> Calib:
    """Stage D's calibration as a frozen `Calib`."""
    d = json.loads((ROOT / cfg.data.out_root / "calibration" / f"hand_calib{calib_sfx(cfg)}.json").read_text())
    init = {"betas": d["betas"], "T_knucklecap_from_canonical": T_from_json(d["T_knucklecap_from_canonical"]),
            "T_distal_from_tipcap": T_from_json(d["T_distal_from_tipcap"]),
            "floating": {L: {"face": v["face"], "bary": v["bary"], "height_mm": v["height_mm"]}
                         for L, v in d["floating"].items()},
            "joint_offset_radius_m": float(d.get("joint_offset_radius_mm", 0.0)) / 1000.0}
    if d.get("joint_offset_radius_mm", 0.0) > 0:
        init["index_joint_offset_m"] = d["index_joint_offset_m"]
    calib = Calib(mano, init, rig_cal.bodies["tip"].template_body, np.asarray(d["base_hand_pose"]))
    for p in calib.parameters():
        p.requires_grad_(False)
    return calib


def smooth_wrist(rig, calib: Calib, cfg) -> dict:
    """Smooth the fitted MANO root AT THE WRIST and re-derive the knuckle cap pose from it (in place).

        T_world_from_canonical = T_world_from_knucklecap @ T_knucklecap_from_canonical
        wrist_world            = R_world_from_canonical @ J_wrist + t_world_from_canonical

    The wrist position and the root rotation are smoothed (`smooth.wrist`), then t_world_from_canonical =
    wrist_s - R_s J_wrist and T_world_from_knucklecap = T_world_from_canonical_s @ inv(T_knucklecap_from_canonical).
    Smoothing the cap's own position instead would leave the rotation jitter x the cap->wrist lever in the
    wrist; this filters the wrist trajectory itself. Returns the wrist jitter before/after (mm, about 54 ms)."""
    from types import SimpleNamespace
    from mosca.smooth import smooth_body
    from scripts.rb_jitter import jitter
    kn = rig.bodies["knuckle"]
    T_kc = calib.T_knucklecap_from_canonical.numpy()
    with torch.no_grad():
        J_w = calib.mano.rest_joints(calib.betas)[0, 0].cpu().double().numpy()     # wrist, canonical (no offset: joint 0)
    R_wc = np.einsum("tij,jk->tik", kn.R_world_from_body_seq, T_kc[:3, :3])
    t_wc = np.einsum("tij,j->ti", kn.R_world_from_body_seq, T_kc[:3, 3]) + kn.t_world_from_body_seq
    wrist = np.einsum("tij,j->ti", R_wc, J_w) + t_wc
    wc = cfg.smooth.wrist
    bw = float(wc.bandwidth_hz)
    sc = SimpleNamespace(butter=SimpleNamespace(cutoff_hz=bw, order=int(cfg.smooth.butter.order)),
                         rts=SimpleNamespace(bandwidth_hz=bw, floor_mm=float(cfg.smooth.rts.floor_mm),
                                             max_gap_s=float(cfg.smooth.rts.max_gap_s)))
    v = kn.valid
    R_s, w_s = smooth_body(str(wc.method), R_wc, wrist * 1000.0, v, kn.rmse_mm, np.full(len(v), 5), kn.template_body * 1000.0,
                           rig.rate_hz, sc)
    w_s = w_s / 1000.0
    t_s = w_s - np.einsum("tij,j->ti", R_s, J_w)
    T_ck = np.linalg.inv(T_kc)
    before = jitter(R_wc, wrist * 1000.0, v, 6)
    after = jitter(R_s, w_s * 1000.0, v, 6)
    kn.R_world_from_body_seq = np.where(v[:, None, None], np.einsum("tij,jk->tik", R_s, T_ck[:3, :3]), kn.R_world_from_body_seq)
    kn.t_world_from_body_seq = np.where(v[:, None], np.einsum("tij,j->ti", R_s, T_ck[:3, 3]) + t_s, kn.t_world_from_body_seq)
    return {"method": str(wc.method), "bandwidth_hz": bw,
            "wrist_jitter_mm_rms_before_after": [before["trans_mm_rms_p50_p95"][0], after["trans_mm_rms_p50_p95"][0]],
            "root_rot_jitter_deg_rms_before_after": [before["rot_deg_rms_p50_p95"][0], after["rot_deg_rms_p50_p95"][0]]}


def solve_chunk(calib: Calib, fr, theta0: torch.Tensor, tc, use_float: bool, iters: int) -> torch.Tensor:
    theta = torch.nn.Parameter(theta0.clone())
    opt = torch.optim.Adam([theta], lr=float(tc.lr))
    consec = torch.as_tensor(np.diff(fr.idx) == 1, device=theta.device)
    for _ in range(iters):
        opt.zero_grad()
        p = calib.predict(fr, theta)
        loss = robust_mm(p["tip"] - fr.tip_obs, fr.tip_mask, 5.0) + joint_penalty(theta)
        if use_float:
            loss = loss + robust_mm(p["float"] - fr.float_obs, fr.float_mask, 5.0)
        d = torch.rad2deg(theta[1:] - theta[:-1])
        loss = loss + float(tc.smooth) * ((d ** 2).sum(1) * consec).sum() / consec.sum().clamp_min(1)
        loss.backward()
        opt.step()
    return theta.detach()


def track(take: str, cfg) -> dict:
    tc = cfg.track
    rig = load_rig(take, cfg)
    rig_cal = rig if take == "calibration" else load_rig("calibration", cfg)
    mano = ManoIndex(str(cfg.mano.model_dir))
    dev = mano.device
    calib = load_calib(cfg, mano, rig_cal)
    wrist_rep = smooth_wrist(rig, calib, cfg) if bool(cfg.smooth.wrist.enable) else None
    if wrist_rep:
        print(f"[{take}] wrist smoothing {wrist_rep}", flush=True)
    kn, tp = rig_cal.bodies["knuckle"], rig_cal.bodies["tip"]
    ok_cal = valid_frames(rig_cal)
    R_ref = rot_mean(np.einsum("tji,tjk->tik", kn.R_world_from_body_seq[ok_cal], tp.R_world_from_body_seq[ok_cal]))
    base9 = torch.as_tensor(calib.base_pose[0, :9])
    idx_all = valid_frames(rig)
    T = rig.n_frames
    theta_seq = np.full((T, 9), np.nan, np.float32)
    t0 = time.perf_counter()

    def run_pass(use_float: bool, theta_prev: np.ndarray | None):
        for a in range(0, len(idx_all), int(tc.chunk)):
            idx = idx_all[a:a + int(tc.chunk)]
            fr = make_frames(rig, idx, dev)
            th0 = (torch.as_tensor(theta_prev[idx], device=dev) if theta_prev is not None
                   else init_index_pose(calib, fr, R_ref) + base9.to(dev))
            th = solve_chunk(calib, fr, th0, tc, use_float, int(tc.iters_full if use_float else tc.iters_tip))
            theta_seq[idx] = th.cpu().numpy()

    run_pass(False, None)
    print(f"[{take}] pass 1 (tip cap) {time.perf_counter() - t0:.1f}s", flush=True)

    # re-claim the floating labels against the pass-1 prediction
    pred_fl = np.full((2, T, 3), np.nan)
    for a in range(0, len(idx_all), int(tc.chunk)):
        idx = idx_all[a:a + int(tc.chunk)]
        with torch.no_grad():
            p = calib.predict(make_frames(rig, idx, dev), torch.as_tensor(theta_seq[idx], device=dev))
        pred_fl[:, idx] = np.swapaxes(p["float"].cpu().numpy(), 0, 1)
    z = np.load(ROOT / cfg.data.out_root / take / "mosca.npz")
    new_lab, rrep = floating.reclaim(z["frame"], z["p_world_mm"].astype(float), T, z["float_label"],
                                     pred_fl * 1000.0, float(tc.reclaim_gate_mm), z["body"] >= 0)
    fl_world = np.full((2, T, 3), np.nan)
    for j in range(2):
        i = np.flatnonzero(new_lab == j)
        fl_world[j, z["frame"][i]] = z["p_world_mm"][i] / 1000.0
    from calibration.common import smooth_floats
    fl_world_raw = fl_world
    fl_world = smooth_floats(fl_world_raw, rig.rate_hz, cfg)          # the re-claimed tracks, smoothed if asked
    rig.float_world_seq = fl_world
    print(f"[{take}] re-claim {rrep}", flush=True)

    run_pass(True, theta_seq.copy())
    print(f"[{take}] pass 2 (tip + floating) {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- outputs for every valid frame
    out = {"tip_pred": np.full((T, 5, 3), np.nan, np.float32), "float_pred": np.full((T, 2, 3), np.nan, np.float32),
           "T_world_from_canonical": np.full((T, 4, 4), np.nan, np.float32),
           "joints_world": np.full((T, 16, 3), np.nan, np.float32),
           "tip_vertex_world": np.full((T, 3), np.nan, np.float32)}
    for a in range(0, len(idx_all), int(tc.chunk)):
        idx = idx_all[a:a + int(tc.chunk)]
        with torch.no_grad():
            p = calib.predict(make_frames(rig, idx, dev), torch.as_tensor(theta_seq[idx], device=dev), verts=True)
        out["tip_pred"][idx] = p["tip"].cpu().numpy()
        out["float_pred"][idx] = p["float"].cpu().numpy()
        out["T_world_from_canonical"][idx] = p["T_world_from_canonical"].cpu().numpy()
        out["joints_world"][idx] = p["T_world_from_joint"][:, :, :3, 3].cpu().numpy()
        out["tip_vertex_world"][idx] = p["vertices"][:, INDEX_TIP_VERTEX].cpu().numpy()
    out_dir = ROOT / cfg.data.out_root / take
    np.savez_compressed(out_dir / f"mano_track{sfx(cfg)}.npz", index_pose=theta_seq, valid=np.isfinite(theta_seq[:, 0]),
                        tip_obs=rig.bodies["tip"].marker_world_seq.astype(np.float32),
                        knuckle_obs=rig.bodies["knuckle"].marker_world_seq.astype(np.float32),
                        float_obs=np.swapaxes(fl_world, 0, 1).astype(np.float32), float_label=new_lab.astype(np.int8),
                        tip_obs_raw=rig.bodies["tip"].marker_world_raw_seq.astype(np.float32),
                        knuckle_obs_raw=rig.bodies["knuckle"].marker_world_raw_seq.astype(np.float32),
                        float_obs_raw=np.swapaxes(fl_world_raw, 0, 1).astype(np.float32),
                        **out)
    print(f"wrote {out_dir / f'mano_track{sfx(cfg)}.npz'}", flush=True)
    rep = {"take": take, "frames_tracked": int(len(idx_all)), "seconds": round(time.perf_counter() - t0, 1),
           "reclaim": rrep, "wrist_smoothing": wrist_rep}
    write_json(out_dir / f"mano_track{sfx(cfg)}.json", rep)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--take", nargs="+", default=["calibration", "free_motion"])
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    for t in args.take:
        track(t, cfg)


if __name__ == "__main__":
    main()
