#!/usr/bin/env python3
"""Stage C: the rigid calibration take -> the fixture's place in the CAD frame and every initial binding.

During calibration_1018 the open hand rests in the printed fixture, so every relative pose is (near)
constant, and the chain that places MANO in the world is

    T_world_from_canonical = T_world_from_fixture @ T_fixture_from_cad @ T_cad_from_canonical

where
    T_cad_from_canonical   is GIVEN: `hand_aligned_mm.obj` is `custom_pose_metric.obj` moved into the CAD
                           frame (one Kabsch on the 1538x3 corresponding vertices, residual 0.0 mm);
    T_fixture_from_cad     is SOLVED here: the fixture CAD has no marker positions, so the 18 fixture
                           marker centres (Motive body frame, the `.motive` asset's DEFINED offsets -- not
                           refined on the calibration take, `init.fixture_template`) are registered onto
                           the CAD surface offset by their radius -- point-to-offset-surface ICP, unsigned
                           (the CAD's normals are not consistently oriented).

EVERY FIXTURE MARKER IS SNAPPED ON TOP OR AT THE SIDE. In the asset, markers 8-17 are coplanar to 0.06 mm: the
TOP markers, resting on the CAD's largest upward-facing level (z = 35 mm). The other 8 sit 7-25 mm below that
plane: the SIDE markers, stuck to walls. So:
  1. the ten top centres go one radius above the top face, which fixes tilt and height exactly;
  2. yaw and x/y are searched from a start every `init.flat.yaw_step_deg` with a planar ICP over all 18 markers on
     any surface; the best basin is checked to have every top marker on a +z face and every side marker on a wall;
  3. a weighted 6-dof polish puts each marker on its own surface class -- top face or walls (|n_z| <=
     `init.flat.side_nz_max`) -- with the side markers weighted `init.flat.side_weight`: a top marker slides
     along its face at no cost, so the fixture's in-plane DIRECTION is set by the side markers.
Measured 2026-10-22 on the updated hand_fixtured.glb: planar search 0.22 mm rms, next basin 0.73 mm; after the
weighted polish top 0.13 / side 0.27 mm rms, max 0.38 mm. The earlier unconstrained offset-surface ICP found a
family of 1.23-1.8 mm basins 5-17 deg apart with markers on side walls of the wrong kind and the underside.

Then, with the hand placed:

    T_knucklecap_from_canonical = inv(T_fixture_from_knucklecap) @ T_fixture_from_cad @ T_cad_from_canonical
    T_distal_from_tipcap        = inv(T_fixture_from_canonical @ T_canonical_from_joint[3]) @ T_fixture_from_tipcap
    floating marker             = nearest skin point of its phalanx (face, barycentric, height)

at the custom pose. These are STARTS for Stage D, which refines all of them together with the betas.

    ~/miniforge3/envs/far_grasp/bin/python -m calibration.init_rig
Writes outputs/calibration/init_rig.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import (ROOT, T_cad_from_canonical, T_json, apply, fixture_markers_for_cad,  # noqa: E402
                                inv_se3, load_custom_pose, load_rig, procrustes, rot_deg, rot_mean, se3, write_json)
from config import load_config, strip_overrides                                                  # noqa: E402


def rigid_summary(rig, ok_frac: float = 0.98) -> dict:
    """Robust constant relative poses over the calibration take, plus how much each one wanders (slip)."""
    fx, kn, tp = rig.bodies["fixture"], rig.bodies["knuckle"], rig.bodies["tip"]
    ok = fx.valid & kn.valid & tp.valid
    out = {}
    for role, b in (("knuckle", kn), ("tip", tp)):
        R = np.einsum("tji,tjk->tik", fx.R_world_from_body_seq[ok], b.R_world_from_body_seq[ok])
        t = np.einsum("tji,tj->ti", fx.R_world_from_body_seq[ok],
                      b.t_world_from_body_seq[ok] - fx.t_world_from_body_seq[ok])
        Rm, tm = rot_mean(R), np.median(t, 0)
        dev_mm = np.linalg.norm(t - tm, axis=1) * 1000.0
        dev_deg = rot_deg(R, Rm[None])
        out[role] = {"T": se3(Rm, tm), "slip_mm_p50_p95": np.percentile(dev_mm, [50, 95]),
                     "slip_deg_p50_p95": np.percentile(dev_deg, [50, 95])}
    for j, L in enumerate(("proximal", "middle")):
        p = rig.float_world_seq[j]
        okf = ok & np.isfinite(p[:, 0])
        loc = np.einsum("tji,tj->ti", fx.R_world_from_body_seq[okf], p[okf] - fx.t_world_from_body_seq[okf])
        pm = np.median(loc, 0)
        out[L] = {"p": pm, "slip_mm_p50_p95": np.percentile(np.linalg.norm(loc - pm, axis=1) * 1000.0, [50, 95])}
    out["frames"] = int(ok.sum())
    return out


def cad_surface(cfg, n: int):
    """(points (n,3) m, KD-tree) sampled on the fixture CAD surface."""
    import trimesh
    from scipy.spatial import cKDTree
    m = trimesh.load(ROOT / cfg.data.fixture_cad_glb, force="mesh")
    pts, _ = trimesh.sample.sample_surface_even(m, n, seed=0)
    return np.asarray(pts), cKDTree(pts)


def icp_offset(X_fix: np.ndarray, radius: np.ndarray, T0_cad_from_fixture: np.ndarray, tree, pts,
               iters: int) -> tuple[np.ndarray, float]:
    """Marker centres onto the CAD surface offset by their radius. (T_cad_from_fixture, rms mm of |d|-r)."""
    T = T0_cad_from_fixture.copy()
    for _ in range(iters):
        Y = apply(T, X_fix)
        d, j = tree.query(Y)
        s = pts[j]
        u = (Y - s) / np.maximum(d, 1e-9)[:, None]
        target = s + u * radius[:, None]                 # the point one radius off the surface, on Y's side
        R, t = procrustes(X_fix, target)
        T = se3(R, t)
    d, _ = tree.query(apply(T, X_fix))
    return T, float(np.sqrt(((d - radius) ** 2).mean())) * 1000.0


def coplanar_markers(X: np.ndarray, tol: float, min_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(indices, unit normal, centroid) of the largest marker subset within `tol` of one plane.

    The normal points AWAY from the other markers: on the fixture they all sit on lower steps of the CAD, so
    the plane's normal is the CAD's up direction. Raises if fewer than `min_n` markers are coplanar."""
    from itertools import combinations
    best = None
    for a, b, c in combinations(range(len(X)), 3):
        n = np.cross(X[b] - X[a], X[c] - X[a])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.flatnonzero(np.abs((X - X[a]) @ n) <= tol)
        if best is None or len(inl) > len(best):
            best = inl
    if best is None or len(best) < min_n:
        raise ValueError(f"no {min_n} coplanar fixture markers within {tol * 1000:.1f} mm")
    c = X[best].mean(0)
    n = np.linalg.svd(X[best] - c)[2][2]
    rest = np.setdiff1d(np.arange(len(X)), best)
    if len(rest) and ((X[rest] - c) @ n).mean() > 0:
        n = -n
    return best, n, c


def cad_top_level(cfg, nz_min: float = 0.99) -> tuple[float, float]:
    """(z m, area mm^2) of the CAD's largest upward-facing flat level: the face the coplanar markers rest on."""
    import trimesh
    m = trimesh.load(ROOT / cfg.data.fixture_cad_glb, force="mesh")
    up = m.face_normals[:, 2] >= nz_min
    z = np.round(m.triangles_center[up, 2] * 1000.0, 1)
    area = m.area_faces[up] * 1e6
    lv, inv = np.unique(z, return_inverse=True)
    acc = np.bincount(inv, weights=area)
    k = int(np.argmax(acc))
    return float(lv[k]) / 1000.0, float(acc[k])


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The minimal rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def fit_fixture_flat(X: np.ndarray, radius: np.ndarray, flat: np.ndarray, n_fix: np.ndarray, c_fix: np.ndarray,
                     z_face: float, tree, pts, yaw_step_deg: float, iters: int) -> list[tuple[float, np.ndarray]]:
    """Fixture markers onto the CAD with the coplanar ones ON the CAD's top face: [(rms mm, T_cad_from_fixture)],
    distinct basins, best first.

    The plane fixes 3 of the 6 dof exactly -- the marker plane's normal is CAD +z and its centres sit one radius
    above the face at `z_face` -- so only yaw about z and the x/y translation are searched: every `yaw_step_deg`
    start runs a planar ICP (nearest CAD point, target one radius off it, 2D Kabsch on x/y) over ALL markers."""
    R0 = _align(n_fix, np.array([0.0, 0.0, 1.0]))
    z0 = z_face + float(radius[flat].mean())
    sols = []
    for yaw in np.radians(np.arange(0.0, 360.0, yaw_step_deg)):
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        R = Rz @ R0
        t = np.r_[pts[:, :2].mean(0) - ((X[flat] - c_fix) @ R.T)[:, :2].mean(0), z0] - R @ c_fix
        T = se3(R, t)
        for _ in range(iters):
            Y = apply(T, X)
            d, j = tree.query(Y)
            u = (Y - pts[j]) / np.maximum(d, 1e-9)[:, None]
            tgt = pts[j] + u * radius[:, None]
            R2, t2 = procrustes(Y[:, :2], tgt[:, :2])           # 2D: yaw + x/y only, tilt and height kept
            U = np.eye(4)
            U[:2, :2], U[:2, 3] = R2, t2
            T = U @ T
        d, _ = tree.query(apply(T, X))
        sols.append((float(np.sqrt(((d - radius) ** 2).mean())) * 1000.0, T))
    sols.sort(key=lambda s: s[0])
    out = []
    for rms, T in sols:
        if all(rot_deg(T[:3, :3], U[:3, :3]) > 2.0 or np.linalg.norm(T[:3, 3] - U[:3, 3]) > 0.002 for _, U in out):
            out.append((rms, T))
    return out


def cad_surface_classes(cfg, n: int, z_top: float, side_nz_max: float) -> dict:
    """{'all' | 'top' | 'side': (points (k,3) m, KD-tree)} sampled on the fixture CAD.

    A fixture marker is snapped either ON TOP (resting on the top face, z = `z_top`) or AT THE SIDE (stuck to a
    wall), so each marker is matched only against the surface class it can touch: the top face for the
    coplanar markers, the near-vertical faces (|n_z| <= `side_nz_max`) for the others."""
    import trimesh
    from scipy.spatial import cKDTree
    m = trimesh.load(ROOT / cfg.data.fixture_cad_glb, force="mesh")
    pts, fi = trimesh.sample.sample_surface_even(m, n, seed=0)
    pts, nz = np.asarray(pts), m.face_normals[fi, 2]
    top = (nz >= 0.99) & (np.abs(pts[:, 2] - z_top) <= 5e-4)
    side = np.abs(nz) <= side_nz_max
    return {k: (pts[s_], cKDTree(pts[s_])) for k, s_ in (("all", np.ones(len(pts), bool)), ("top", top), ("side", side))}


def _class_targets(Y: np.ndarray, radius: np.ndarray, cls: np.ndarray, surf: dict) -> tuple[np.ndarray, np.ndarray]:
    """(targets (M,3): one radius off the nearest point of each marker's own surface class, gap (M,) m)."""
    tgt, gap = np.zeros_like(Y), np.zeros(len(Y))
    for c in ("top", "side"):
        k = cls == c
        if not k.any():
            continue
        pts, tree = surf[c]
        d, j = tree.query(Y[k])
        u = (Y[k] - pts[j]) / np.maximum(d, 1e-9)[:, None]
        tgt[k], gap[k] = pts[j] + u * radius[k, None], d - radius[k]
    return tgt, gap


def polish_topside(X: np.ndarray, radius: np.ndarray, cls: np.ndarray, w: np.ndarray, T0: np.ndarray, surf: dict,
                   iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted 6-dof ICP, every marker on its own surface class. (T_cad_from_fixture, gap (M,) m)."""
    T = T0.copy()
    for _ in range(iters):
        tgt, _g = _class_targets(apply(T, X), radius, cls, surf)
        R, t = procrustes(X, tgt, w)
        T = se3(R, t)
    return T, _class_targets(apply(T, X), radius, cls, surf)[1]


def dorsal_point(V: np.ndarray, near: np.ndarray, reach: float, up: np.ndarray) -> np.ndarray:
    """The most dorsal vertex within `reach` of `near` (all in one frame; `up` = dorsal direction)."""
    c = V[np.linalg.norm(V - near, axis=1) <= reach]
    return c[np.argmax(c @ up)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(strip_overrides())
    cfg = load_config()
    ic = cfg.init
    import torch
    from calibration.mano_index import INDEX_TIP_VERTEX, ManoIndex, closest_on_faces

    rig = load_rig("calibration", cfg)
    rs = rigid_summary(rig)
    print(f"calibration take: {rs['frames']} frames with all three bodies posed", flush=True)
    for k in ("knuckle", "tip", "proximal", "middle"):
        extra = f" / {rs[k]['slip_deg_p50_p95'].round(2)} deg" if "slip_deg_p50_p95" in rs[k] else ""
        print(f"  {k:9s} in fixture frame: slip p50/p95 {rs[k]['slip_mm_p50_p95'].round(2)} mm{extra}", flush=True)

    T_cad_can, rms0 = T_cad_from_canonical(cfg)
    print(f"T_cad_from_canonical Kabsch residual {rms0:.4f} mm", flush=True)

    mano = ManoIndex(str(Path(cfg.mano.model_dir)), device="cpu")
    cp = load_custom_pose(cfg)
    betas = torch.tensor(cp["final_betas"], dtype=torch.float32)[None]
    pose0 = torch.tensor(cp["mano_pose_axis_angle"][3:], dtype=torch.float32)[None]
    P0 = mano.forward(betas, pose0)
    V_can = P0.vertices[0].numpy().astype(float)
    J_can = P0.T_canonical_from_joint[0].numpy().astype(float)
    up_can = np.array([0.0, 1.0, 0.0])                    # dorsum = canonical +y (see mano_index.py)

    # --- start: anatomical points (CAD frame) <-> measured (fixture frame)
    V_cad = apply(T_cad_can, V_can)
    up_cad = T_cad_can[:3, :3] @ up_can
    Jc = apply(T_cad_can, J_can[:, :3, 3])
    tip_cad = V_cad[INDEX_TIP_VERTEX]
    cap_h = float(ic.cap_height_mm) / 1000.0
    flt_h = float(ic.float_height_mm) / 1000.0
    anat = np.stack([dorsal_point(V_cad, Jc[1], 0.012, up_cad) + up_cad * cap_h,
                     dorsal_point(V_cad, 0.5 * (Jc[1] + Jc[2]), 0.012, up_cad) + up_cad * flt_h,
                     dorsal_point(V_cad, 0.5 * (Jc[2] + Jc[3]), 0.012, up_cad) + up_cad * flt_h,
                     dorsal_point(V_cad, 0.5 * (Jc[3] + tip_cad), 0.012, up_cad) + up_cad * cap_h])
    meas = np.stack([rs["knuckle"]["T"][:3, 3], rs["proximal"]["p"], rs["middle"]["p"], rs["tip"]["T"][:3, 3]])
    R0, t0 = procrustes(meas, anat)                                   # cad <- fixture
    res0 = np.linalg.norm(meas @ R0.T + t0 - anat, axis=1) * 1000.0
    print(f"hand-landmark start: residual per point mm {res0.round(1)}", flush=True)

    X_fix, radius, fx_src = fixture_markers_for_cad(cfg, rig)
    print(f"fixture markers for the CAD fit: {fx_src} template", flush=True)
    flat, n_fix, c_fix = coplanar_markers(X_fix, float(ic.flat.tol_mm) / 1000.0, int(ic.flat.min_markers))
    rest = np.setdiff1d(np.arange(len(X_fix)), flat)
    plane_res = np.abs((X_fix[flat] - c_fix) @ n_fix).max() * 1000.0
    below = -((X_fix[rest] - c_fix) @ n_fix) * 1000.0
    z_face, area = cad_top_level(cfg)
    surf = cad_surface_classes(cfg, int(ic.cad_samples), z_face, float(ic.flat.side_nz_max))
    pts, tree = surf["all"]
    cls = np.where(np.isin(np.arange(len(X_fix)), flat), "top", "side")
    w = np.where(cls == "side", float(ic.flat.side_weight), 1.0)
    print(f"top markers (coplanar, max off-plane {plane_res:.2f} mm): {[int(i) + 1 for i in flat]}; side markers "
          f"{[int(i) + 1 for i in rest]} sit {np.round(below, 1).tolist()} mm below the top plane; side weight "
          f"{float(ic.flat.side_weight):g}", flush=True)
    print(f"CAD top face: z = {z_face * 1000:.1f} mm ({area:.0f} mm^2 of +z faces); {len(surf['top'][0])} top / "
          f"{len(surf['side'][0])} side surface samples", flush=True)
    # BASIN SEARCH on any surface: the plane-constrained yaw/xy search (`fit_fixture_flat`). Searching with each
    # marker already restricted to its class missed the right basin (0.73-0.93 mm vs 0.20 mm, 2026-10-22: a top
    # marker restricted to the top face is pulled inward from wherever it starts); the unrestricted search
    # finds it, and in that basin every top marker touches the +z top face and every side marker a wall --
    # checked below, not assumed.
    basins = fit_fixture_flat(X_fix, radius, flat, n_fix, c_fix, z_face, tree, pts, float(ic.flat.yaw_step_deg),
                              int(ic.icp_iters))
    rms_flat, T_flat = basins[0]
    import trimesh
    cad_mesh = trimesh.load(ROOT / cfg.data.fixture_cad_glb, force="mesh")
    _cp, _d, fi = trimesh.proximity.closest_point(cad_mesh, apply(T_flat, X_fix))
    contact_nz = cad_mesh.face_normals[fi, 2]
    consistent = bool(np.all(contact_nz[cls == "top"] >= 0.99) and np.all(np.abs(contact_nz[cls == "side"]) <= float(ic.flat.side_nz_max)))
    print(f"basin contact check: top markers on +z faces and side markers on walls: {consistent} "
          f"(contact n_z top min {contact_nz[cls == 'top'].min():.2f}, side max |n_z| {np.abs(contact_nz[cls == 'side']).max():.2f})",
          flush=True)
    # then the WEIGHTED polish, each marker on its own surface class, side markers up-weighted
    T_cad_fix, gap = polish_topside(X_fix, radius, cls, w, T_flat, surf, int(ic.icp_iters))
    rms = float(np.sqrt((gap ** 2).mean())) * 1000.0
    rms_top, rms_side = (float(np.sqrt((gap[cls == c] ** 2).mean())) * 1000.0 for c in ("top", "side"))
    polish = (float(rot_deg(T_cad_fix[:3, :3], T_flat[:3, :3])), float(np.linalg.norm(T_cad_fix[:3, 3] - T_flat[:3, 3]) * 1000))
    drift = float(np.linalg.norm(apply(T_cad_fix, meas) - anat, axis=1).mean() * 1000.0)
    print("plane-constrained fixture fit, distinct basins (rms |d|-r mm, offset from best deg / mm):", flush=True)
    for r_, T_ in basins[:5]:
        print(f"  {r_:6.2f}  {float(rot_deg(T_[:3, :3], T_flat[:3, :3])):6.1f}  "
              f"{np.linalg.norm(T_[:3, 3] - T_flat[:3, 3]) * 1000:6.1f}", flush=True)
    print(f"6-dof polish moved {polish[0]:.2f} deg / {polish[1]:.2f} mm; gap rms all {rms:.2f}, top {rms_top:.2f}, "
          f"side {rms_side:.2f} mm, max {np.abs(gap).max() * 1000:.2f} mm", flush=True)
    print("  per-marker gap mm: " + " ".join(f"{i + 1}{'s' if c == 'side' else 't'}:{g * 1000:+.2f}"
                                           for i, (c, g) in enumerate(zip(cls, gap))), flush=True)
    T_fix_cad = inv_se3(T_cad_fix)
    T_fix_can = T_fix_cad @ T_cad_can

    # --- initial bindings at the custom pose
    T_kn_can = inv_se3(rs["knuckle"]["T"]) @ T_fix_can
    T_dist_tip = inv_se3(T_fix_can @ J_can[3]) @ rs["tip"]["T"]
    floats = {}
    F = np.asarray(mano.layer.faces)
    dom = mano.weights.argmax(1).numpy()
    for L, bone in (("proximal", 1), ("middle", 2)):
        p_can = apply(inv_se3(T_fix_can), rs[L]["p"])
        cand = np.flatnonzero(np.isin(dom[F], [bone]).any(1))
        fid, bary, h = closest_on_faces(p_can, V_can, F, cand)
        floats[L] = {"face": fid, "bary": bary, "height_mm": h * 1000.0, "bone": bone,
                     "p_canonical_mm": p_can * 1000.0}
        print(f"  floating {L:8s}: face {fid:4d}  height above skin {h * 1000:6.2f} mm", flush=True)
    tip_cap_to_tipvertex = np.linalg.norm(apply(T_fix_can, V_can[INDEX_TIP_VERTEX]) - rs["tip"]["T"][:3, 3]) * 1000
    kn_to_mcp = np.linalg.norm(apply(T_fix_can, J_can[1, :3, 3]) - rs["knuckle"]["T"][:3, 3]) * 1000
    print(f"  tip cap origin -> MANO tip vertex {tip_cap_to_tipvertex:.1f} mm; knuckle cap origin -> MCP joint "
          f"{kn_to_mcp:.1f} mm", flush=True)

    out = {"stage": "C init_rig", "frames": rs["frames"],
           "slip": {k: {kk: v for kk, v in rs[k].items() if kk.startswith("slip")} for k in
                    ("knuckle", "tip", "proximal", "middle")},
           "T_cad_from_canonical": T_json(T_cad_can), "T_cad_from_canonical_rms_mm": rms0,
           "T_fixture_from_cad": T_json(T_fix_cad),
           "fixture_icp": {"template": fx_src, "method": "plane-constrained yaw/xy search (coplanar markers on "
                                                         "the CAD top face), then a weighted 6-dof polish with top "
                                                         "markers on the top face and side markers on walls",
                           "side_markers": [int(i) + 1 for i in rest], "side_weight": float(ic.flat.side_weight),
                           "gap_mm": gap * 1000.0, "rms_top_mm": rms_top, "rms_side_mm": rms_side,
                           "basin_contact_consistent": consistent, "contact_nz": contact_nz,
                           "coplanar_markers": [int(i) + 1 for i in flat], "coplanar_max_off_plane_mm": plane_res,
                           "others_below_plane_mm": below, "cad_top_face_z_mm": z_face * 1000.0,
                           "rms_mm_plane_constrained": rms_flat, "rms_mm": rms, "polish_moved_deg_mm": polish,
                           "basins_rms_mm": [round(b_[0], 3) for b_ in basins[:8]],
                           "basins_offset_deg_mm": [[round(float(rot_deg(b_[1][:3, :3], T_flat[:3, :3])), 2),
                                                     round(float(np.linalg.norm(b_[1][:3, 3] - T_flat[:3, 3]) * 1000), 2)]
                                                    for b_ in basins[:8]],
                           "landmark_drift_mm": drift, "marker_radius_mm": radius * 1000.0},
           "T_fixture_from_knucklecap": T_json(rs["knuckle"]["T"]), "T_fixture_from_tipcap": T_json(rs["tip"]["T"]),
           "float_p_fixture_mm": {L: rs[L]["p"] * 1000.0 for L in ("proximal", "middle")},
           "T_knucklecap_from_canonical": T_json(T_kn_can), "T_distal_from_tipcap": T_json(T_dist_tip),
           "floating": floats, "betas": cp["final_betas"], "hand_pose": cp["mano_pose_axis_angle"][3:],
           "checks_mm": {"tip_cap_origin_to_tip_vertex": tip_cap_to_tipvertex, "knuckle_cap_origin_to_mcp": kn_to_mcp,
                         "landmark_start_residual": res0}}
    write_json(ROOT / cfg.data.out_root / "calibration" / "init_rig.json", out)


if __name__ == "__main__":
    main()
