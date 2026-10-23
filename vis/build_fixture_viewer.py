#!/usr/bin/env python3
"""A static 3D HTML page to inspect the fixture registration: Motive's fixture asset (marker constellation) on the
fixture CAD, under every candidate T_fixture_from_cad, together with the designed hand and the mocap-fitted hand.

Everything is drawn in the CAD frame, mm:

    fixture CAD          data/assets/hand_fixtured.glb, decimated for display
    designed hand        data/assets/hand_aligned_mm.obj (custom_pose MANO placed in the CAD by design)
    index tip scan       data/assets/index_aligned_mm.obj (glove-wrapped tip, CAD frame)
    fixture markers      the `.motive` asset's defined offsets (fixture body frame) mapped by inv(T_fixture_from_cad),
                         drawn at their true radius, coloured by |d| - r: the signed gap between the marker
                         centre's distance to the CAD surface and its radius (0 = the ball touches the surface)
    fitted hand          the mocap calibration's MANO (fitted betas, calibration-take posture) placed through
                         inv(T_fixture_from_cad) @ T_fixture_from_knucklecap @ T_knucklecap_from_canonical
    caps / floats        the knuckle and tip cap markers and both floating markers, median over the calibration
                         take in the fixture frame, mapped the same way

CANDIDATES. Stage C's registration (`init_rig.json`: the coplanar markers 8-17 on the CAD top face, yaw and x/y
searched, then a free 6-dof polish -- 0.20 mm rms), the next plane-constrained basins, the distinct basins of
the earlier UNCONSTRAINED offset-surface ICP (`--starts` random rotations; 1.23-1.8 mm, markers on walls and
the underside), and the Stage D value of the last `hand_calib.json`.

    ~/miniforge3/envs/far_grasp/bin/python -m vis.build_fixture_viewer
Writes notes/figures/fixture_registration.html.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import (ROOT, T_cad_from_canonical, T_from_json, apply, calib_sfx, inv_se3,  # noqa: E402
                                load_rig, rot_deg, se3, sfx)
from calibration.init_rig import cad_surface, icp_offset                                           # noqa: E402
from calibration.mano_index import ManoIndex                                                       # noqa: E402
from config import load_config, strip_overrides                                                   # noqa: E402
from mosca.take import parse_bodies                                                               # noqa: E402

HERE = Path(__file__).resolve().parent


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def mesh_blob(V: np.ndarray, F: np.ndarray, target_faces: int | None = None) -> dict:
    if target_faces and len(F) > target_faces:
        import fast_simplification
        V, F = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int64), 1.0 - target_faces / len(F))
    return {"v": b64(np.asarray(V, np.float32)), "f": b64(np.asarray(F, np.uint32)), "nf": int(len(F))}


def icp_basins(X, radius, tree, pts, n_starts: int, iters: int) -> list[tuple[float, np.ndarray]]:
    """Distinct ICP solutions (rms mm, T_cad_from_fixture) from random starts, best first."""
    from scipy.spatial.transform import Rotation
    sols = []
    for k in range(n_starts):
        R = Rotation.random(random_state=k).as_matrix()
        T, rms = icp_offset(X, radius, se3(R, pts.mean(0) - R @ X.mean(0)), tree, pts, iters)
        sols.append((rms, T))
    sols.sort(key=lambda s: s[0])
    out = []
    for rms, T in sols:
        if all(rot_deg(T[:3, :3], U[:3, :3]) > 2.0 or np.linalg.norm(T[:3, 3] - U[:3, 3]) > 0.002 for _, U in out):
            out.append((rms, T))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--starts", type=int, default=300)
    ap.add_argument("--basins", type=int, default=8)
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    import trimesh

    init = json.loads((ROOT / cfg.data.out_root / "calibration" / "init_rig.json").read_text())
    calib = json.loads((ROOT / cfg.data.out_root / "calibration" / f"hand_calib{calib_sfx(cfg)}.json").read_text())
    rig = load_rig("calibration", cfg)
    from calibration.common import fixture_markers_for_cad
    X_fix, radius, fx_src = fixture_markers_for_cad(cfg, rig)                     # (18,3) m, the asset's offsets
    pts, tree = cad_surface(cfg, int(cfg.init.cad_samples))

    # ---- candidate registrations (all as T_cad_from_fixture)
    from calibration.init_rig import cad_top_level, coplanar_markers, fit_fixture_flat
    flat, n_fix, c_fix = coplanar_markers(X_fix, float(cfg.init.flat.tol_mm) / 1000.0, int(cfg.init.flat.min_markers))
    z_face, _ = cad_top_level(cfg)
    flat_basins = fit_fixture_flat(X_fix, radius, flat, n_fix, c_fix, z_face, tree, pts,
                                   float(cfg.init.flat.yaw_step_deg), int(cfg.init.icp_iters))
    cands = [("Stage C: plane + polish (used)", inv_se3(T_from_json(init["T_fixture_from_cad"])))]
    for j, (rms, T) in enumerate(flat_basins[1:4]):
        cands.append((f"plane-constrained basin {j + 2}", T))
    for j, (rms, T) in enumerate(icp_basins(X_fix, radius, tree, pts, args.starts, int(cfg.init.icp_iters))[:args.basins]):
        cands.append((f"unconstrained ICP basin {j + 1}", T))
    if calib.get("fixture"):
        T_d = inv_se3(T_from_json(calib["fixture"]["T_fixture_from_cad"]))
        if not np.allclose(T_d, cands[0][1], atol=1e-6):          # Stage D holds it fixed: same as row 1 then
            cands.append(("Stage D (hand_calib.json)", T_d))
    print(f"{len(cands)} candidates", flush=True)

    # ---- the fitted hand in the fixture frame (calibration-take posture), and the measured caps / floats
    mano = ManoIndex(str(cfg.mano.model_dir), device="cpu")
    hp = np.asarray(calib["base_hand_pose"], np.float32).copy()
    hp[:9] = np.radians(np.asarray(calib["theta_cal_index_deg_mean"], float).reshape(-1))
    joff = None
    if calib.get("joint_offset_radius_mm", 0.0) > 0:            # the fitted index pivot corrections
        joff = torch.zeros(16, 3)
        joff[1:4] = torch.as_tensor(np.asarray(calib["index_joint_offset_m"], np.float32))
    with torch.no_grad():
        P = mano.forward(torch.tensor(calib["betas"], dtype=torch.float32)[None], torch.as_tensor(hp)[None], joff)
    V_can = P.vertices[0].numpy().astype(float)
    T_fix_kn = T_from_json(init["T_fixture_from_knucklecap"])
    T_fix_tip = T_from_json(init["T_fixture_from_tipcap"])
    T_fix_can = T_fix_kn @ T_from_json(calib["T_knucklecap_from_canonical"])
    hand_fix = apply(T_fix_can, V_can)
    caps_fix = {"knuckle": apply(T_fix_kn, rig.bodies["knuckle"].template_body),
                "tip": apply(T_fix_tip, rig.bodies["tip"].template_body)}
    floats_fix = np.array([init["float_p_fixture_mm"][L] for L in ("proximal", "middle")]) / 1000.0

    T_cad_can, _ = T_cad_from_canonical(cfg)
    cp = json.loads((ROOT / cfg.data.mano_pose_json).read_text())
    with torch.no_grad():
        V_design = mano.forward(torch.tensor(cp["final_betas"], dtype=torch.float32)[None],
                                torch.tensor(cp["mano_pose_axis_angle"][3:], dtype=torch.float32)[None]).vertices[0]
    designed = apply(T_cad_can, V_design.numpy().astype(float))       # hand_aligned_mm.obj, per MANO vertex
    dom = mano.weights.argmax(1).numpy()
    index_v, palm_v = np.isin(dom, [1, 2, 3]), dom == 0
    T_ref = cands[0][1]
    sol = []
    for name, T in cands:
        mk = apply(T, X_fix)
        d, j = tree.query(mk)
        gap = (d - radius) * 1000.0
        hand_cad = apply(T, hand_fix)
        sol.append({
            "name": name, "rms_gap_mm": round(float(np.sqrt((gap ** 2).mean())), 3),
            "max_abs_gap_mm": round(float(np.abs(gap).max()), 2),
            "rot_from_first_deg": round(float(rot_deg(T[:3, :3], T_ref[:3, :3])), 2),
            "trans_from_first_mm": round(float(np.linalg.norm(T[:3, 3] - T_ref[:3, 3]) * 1000.0), 2),
            "hand_vs_designed_p50_mm": round(float(np.median(np.linalg.norm(hand_cad - designed, axis=1)) * 1000.0), 2),
            "index_vs_designed_p50_mm": round(float(np.median(np.linalg.norm(hand_cad - designed, axis=1)[index_v]) * 1000.0), 2),
            "palm_vs_designed_p50_mm": round(float(np.median(np.linalg.norm(hand_cad - designed, axis=1)[palm_v]) * 1000.0), 2),
            "markers": b64((mk * 1000.0).astype(np.float32)), "surface": b64((pts[j] * 1000.0).astype(np.float32)),
            "gap": b64(gap.astype(np.float32)),
            "hand": b64((hand_cad * 1000.0).astype(np.float32)),
            "caps": b64((np.concatenate([apply(T, caps_fix["knuckle"]), apply(T, caps_fix["tip"]),
                                         apply(T, floats_fix)]) * 1000.0).astype(np.float32))})
        print(f"  {name:22s} rms gap {sol[-1]['rms_gap_mm']:5.2f} mm  max {sol[-1]['max_abs_gap_mm']:5.2f}  "
              f"vs row 1 {sol[-1]['rot_from_first_deg']:6.2f} deg {sol[-1]['trans_from_first_mm']:6.2f} mm  "
              f"hand vs designed p50 {sol[-1]['hand_vs_designed_p50_mm']:6.2f} mm (palm {sol[-1]['palm_vs_designed_p50_mm']:.1f}, "
              f"index {sol[-1]['index_vs_designed_p50_mm']:.1f})", flush=True)

    cad = trimesh.load(ROOT / cfg.data.fixture_cad_glb, force="mesh")
    hand_obj = trimesh.load(ROOT / cfg.data.hand_cad_obj, process=True)
    tip_obj = trimesh.load(ROOT / cfg.data.index_tip_scan_obj, process=True)
    blob = {
        "cad": mesh_blob(np.asarray(cad.vertices) * 1000.0, np.asarray(cad.faces), 80000),
        "designed_hand": mesh_blob(np.asarray(hand_obj.vertices), np.asarray(hand_obj.faces)),
        "tip_scan": mesh_blob(np.asarray(tip_obj.vertices), np.asarray(tip_obj.faces), 20000),
        "mano_faces": b64(mano.faces.numpy().astype(np.uint32)),
        "marker_radius": (radius * 1000.0).round(3).tolist(), "float_radius": float(cfg.mosca.floating.radius_mm),
        "n_caps": [len(caps_fix["knuckle"]), len(caps_fix["tip"]), 2],
        "solutions": sol, "template": fx_src, "coplanar": [int(i) for i in flat],
    }
    js = (HERE / "fixture_viewer.js").read_text()
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>fixture registration</title><style>"
            + (HERE / "viewer.css").read_text() + FIXTURE_CSS + "</style></head><body>" + BODY
            + "<script>" + (HERE / "vendor" / "three.min.js").read_text() + "</script>"
            + "<script>window.FIX_VIEW=" + json.dumps(blob) + ";</script><script>" + js + "</script></body></html>")
    out = ROOT / "notes" / "figures" / f"fixture_registration{sfx(cfg)}.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


FIXTURE_CSS = """
main { grid-template-columns: 1fr 380px; }
body { grid-template-rows: auto 1fr; }
#sols table { border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 11.5px; }
#sols td, #sols th { padding: 2px 5px; text-align: right; border-bottom: 1px solid #2a2e36; }
#sols td:first-child, #sols th:first-child { text-align: left; }
#sols tr { cursor: pointer; } #sols tr:hover { background: #232730; } #sols tr.sel { background: #33405a; }
#gapbar { height: 10px; background: linear-gradient(90deg, #3e8ef7, #e8e8e8, #e5484d); margin: 4px 0 2px; }
#markers td { padding: 1px 6px; font-family: ui-monospace, monospace; font-size: 11.5px; text-align: right; }
"""

BODY = """
<header><div id="title">fixture registration: Motive fixture asset on the CAD, CAD frame (mm)</div>
<div id="stats">drag rotate / right-drag pan / wheel zoom</div></header>
<main>
  <section class="panel"><canvas id="cv"></canvas><div class="count" id="count"></div></section>
  <aside>
    <h3>Candidate T_fixture_from_cad <small>click a row</small></h3><div id="sols"></div>
    <h3>Show</h3>
    <label><input type="checkbox" id="opt-cad" checked> fixture CAD</label>
    <label><input type="checkbox" id="opt-markers" checked> fixture markers (true radius)</label>
    <label><input type="checkbox" id="opt-gaps" checked> marker &rarr; nearest CAD point</label>
    <label><input type="checkbox" id="opt-designed" checked> designed hand (hand_aligned_mm.obj)</label>
    <label><input type="checkbox" id="opt-fitted" checked> mocap-fitted MANO hand</label>
    <label><input type="checkbox" id="opt-caps" checked> cap + floating markers (measured)</label>
    <label><input type="checkbox" id="opt-tip"> index tip scan</label>
    <h3>Marker gap |d| - r (mm)</h3>
    <div id="gapbar"></div><div style="display:flex;justify-content:space-between"><span>-3 inside</span><span>0</span><span>+3 off</span></div>
    <table id="markers"></table>
  </aside>
</main>
"""


if __name__ == "__main__":
    main()
