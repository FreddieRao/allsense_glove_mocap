#!/usr/bin/env python3
"""A self-contained 3D HTML viewer of one take: markers by MoSca label, both caps' poses (MoSca vs Motive), the
tracked MANO hand, the predicted cap / floating markers and their residuals.

The page pattern is TableMocap's (`scripts/mosca_steps/common.py:page`, `scripts/marker_assignment_mosca/viewer.js`):
three.js inlined, the data inlined as one JSON blob of base64 typed arrays, because a `file://` page cannot fetch.
THE VIEWER DOES NO GEOMETRY (TableMocap vis/README.md) with one exception that is pure bookkeeping: MANO is skinned
in the page, from the betas-shaped rest mesh + top-4 skin weights sent once and 16 world bone transforms per frame
(TableMocap vis/export_motive_bundle.py:load_hand_mesh), so a frame costs 768 B instead of 4.7 kB of vertices.
Pose corrective blend shapes are not applied in the page (they move the skin < 2 mm on this take); every NUMBER
on the page comes from Python.

World is Motive's (Y up), mm, re-centred on the take's median marker. Downsampled to `--hz` (60 by default).

    ~/miniforge3/envs/far_grasp/bin/python -m vis.build_viewer --take calibration free_motion
Writes notes/figures/hand_calib_<take>[_<run.tag>].html.
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

from calibration.common import ROOT, calib_sfx, load_rig, sfx                                        # noqa: E402
from calibration.fit_hand import holdout_mask                                       # noqa: E402
from calibration.hand_model import Calib, make_frames                               # noqa: E402
from calibration.mano_index import ManoIndex                                        # noqa: E402
from calibration.track import load_calib                                            # noqa: E402
from config import load_config, strip_overrides                                     # noqa: E402

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ["fixture", "knuckle cap", "tip cap", "floating proximal", "floating middle"]
CLASS_COLOURS = ["#8a7f6a", "#3e8ef7", "#e5484d", "#46c46a", "#f5a524"]
FREE = 255


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def mm16(p_mm: np.ndarray, centre: np.ndarray) -> str:
    """World mm re-centred, as int16 tenths of a mm (the viewer divides by 10), base64. NaN -> -32768."""
    q = np.round((np.asarray(p_mm, float) - centre) * 10)
    q = np.where(np.isfinite(q), q.clip(-32767, 32767), -32768)
    return b64(q.astype(np.int16))


def _script(text: str) -> str:
    return "<script>" + text.replace("</script", "<\\/script") + "</script>"


def page(title: str, blob: dict) -> str:
    return ("<!doctype html><html><head><meta charset='utf-8'><title>" + title + "</title><style>"
            + (HERE / "viewer.css").read_text() + "</style></head><body>" + (HERE / "viewer_body.html").read_text()
            + _script((HERE / "vendor" / "three.min.js").read_text())
            + _script("window.HAND_VIEW=" + json.dumps(blob) + ";") + _script((HERE / "viewer.js").read_text())
            + "</body></html>")


def pose7(R: np.ndarray, t_mm: np.ndarray, valid: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """(T,7) q_xyzw + t (mm, re-centred), NaN where invalid."""
    from scipy.spatial.transform import Rotation
    out = np.full((len(R), 7), np.nan, np.float32)
    ok = valid & np.isfinite(R[:, 0, 0])
    out[ok, :4] = Rotation.from_matrix(R[ok]).as_quat()
    out[ok, 4:] = t_mm[ok] - centre
    return out


def bone_transforms(tr, frames: np.ndarray, calib: Calib, mano: ManoIndex, centre: np.ndarray):
    """(A (F,16,12) float32: per-frame world bone transforms rest -> world, mm, re-centred; NaN untracked,
    chain (F,5,3) mm: wrist, index MCP/PIP/DIP joints, tip vertex) from a Stage E track's own root pose."""
    A = np.full((len(frames), 16, 12), np.nan, np.float32)
    sel = np.flatnonzero(tr["valid"][frames])
    for a in range(0, len(sel), 1024):
        s = sel[a:a + 1024]
        idx = frames[s]
        with torch.no_grad():
            theta = torch.as_tensor(tr["index_pose"][idx], device=mano.device)
            posed = mano.forward(calib.betas, calib.full_pose(theta), calib.joint_offset_full())
            T_wc = torch.as_tensor(tr["T_world_from_canonical"][idx], dtype=torch.float32, device=mano.device)
            Tj = posed.T_canonical_from_joint.clone()
            # LBS's rest->posed transform: [R_j | J_posed - R_j J_rest]
            Tj[:, :, :3, 3] = Tj[:, :, :3, 3] - torch.einsum("bjik,bjk->bji", Tj[:, :, :3, :3], posed.J_rest)
            Aw = (T_wc[:, None] @ Tj).cpu().numpy()
        Aw[:, :, :3, 3] = Aw[:, :, :3, 3] * 1000.0 - centre
        A[s] = Aw[:, :, :3, :].reshape(len(s), 16, 12)
    joints = tr["joints_world"][frames] * 1000.0
    chain = np.concatenate([joints[:, [0, 1, 2, 3]], tr["tip_vertex_world"][frames][:, None] * 1000.0], 1)
    return A, chain


def knuckle_pred_m(tr, frames: np.ndarray, T_kc: np.ndarray, template_m: np.ndarray) -> np.ndarray:
    """(F,m,3) m: the knuckle-cap markers where the TRACKED ROOT puts the cap,
    T_world_from_canonical @ inv(T_knucklecap_from_canonical) @ x_row (NaN where untracked)."""
    T_wk = tr["T_world_from_canonical"][frames].astype(float) @ np.linalg.inv(T_kc)[None]
    return np.einsum("fij,mj->fmi", T_wk[:, :3, :3], template_m) + T_wk[:, None, :3, 3]


def residual_stats(tr, frames: np.ndarray, kn_pred_m: np.ndarray) -> np.ndarray:
    """(F,4) mm: mean residual of the tip-cap markers, floating proximal, floating middle, mean of the knuckle-cap
    markers -- against the RAW markers."""
    raw = lambda k: tr[f"{k}_raw"] if f"{k}_raw" in tr.files else tr[k]                  # noqa: E731
    with np.errstate(all="ignore"):
        e_tip = np.nanmean(np.linalg.norm(tr["tip_pred"][frames] - raw("tip_obs")[frames], axis=2), 1) * 1000.0
        e_kn = np.nanmean(np.linalg.norm(kn_pred_m - raw("knuckle_obs")[frames], axis=2), 1) * 1000.0
    e_fl = np.linalg.norm(tr["float_pred"][frames] - raw("float_obs")[frames], axis=2) * 1000.0
    return np.stack([e_tip, e_fl[:, 0], e_fl[:, 1], e_kn], 1)


def build(take: str, cfg, hz: float) -> Path:
    z = np.load(ROOT / cfg.data.out_root / take / "mosca.npz")
    tr = np.load(ROOT / cfg.data.out_root / take / f"mano_track{sfx(cfg)}.npz")
    rig = load_rig(take, cfg)
    rate = float(z["rate_hz"])
    step = max(1, int(round(rate / hz)))
    frames = np.arange(0, rig.n_frames, step)
    names = list(z["body_names"])
    roles = [cfg.data.bodies.fixture, cfg.data.bodies.knuckle, cfg.data.bodies.tip]
    cls_of_body = np.array([roles.index(n) for n in names])

    # ---- markers of the shown frames, classed
    fr_obs, p_obs = z["frame"], z["p_world_mm"].astype(float)
    cls = np.full(len(fr_obs), FREE, np.uint8)
    cls[z["body"] >= 0] = cls_of_body[z["body"][z["body"] >= 0]]
    fl = tr["float_label"]
    cls[fl == 0], cls[fl == 1] = 3, 4
    keep = np.isin(fr_obs, frames)
    centre = np.median(p_obs[keep], 0)
    order = np.argsort(np.searchsorted(frames, fr_obs[keep]), kind="stable")
    f_new = np.searchsorted(frames, fr_obs[keep])[order]
    ptr = np.searchsorted(f_new, np.arange(len(frames) + 1)).astype(np.int32)

    # ---- poses: MoSca re-solve vs Motive, knuckle and tip caps (+ fixture)
    poses = {}
    for role, n in zip(("fixture", "knuckle", "tip"), roles):
        poses[role] = {
            "mosca": b64(pose7(z[f"{n}__R_world_from_body_seq"][frames].astype(float),
                               z[f"{n}__t_world_from_body_mm_seq"][frames].astype(float), z[f"{n}__valid"][frames], centre)),
            "motive": b64(pose7(z[f"{n}__motive_R_world_from_body_seq"][frames].astype(float),
                                z[f"{n}__motive_t_world_from_body_mm_seq"][frames].astype(float),
                                z[f"{n}__motive_valid"][frames], centre))}

    # ---- MANO: shaped rest mesh, top-4 weights, per-frame world bone transforms (rest -> world)
    mano = ManoIndex(str(cfg.mano.model_dir))
    calib: Calib = load_calib(cfg, mano, load_rig("calibration", cfg) if take != "calibration" else rig)
    L = mano.layer
    with torch.no_grad():
        from smplx.lbs import blend_shapes
        v_rest = (L.v_template[None] + blend_shapes(calib.betas, L.shapedirs))[0].cpu().numpy() * 1000.0
    W = mano.weights.cpu().numpy()
    top = np.argsort(-W, 1)[:, :4]
    tw = np.take_along_axis(W, top, 1)
    tw /= tw.sum(1, keepdims=True)
    A, chain = bone_transforms(tr, frames, calib, mano, centre)
    tracked = tr["valid"][frames]

    # ---- predictions and residuals (from Stage E / evaluation numbers, mm)
    tip_pred, fl_pred = tr["tip_pred"][frames] * 1000.0, tr["float_pred"][frames] * 1000.0
    T_kc = calib.T_knucklecap_from_canonical.numpy()
    kn_tmpl = rig.bodies["knuckle"].template_body
    kn_pred = knuckle_pred_m(tr, frames, T_kc, kn_tmpl)
    stats = residual_stats(tr, frames, kn_pred)
    held = holdout_mask(rig.n_frames, cfg.fit)[frames] if take == "free_motion" else np.zeros(len(frames), bool)
    calj = json.loads((ROOT / cfg.data.out_root / "calibration" / f"hand_calib{calib_sfx(cfg)}.json").read_text())
    evalj = json.loads((ROOT / "notes" / f"2026-10-22_calibration_eval{sfx(cfg)}.json").read_text())["takes"][take]["all"]
    blob = {
        "take": take + sfx(cfg), "rate_hz": rate / step, "n_frames": len(frames), "frame_idx": b64(frames.astype(np.int32)),
        "centre": centre.round(2).tolist(),
        "ptr": b64(ptr), "p": mm16(p_obs[keep][order], centre), "cls": b64(cls[keep][order]),
        "class_names": CLASS_NAMES, "class_colours": CLASS_COLOURS,
        "poses": poses,
        "templates": {r: (rig.bodies[r].template_body * 1000.0).round(3).tolist() for r in ("fixture", "knuckle", "tip")},
        "mano": {"v_rest": b64(v_rest.astype(np.float32)), "faces": b64(mano.faces.cpu().numpy().astype(np.uint16)),
                 "w_idx": b64(top.astype(np.uint8)), "w": b64(tw.astype(np.float32)), "A": b64(A),
                 "chain": mm16(chain, centre)},
        "pred": {"tip": mm16(tip_pred, centre), "float": mm16(fl_pred, centre), "knuckle": mm16(kn_pred * 1000.0, centre)},
        "stats": b64(np.nan_to_num(stats, nan=-1).astype(np.float32)), "held": b64(held.astype(np.uint8)),
        "summary": {"betas": [round(b, 3) for b in calj["betas"]],
                    "rms_mm": {k: evalj[k]["rms_mm"] for k in ("tip_cap", "knuckle_cap", "float_proximal", "float_middle")},
                    "heights_mm": {k: round(v["height_mm"], 2) for k, v in calj["floating"].items()}},
    }
    # ---- optional overlay: the same calibration tracked with the SMOOTHED WRIST (smooth.wrist, track.py)
    ov_tag = str(cfg.vis.wrist_overlay_tag or "")
    ov_path = ROOT / cfg.data.out_root / take / f"mano_track_{ov_tag}.npz"
    if ov_tag and ov_path.exists() and not sfx(cfg):
        to = np.load(ov_path)
        A_o, chain_o = bone_transforms(to, frames, calib, mano, centre)
        T_wk = to["T_world_from_canonical"][frames].astype(float) @ np.linalg.inv(
            calib.T_knucklecap_from_canonical.numpy())[None]           # world <- knuckle cap, from the smoothed wrist
        blob["overlay"] = {"label": f"smoothed wrist ({cfg.smooth.wrist.method}, "
                                    f"{float(cfg.smooth.wrist.bandwidth_hz):g} Hz)",
                           "A": b64(A_o), "chain": mm16(chain_o, centre),
                           "knuckle": b64(pose7(T_wk[:, :3, :3], T_wk[:, :3, 3] * 1000.0, to["valid"][frames], centre)),
                           "pred_knuckle": mm16(knuckle_pred_m(to, frames, T_kc, kn_tmpl) * 1000.0, centre),
                           "stats": b64(np.nan_to_num(residual_stats(to, frames, knuckle_pred_m(to, frames, T_kc, kn_tmpl)),
                                                      nan=-1).astype(np.float32))}
        print(f"  overlay: {ov_path.name}", flush=True)
    out = ROOT / "notes" / "figures" / f"hand_calib_{take}{sfx(cfg)}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(f"hand calibration - {take}{sfx(cfg)}", blob))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames at {rate / step:.0f} Hz)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--take", nargs="+", default=["calibration", "free_motion"])
    ap.add_argument("--hz", type=float, default=60.0)
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    for t in args.take:
        build(t, cfg, args.hz)


if __name__ == "__main__":
    main()
