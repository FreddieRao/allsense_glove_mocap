#!/usr/bin/env python3
"""Shared loaders and SE(3) helpers for the calibration stages. Everything here is METRES.

`load_rig(take)` reads `outputs/<take>/mosca.npz` (Stage B) and converts it at the boundary: the MoSca package
works in mm because its gates are in mm; every stage from here on works in metres (`_mm` only in reports).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BodySeq:
    """One rigid body's re-solved pose sequence (Stage B), metres."""
    name: str
    valid: np.ndarray                       # (T,)
    R_world_from_body_seq: np.ndarray       # (T,3,3)
    t_world_from_body_seq: np.ndarray       # (T,3)
    rmse_mm: np.ndarray                     # (T,)
    template_body: np.ndarray               # (m,3) marker positions in the body frame
    marker_world_seq: np.ndarray            # (T,m,3) the ACCEPTED marker per row, NaN where none (smoothed if asked)
    marker_world_raw_seq: np.ndarray = None # (T,m,3) the same, never smoothed: what errors are measured against

    def T_world_from_body(self, i: int) -> np.ndarray:
        return se3(self.R_world_from_body_seq[i], self.t_world_from_body_seq[i])


@dataclass
class Rig:
    take: str
    rate_hz: float
    n_frames: int
    bodies: dict                            # {role: BodySeq}, role in fixture / knuckle / tip
    float_world_seq: np.ndarray             # (2,T,3) proximal, middle; NaN where unseen (smoothed if asked)
    float_world_raw_seq: np.ndarray = None  # (2,T,3) the same, never smoothed


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def inv_se3(T: np.ndarray) -> np.ndarray:
    R, t = T[..., :3, :3], T[..., :3, 3]
    out = np.zeros_like(T)
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3] = -np.einsum("...ji,...j->...i", R, t)
    out[..., 3, 3] = 1.0
    return out


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """T (4,4) or (...,4,4) applied to points p (...,3)."""
    return np.einsum("...ij,...j->...i", T[..., :3, :3], p) + T[..., :3, 3]


def procrustes(Y: np.ndarray, P: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted orthogonal Procrustes: min sum w||P - (R Y + t)||^2. Returns (R (d,d), t (d,)), d = 2 or 3.
    (TableMocap calibration/solve_plate_binding.py:procrustes.)"""
    w = np.ones(len(Y)) if w is None else w
    sw = w.sum()
    yc = (Y * w[:, None]).sum(0) / sw
    pc = (P * w[:, None]).sum(0) / sw
    H = ((P - pc) * w[:, None]).T @ (Y - yc)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(Y.shape[1])                          # 2D or 3D
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))      # no reflections
    R = U @ D @ Vt
    return R, pc - R @ yc


def rot_mean(R_seq: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R_seq).mean().as_matrix()


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Angle(s) between rotations, degrees; broadcasts over leading axes."""
    c = (np.einsum("...ji,...ji->...", Ra, Rb) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def load_rig(take: str, cfg) -> Rig:
    z = np.load(ROOT / cfg.data.out_root / take / "mosca.npz")
    names = list(z["body_names"])
    T = len(z[f"{names[0]}__valid"])
    frame, p, body, row = z["frame"], z["p_world_mm"].astype(float) / 1000.0, z["body"], z["row"]
    roles = {"fixture": cfg.data.bodies.fixture, "knuckle": cfg.data.bodies.knuckle, "tip": cfg.data.bodies.tip}
    bodies = {}
    for role, n in roles.items():
        k = names.index(n)
        tk = z[f"{n}__template_mm"] / 1000.0
        mw = np.full((T, len(tk), 3), np.nan)
        on = body == k
        mw[frame[on], row[on]] = p[on]
        R = z[f"{n}__R_world_from_body_seq"].astype(float)
        t_mm = z[f"{n}__t_world_from_body_mm_seq"].astype(float)
        valid = z[f"{n}__valid"]
        mw_raw = mw.copy()
        # optional temporal smoothing (mosca/smooth.py): of the raw marker tracks, then the pose re-solved from
        # them (`smooth.target: markers`), or of the re-solved pose itself (`target: pose`)
        method, target = str(cfg.smooth.method), str(cfg.smooth.target)
        if method != "none" and target == "markers":
            from mosca.smooth import resolve_from_markers, smooth_marker_tracks
            mw = smooth_marker_tracks(mw * 1000.0, float(z["rate_hz"]), method, cfg.smooth) / 1000.0
            R, t_mm, valid = resolve_from_markers(z[f"{n}__template_mm"], mw * 1000.0, valid)
        elif method != "none":
            from mosca.smooth import smooth_body
            R, t_mm = smooth_body(method, R, t_mm, valid, z[f"{n}__rmse_mm"].astype(float),
                                  z[f"{n}__n"].astype(int), z[f"{n}__template_mm"], float(z["rate_hz"]), cfg.smooth)
        bodies[role] = BodySeq(n, valid, R, t_mm / 1000.0, z[f"{n}__rmse_mm"].astype(float), tk, mw, mw_raw)
    fl = z["float_p_world_mm_seq"].astype(float) / 1000.0
    return Rig(take, float(z["rate_hz"]), T, bodies, smooth_floats(fl, float(z["rate_hz"]), cfg), fl)


def smooth_floats(fl_m: np.ndarray, rate_hz: float, cfg) -> np.ndarray:
    """(2,T,3) m floating tracks, smoothed when `smooth.target: markers` (else returned as is)."""
    if str(cfg.smooth.method) == "none" or str(cfg.smooth.target) != "markers":
        return fl_m
    from mosca.smooth import smooth_marker_tracks
    return np.swapaxes(smooth_marker_tracks(np.swapaxes(fl_m, 0, 1) * 1000.0, rate_hz, str(cfg.smooth.method),
                                            cfg.smooth), 0, 1) / 1000.0


def fixture_markers_for_cad(cfg, rig) -> tuple[np.ndarray, np.ndarray, str]:
    """(marker centres (18,3) m in the fixture body frame, radii (18,) m, source) for the fixture->CAD fit.

    `init.fixture_template: asset` (default, user decision 2026-10-22) uses the `.motive` asset's DEFINED
    offsets verbatim; `refined` uses Stage B's per-take template from the calibration take. The radius is
    always the asset's measured diameter / 2 + `init.marker_base_mm`."""
    from mosca.take import parse_bodies
    ab = parse_bodies(ROOT / cfg.data.assets[0])[cfg.data.bodies.fixture]
    src = str(cfg.init.fixture_template)
    X = ab.marker_pos_body if src == "asset" else rig.bodies["fixture"].template_body
    d = ab.marker_diameter_m
    r = 0.5 * np.nan_to_num(d, nan=np.nanmedian(d)) + float(cfg.init.marker_base_mm) / 1000.0
    return np.asarray(X, float), r, src


def sfx(cfg) -> str:
    """'_<run.tag>' or '': the suffix every Stage D-F artefact and viewer carries, so variants sit side by side
    (`python -m calibration.fit_hand run.tag=prior2`, then the same override for track / eval / viewer)."""
    tag = str(cfg.run.tag or "")
    return f"_{tag}" if tag else ""


def calib_sfx(cfg) -> str:
    """Suffix of the hand_calib that Stage E-F and the viewers read: `run.calib_tag` if set (`default` = the default
    calibration), else `run.tag`. Lets a TRACKING variant (e.g. the smoothed knuckle root) reuse a calibration."""
    ct = cfg.run.calib_tag
    if ct is None:
        return sfx(cfg)
    return "" if str(ct) in ("", "default") else f"_{ct}"


def load_obj_vertices(path: Path) -> np.ndarray:
    import trimesh
    return np.asarray(trimesh.load(path, process=False).vertices, float)


def T_cad_from_canonical(cfg) -> tuple[np.ndarray, float]:
    """(T (4,4) m, rms mm): `hand_aligned_mm.obj` IS `custom_pose_metric.obj` moved into the CAD frame
    (same 1538 faces, unwelded), so one Kabsch on corresponding vertices gives the transform exactly."""
    import trimesh
    A = trimesh.load(ROOT / "data/assets/custom_pose_metric.obj", process=False)
    B = trimesh.load(ROOT / cfg.data.hand_cad_obj, process=False)
    X = np.asarray(A.vertices)[np.asarray(A.faces).reshape(-1)]
    Y = np.asarray(B.vertices)[np.asarray(B.faces).reshape(-1)] / 1000.0
    R, t = procrustes(X, Y)
    rms = float(np.sqrt(((X @ R.T + t - Y) ** 2).sum(1).mean())) * 1000.0
    return se3(R, t), rms


def load_custom_pose(cfg) -> dict:
    return json.loads((ROOT / cfg.data.mano_pose_json).read_text())


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"wrote {path}", flush=True)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 6).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def T_json(T: np.ndarray) -> dict:
    """A transform for a JSON artefact: R, t (m and mm) and the scipy quaternion."""
    from scipy.spatial.transform import Rotation
    return {"R": np.round(T[:3, :3], 7).tolist(), "t_m": np.round(T[:3, 3], 7).tolist(),
            "t_mm": np.round(T[:3, 3] * 1000.0, 3).tolist(),
            "q_xyzw": np.round(Rotation.from_matrix(T[:3, :3]).as_quat(), 7).tolist()}


def T_from_json(d: dict) -> np.ndarray:
    return se3(np.asarray(d["R"]), np.asarray(d["t_m"]))
