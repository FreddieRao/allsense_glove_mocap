#!/usr/bin/env python3
"""The calibrated index-finger observation model, shared by the fit (Stage D) and the tracker (Stage E).

One frame i of either take predicts every marker from the knuckle cap's pose, the index pose and the
calibration:

    T_world_from_canonical[i] = T_world_from_knucklecap[i] @ T_knucklecap_from_canonical
    tip cap row r             = T_world_from_canonical[i] @ T_canonical_from_joint[3](beta, theta_i)
                                                          @ T_distal_from_tipcap @ x_tipcap[r]
    floating marker m         = T_world_from_canonical[i] @ skin(face_m, bary_m, h_m; beta, theta_i)

WHY THE KNUCKLE CAP IS THE ROOT IN BOTH TAKES. The calibration take could instead be rooted on the fixture
(`T_world_from_fixture @ T_fixture_from_cad @ T_cad_from_canonical`), but Stage C measured the fixture->CAD
registration to be ambiguous at ~10 mm / 5-17 deg (a family of ICP basins at 1.23-1.8 mm rms: the fixture is
a near-planar plate whose markers do not touch its CAD surface to better than +-2.5 mm). Rooting on the
knuckle cap makes the two takes share ONE binding, and the fixture/CAD survives only as a soft prior on where
that binding puts the hand in the fixture (`fit_hand.fixture_prior`).

THE INDEX POSE is 9 numbers per frame (axis-angle of MANO joints 1-3 = hand_pose[0:9]); every other joint is
held at the calibration take's pose. In MANO canonical, flexion is +z (a +1 rad PIP turn about z moves the
tip 40 mm toward -y, the palm; the dorsum is +y), abduction is about y, twist about x (the finger points -x).
`joint_penalty` keeps PIP/DIP hinge-like and every angle inside a human range.

Units: metres inside, mm in every loss scale and report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from calibration.mano_index import INDEX_TIP_VERTEX, ManoIndex, rodrigues, skin_point

FLOAT_LABELS = ("proximal", "middle")
FLOAT_BONE = {"proximal": 1, "middle": 2}


# ------------------------------------------------------------------------------------------- SE(3), torch
def se3_t(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3], T[..., :3, 3], T[..., 3, 3] = R, t, 1.0
    return T


def apply_t(T: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """T (...,4,4) on points p (...,M,3) -> (...,M,3)."""
    return torch.einsum("...ij,...mj->...mi", T[..., :3, :3], p) + T[..., None, :3, 3]


class Delta(torch.nn.Module):
    """A rigid transform as a fixed base times a learnable left increment: T = exp(w) [I|v] @ T0."""

    def __init__(self, T0: np.ndarray, device):
        super().__init__()
        self.register_buffer("T0", torch.as_tensor(T0, dtype=torch.float32, device=device))
        self.w = torch.nn.Parameter(torch.zeros(3, device=device))
        self.v = torch.nn.Parameter(torch.zeros(3, device=device))

    def forward(self) -> torch.Tensor:
        R = rodrigues(self.w[None])[0]
        return se3_t(R @ self.T0[:3, :3], R @ self.T0[:3, 3] + self.v)

    def numpy(self) -> np.ndarray:
        with torch.no_grad():
            return self().detach().cpu().double().numpy()


# ----------------------------------------------------------------------------------------- the frames
@dataclass
class Frames:
    """A batch of frames of one take, torch, metres."""
    take: str
    idx: np.ndarray                    # (F,) frame indices into the take
    T_world_from_knucklecap: torch.Tensor   # (F,4,4)
    tip_obs: torch.Tensor              # (F,m,3), 0 where missing
    tip_mask: torch.Tensor             # (F,m) bool
    float_obs: torch.Tensor            # (F,2,3)
    float_mask: torch.Tensor           # (F,2)
    knuckle_obs: torch.Tensor          # (F,m,3) -- for the report only: the root is this body's own fit
    knuckle_mask: torch.Tensor
    R_world_from_tipcap: torch.Tensor = None   # (F,3,3) the tip cap's re-solved rotation (distal_dir term)
    tipcap_valid: torch.Tensor = None          # (F,) bool


def make_frames(rig, idx: np.ndarray, device) -> Frames:
    kn, tp = rig.bodies["knuckle"], rig.bodies["tip"]
    T = np.zeros((len(idx), 4, 4))
    T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = kn.R_world_from_body_seq[idx], kn.t_world_from_body_seq[idx], 1.0
    tip = tp.marker_world_seq[idx]
    fl = np.swapaxes(rig.float_world_seq[:, idx], 0, 1)
    kno = kn.marker_world_seq[idx]
    f32 = lambda a: torch.as_tensor(np.nan_to_num(a), dtype=torch.float32, device=device)   # noqa: E731
    b = lambda a: torch.as_tensor(np.isfinite(a[..., 0]), device=device)                    # noqa: E731
    Rt = tp.R_world_from_body_seq[idx]
    return Frames(rig.take, np.asarray(idx), f32(T), f32(tip), b(tip), f32(fl), b(fl), f32(kno), b(kno),
                  f32(Rt), torch.as_tensor(tp.valid[idx] & np.isfinite(Rt[:, 0, 0]), device=device))


def valid_frames(rig) -> np.ndarray:
    """Frames where the root (knuckle cap) and the tip cap are both posed."""
    return np.flatnonzero(rig.bodies["knuckle"].valid & rig.bodies["tip"].valid)


# ------------------------------------------------------------------------------------- the calibration
class Calib(torch.nn.Module):
    """Every quantity shared across frames and takes."""

    def __init__(self, mano: ManoIndex, init: dict, tip_template: np.ndarray, base_pose: np.ndarray):
        super().__init__()
        dev = mano.device
        self.mano = mano
        self.betas0 = torch.tensor(init["betas"], dtype=torch.float32, device=dev)[None]
        self.betas = torch.nn.Parameter(self.betas0.clone())
        self.T_knucklecap_from_canonical = Delta(init["T_knucklecap_from_canonical"], dev)
        self.T_distal_from_tipcap = Delta(init["T_distal_from_tipcap"], dev)
        self.register_buffer("tip_template", torch.as_tensor(tip_template, dtype=torch.float32, device=dev))
        d = np.asarray(init.get("distal_dir_tipcap", [1.0, 0.0, 0.0]), np.float32)
        self.register_buffer("distal_dir_tipcap", torch.as_tensor(d / np.linalg.norm(d), device=dev))
        self.register_buffer("base_pose", torch.as_tensor(base_pose, dtype=torch.float32, device=dev)[None])
        fl = init["floating"]
        self.register_buffer("float_face", torch.tensor([fl[L]["face"] for L in FLOAT_LABELS], device=dev))
        bary = np.array([fl[L]["bary"] for L in FLOAT_LABELS], float)
        self.float_uv = torch.nn.Parameter(torch.as_tensor(bary[:, 1:], dtype=torch.float32, device=dev))
        self.float_h = torch.nn.Parameter(torch.tensor([fl[L]["height_mm"] / 1000.0 for L in FLOAT_LABELS],
                                                       dtype=torch.float32, device=dev))
        # optional correction of the index MCP/PIP/DIP rest pivots, confined to a ball (0 = off)
        self.joint_offset_radius = float(init.get("joint_offset_radius_m", 0.0))
        self.joint_offset_u = torch.nn.Parameter(torch.zeros(3, 3, device=dev))
        if "index_joint_offset_m" in init:
            r = max(self.joint_offset_radius, 1e-9)
            o = torch.as_tensor(np.asarray(init["index_joint_offset_m"], np.float32), device=dev)
            n = o.norm(dim=1, keepdim=True).clamp_max(0.999 * r)
            with torch.no_grad():                                     # invert r tanh(|u|) u_hat
                self.joint_offset_u.copy_(torch.atanh(n / r) * o / o.norm(dim=1, keepdim=True).clamp_min(1e-12))

    @property
    def index_joint_offset(self) -> torch.Tensor:
        """(3,3) m, MCP/PIP/DIP pivot corrections: r tanh(|u|) u_hat (TableMocap solve_plate_binding.ball_offset)."""
        if self.joint_offset_radius <= 0:
            return torch.zeros(3, 3, device=self.joint_offset_u.device)
        n = self.joint_offset_u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return (self.joint_offset_radius * torch.tanh(n) / n) * self.joint_offset_u

    def joint_offset_full(self) -> torch.Tensor | None:
        if self.joint_offset_radius <= 0:
            return None
        off = torch.zeros(16, 3, device=self.joint_offset_u.device)
        off[1:4] = self.index_joint_offset
        return off

    @property
    def float_bary(self) -> torch.Tensor:
        return torch.cat([1.0 - self.float_uv.sum(1, keepdim=True), self.float_uv], 1)

    def full_pose(self, index_pose: torch.Tensor) -> torch.Tensor:
        """(F,9) index joints -> (F,45) hand_pose, every other joint at the base (calibration) pose."""
        hp = self.base_pose.expand(index_pose.shape[0], -1).clone()
        hp[:, :9] = index_pose
        return hp

    def predict(self, fr: Frames, index_pose: torch.Tensor, verts: bool = False) -> dict:
        posed = self.mano.forward(self.betas, self.full_pose(index_pose), self.joint_offset_full())
        T_wc = fr.T_world_from_knucklecap @ self.T_knucklecap_from_canonical()[None]     # world <- canonical
        T_w_tipcap = T_wc @ posed.T_canonical_from_joint[:, 3] @ self.T_distal_from_tipcap()[None]
        tip = apply_t(T_w_tipcap, self.tip_template[None].expand(len(fr.idx), -1, -1))
        fl_can = skin_point(posed, self.mano.faces, self.float_face, self.float_bary, self.float_h)
        # distal bone direction, DIP joint -> index tip vertex, world
        d_can = posed.vertices[:, INDEX_TIP_VERTEX] - posed.T_canonical_from_joint[:, 3, :3, 3]
        d_can = d_can / d_can.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        out = {"tip": tip, "float": apply_t(T_wc, fl_can), "T_world_from_canonical": T_wc,
               "T_world_from_tipcap": T_w_tipcap, "distal_dir_world": torch.einsum("fij,fj->fi", T_wc[:, :3, :3], d_can)}
        if verts:
            out["vertices"] = apply_t(T_wc, posed.vertices)
            out["T_world_from_joint"] = T_wc[:, None] @ posed.T_canonical_from_joint
        return out

    def rebind_floats(self, index_pose_ref: torch.Tensor) -> list[int]:
        """Re-attach each floating marker to the face nearest its current point (pose `index_pose_ref` (1,9)),
        among the faces skinned to its phalanx. Keeps the point; changes the parametrisation. Returns faces."""
        from calibration.mano_index import closest_on_faces
        with torch.no_grad():
            posed = self.mano.forward(self.betas, self.full_pose(index_pose_ref), self.joint_offset_full())
            p = skin_point(posed, self.mano.faces, self.float_face, self.float_bary, self.float_h)[0].cpu().numpy()
            V = posed.vertices[0].cpu().numpy().astype(float)
            F = self.mano.faces.cpu().numpy()
            dom = self.mano.weights.argmax(1).cpu().numpy()
            faces = []
            for m, L in enumerate(FLOAT_LABELS):
                cand = np.flatnonzero((dom[F] == FLOAT_BONE[L]).any(1))
                fid, bary, h = closest_on_faces(p[m].astype(float), V, F, cand)
                self.float_face[m] = fid
                self.float_uv[m] = torch.as_tensor(bary[1:], dtype=torch.float32)
                self.float_h[m] = h
                faces.append(fid)
        return faces


# ------------------------------------------------------------------------------------------ the terms
def robust_mm(e_m: torch.Tensor, mask: torch.Tensor, scale_mm: float) -> torch.Tensor:
    """Geman-McClure on residual norms (mm) over masked entries, in mm^2 units."""
    d = e_m.norm(dim=-1) * 1000.0
    rho = (d ** 2) / (1.0 + (d / scale_mm) ** 2)
    m = mask.float()
    return (rho * m).sum() / m.sum().clamp_min(1.0)


# Human index ranges, degrees, about MANO canonical axes (x twist, y abduction, z flexion; +z = flex).
JOINT_RANGE_DEG = {
    "mcp": {"x": (-10.0, 10.0), "y": (-25.0, 25.0), "z": (-30.0, 100.0)},
    "pip": {"x": (-5.0, 5.0), "y": (-8.0, 8.0), "z": (-10.0, 120.0)},
    "dip": {"x": (-5.0, 5.0), "y": (-8.0, 8.0), "z": (-20.0, 100.0)},
}


def distal_dir_deg2(calib: "Calib", pred: dict, fr: Frames) -> torch.Tensor:
    """Mean squared angle (deg^2) between MANO's distal bone direction and the direction the measured tip cap
    implies, R_world_from_tipcap @ distal_dir_tipcap (Stage C, from the designed hand in the certain fixture).

    WHY. The tip cap reaches the distal bone only through the rigid binding T_distal_from_tipcap, and its
    markers sit within ~10 mm of each other, so the binding's rotation can absorb DIP flexion (seen
    2026-10-22: tracked DIP stayed within -11..18 deg while PIP reached 107 deg). Tying the bone direction to
    the cap's measured direction in EVERY frame removes that freedom from the binding and the pose alike."""
    d_meas = torch.einsum("fij,j->fi", fr.R_world_from_tipcap, calib.distal_dir_tipcap)
    c = (pred["distal_dir_world"] * d_meas).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    a = torch.rad2deg(torch.arccos(c))
    m = fr.tipcap_valid.float()
    return (a ** 2 * m).sum() / m.sum().clamp_min(1.0)


def joint_penalty(index_pose: torch.Tensor) -> torch.Tensor:
    """(F,9) -> scalar: squared degrees outside JOINT_RANGE_DEG (axis-angle components as angles)."""
    deg = torch.rad2deg(index_pose.view(-1, 3, 3))
    pen = 0.0
    for j, name in enumerate(("mcp", "pip", "dip")):
        for a, ax in enumerate("xyz"):
            lo, hi = JOINT_RANGE_DEG[name][ax]
            v = deg[:, j, a]
            pen = pen + (torch.relu(lo - v) ** 2 + torch.relu(v - hi) ** 2).mean()
    return pen


def init_index_pose(calib: Calib, fr: Frames, R_kn_tip_ref: np.ndarray, split=(0.45, 0.35, 0.20)) -> torch.Tensor:
    """(F,9) a flexion-only start from the tip cap's rotation relative to the knuckle cap, against the
    calibration take's (open-palm) relative rotation: the total flexion angle about canonical z is split
    MCP/PIP/DIP by `split`."""
    from scipy.spatial.transform import Rotation
    with torch.no_grad():
        T_kn_can = calib.T_knucklecap_from_canonical().cpu().double().numpy()
    R_can_kn = T_kn_can[:3, :3].T
    Tw = fr.T_world_from_knucklecap.cpu().double().numpy()
    # the tip cap pose comes from its markers; frames without enough tip markers get zero flexion
    out = np.zeros((len(fr.idx), 9))
    tip = fr.tip_obs.cpu().double().numpy()
    msk = fr.tip_mask.cpu().numpy()
    tk = calib.tip_template.cpu().double().numpy()
    from calibration.common import procrustes
    for i in range(len(fr.idx)):
        if msk[i].sum() < 3:
            continue
        R_wt, _ = procrustes(tk[msk[i]], tip[i][msk[i]])
        R_kt = Tw[i, :3, :3].T @ R_wt
        dR = R_can_kn @ (R_kt @ R_kn_tip_ref.T) @ R_can_kn.T
        # flexion angle about canonical z, unwrapped: a full fist turns the tip cap ~190 deg (MCP 85 + PIP 50
        # + DIP 60 measured at free_motion frame 7796), past where a rotation vector wraps, and the wrapped
        # value (-170 deg) started a 0.55 s run of the tracker in hyperextension at 13 mm tip error
        ang = float(np.arctan2(dR[1, 0] - dR[0, 1], dR[0, 0] + dR[1, 1]))
        if ang < np.radians(-60.0):
            ang += 2.0 * np.pi
        for j, s in enumerate(split):
            out[i, 3 * j + 2] = s * ang
    return torch.as_tensor(out, dtype=torch.float32, device=calib.betas.device)
