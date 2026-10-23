#!/usr/bin/env python3
"""MANO (smplx, right hand) with what the index-finger calibration needs: bone frames, skin points, normals.

The CANONICAL frame is MANO's own output frame with `global_orient = 0` and `transl = 0`; the wrist joint is
NOT at the origin (~[95.7, 6.4, 6.2] mm). Verified 2026-10-22: `mano_clean/MANO_RIGHT.pkl` with
`flat_hand_mean=True`, the 45-D axis-angle `hand_pose` and `custom_pose.json`'s betas/pose reproduces
`data/assets/custom_pose_metric.obj` to 7e-6 mm (flat_hand_mean=False misses by 69 mm). Units are METRES.

BONE FRAMES. LBS moves every point rigidly attached to joint j by A_j = [R_j | J_posed_j - R_j J_rest_j], so
a point fixed to bone j has one constant coordinate in the frame

    T_canonical_from_joint[j] = [ R_j | J_posed_j ],      p_joint = p_rest - J_rest_j,

whose axes are MANO's canonical axes at rest. That is the frame the caps are bound to:

    p_canonical = T_canonical_from_joint[3] @ T_distal_from_tipcap @ x_tipcap        (index DIP = joint 3)

MANO's 16 joints: 0 wrist, then (MCP, PIP, DIP) per finger in the order index 1-3, middle 4-6, pinky 7-9,
ring 10-12, thumb 13-15. The index tip is VERTEX 320 (it is `custom_pose.json`'s index-tip keypoint).

SKIN POINTS. A floating marker sits on the skin, so it is bound to the SURFACE, not to a bone: a face, a
barycentric point on it, and a height along the interpolated vertex normal,

    p_marker = sum_i b_i v_i(theta, beta) + h * n(theta, beta),

which follows the pose blend shapes and the skinning exactly as the skin it is glued to does. Orientation:
the hand lies palm-down in the fixture and `T_cad_from_canonical` maps canonical +y to CAD +z, so the
DORSUM faces canonical +y.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

INDEX_JOINTS = (1, 2, 3)          # MCP, PIP, DIP
INDEX_TIP_VERTEX = 320
MANO_BONE = {"wrist": 0, "proximal": 1, "middle": 2, "distal": 3}


def rodrigues(r: torch.Tensor) -> torch.Tensor:
    """(...,3) axis-angle -> (...,3,3)."""
    from smplx.lbs import batch_rodrigues
    return batch_rodrigues(r.reshape(-1, 3)).reshape(*r.shape[:-1], 3, 3)


@dataclass
class Posed:
    vertices: torch.Tensor            # (B,778,3) canonical, m
    T_canonical_from_joint: torch.Tensor   # (B,16,4,4)
    J_rest: torch.Tensor              # (B,16,3) shaped rest joints
    normals: torch.Tensor             # (B,778,3) unit vertex normals


class ManoIndex:
    """smplx MANO right layer, flat_hand_mean, no PCA, with LBS exposed so the bone transforms come out."""

    def __init__(self, model_dir: str, device: str | None = None):
        import smplx
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = smplx.MANO(model_dir, is_rhand=True, use_pca=False, flat_hand_mean=True).to(self.device)
        L = self.layer
        self.faces = torch.as_tensor(np.asarray(L.faces, np.int64), device=self.device)
        self.parents = L.parents
        self.weights = L.lbs_weights                                    # (778,16)
        dom = self.weights.argmax(1).cpu().numpy()
        f = L.faces
        # faces whose three vertices are all skinned mostly to one index bone: the candidate skin per phalanx
        self.bone_faces = {j: np.flatnonzero((dom[f] == j).all(1)) for j in range(16)}

    def forward(self, betas: torch.Tensor, hand_pose: torch.Tensor, joint_offset: torch.Tensor | None = None) -> Posed:
        """betas (1|B,10), hand_pose (B,45) -> Posed, canonical frame.

        `joint_offset` (16,3) m moves the REST joint centres the skin rotates about (TableMocap
        hand_scan_processing/fit_mano.py:lbs_with_joint_offset): the regressed MANO joint is an average hand's
        pivot, and this hand's real pivot need not sit there. The rest surface is unchanged."""
        from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints
        L = self.layer
        B = hand_pose.shape[0]
        if betas.shape[0] == 1 and B > 1:
            betas = betas.expand(B, -1)
        v_shaped = L.v_template[None] + blend_shapes(betas, L.shapedirs)
        J = vertices2joints(L.J_regressor, v_shaped)                   # (B,16,3)
        if joint_offset is not None:
            J = J + joint_offset[None]
        full = torch.cat([torch.zeros(B, 3, device=hand_pose.device), hand_pose], 1)
        rot = rodrigues(full.view(B, 16, 3))                            # (B,16,3,3)
        eye = torch.eye(3, device=hand_pose.device)
        pose_feat = (rot[:, 1:] - eye).reshape(B, -1)
        v_posed = v_shaped + (pose_feat @ L.posedirs).view(B, -1, 3)
        J_posed, A = batch_rigid_transform(rot, J, self.parents)
        T = torch.einsum("vj,bjik->bvik", self.weights, A)
        vh = torch.cat([v_posed, torch.ones_like(v_posed[..., :1])], -1)
        verts = torch.einsum("bvik,bvk->bvi", T, vh)[..., :3]
        Tj = A.clone()
        Tj[:, :, :3, 3] = J_posed
        return Posed(verts, Tj, J, vertex_normals(verts, self.faces))

    def rest_joints(self, betas: torch.Tensor) -> torch.Tensor:
        from smplx.lbs import blend_shapes, vertices2joints
        L = self.layer
        return vertices2joints(L.J_regressor, L.v_template[None] + blend_shapes(betas, L.shapedirs))

    def index_bone_lengths_mm(self, betas: torch.Tensor) -> dict[str, float]:
        """Pose-invariant index readouts: wrist-MCP, MCP-PIP, PIP-DIP, DIP-tip (tip = vertex 320 at zero pose)."""
        with torch.no_grad():
            P = self.forward(betas[:1], torch.zeros(1, 45, device=betas.device))
            J = P.T_canonical_from_joint[0, :, :3, 3]
            tip = P.vertices[0, INDEX_TIP_VERTEX]
            pts = [J[0], J[1], J[2], J[3], tip]
            names = ["wrist_mcp", "proximal", "middle", "distal_to_tip"]
            return {n: float(torch.linalg.norm(b - a)) * 1000.0 for n, a, b in zip(names, pts[:-1], pts[1:])}


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """(B,V,3) area-weighted unit vertex normals."""
    v0, v1, v2 = (verts[:, faces[:, k]] for k in range(3))
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    n = torch.zeros_like(verts)
    for k in range(3):
        n.index_add_(1, faces[:, k], fn)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def skin_point(posed: Posed, faces: torch.Tensor, face_id: torch.Tensor, bary: torch.Tensor,
               height: torch.Tensor) -> torch.Tensor:
    """(B,M,3) points `height` above barycentric `bary` (M,3) of faces `face_id` (M,) along the normal."""
    tri = faces[face_id]                                               # (M,3)
    v = posed.vertices[:, tri]                                         # (B,M,3,3)
    n = posed.normals[:, tri]
    p = (bary[None, :, :, None] * v).sum(2)
    nn = (bary[None, :, :, None] * n).sum(2)
    nn = nn / nn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return p + height[None, :, None] * nn


def closest_on_faces(p: np.ndarray, V: np.ndarray, F: np.ndarray, cand: np.ndarray) -> tuple[int, np.ndarray, float]:
    """(face id, barycentric (3,), signed distance along the face normal) of the point of faces `cand`
    nearest p. numpy; used to (re)bind a floating marker to the skin."""
    import trimesh
    tri = V[F[cand]]
    cp = trimesh.triangles.closest_point(tri, np.repeat(p[None], len(cand), 0))
    d = np.linalg.norm(cp - p, axis=1)
    j = int(np.argmin(d))
    bary = trimesh.triangles.points_to_barycentric(tri[j:j + 1], cp[j:j + 1])[0]
    n = np.cross(tri[j, 1] - tri[j, 0], tri[j, 2] - tri[j, 0])
    n /= np.linalg.norm(n)
    return int(cand[j]), bary, float((p - cp[j]) @ n)
