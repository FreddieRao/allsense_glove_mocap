#!/usr/bin/env python3
"""Stage A: one Motive CSV take as the MoSca scaffold sees it -- every marker observation plus every template.

THE INPUT IS A MOTIVE CSV EXPORT (format 1.25), not the NatNet `motive.jsonl` the TableMocap scaffold was built
on, so this module replaces `fitting/reassignment/frames.py:load_take` and keeps its containers (`Obs`, `Body`,
`Take`) field for field -- `mosca/scaffold.py` is vendored unchanged and reads exactly those fields.

The CSV has a 7-row header (take metadata, blank, Type, Name, ID, Parent, Rotation/Position, X/Y/Z/W) and one
row per frame. Columns that matter:

    Type == "Rigid Body"   Motive's solved pose: quaternion X,Y,Z,W then position X,Y,Z (mm, world)
    Type == "Marker"       a MEASURED marker: "<body>:Marker NNN" (Motive's label, asset row NNN-1) or
                           "Unlabeled NNNN" (no label; NNNN is Motive's own short-lived track id)

Motive's label rides along only as weak, per-SEGMENT evidence (`scaffold.seed_from_motive`). It cannot be
trusted per frame here: on calibration_1018 -- a hand held rigid in the fixture -- the fingertip cap's
orientation relative to the knuckle cap jumps by up to 175 deg (p90 175 deg), i.e. Motive flips the 5-marker
caps between near-symmetric solutions, so the label and the pose it implies are wrong together.

Templates are the `.motive` asset's constraint offsets (body frame, m -> mm), parsed with the function from
`data_io/asset_parser.py:parse_bodies`. EVERY body is tagged `kind="object"`: the scaffold's "cap"/"plate"
kinds gate a cap on a glove's dorsal plate (`plate_motive`, `cap_anchor`), and this rig has no glove.

Positions stay in mm in this package because the vendored scaffold's gates are all in mm; the calibration
package converts to metres at its boundary (`_mm` names say so).

    ~/miniforge3/envs/far_grasp/bin/python -m mosca.take data/motive/calibration_1018.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THIN_LEVER = 20.0          # length / off-axis spread above which roll is not measured (frames.py's threshold)
_LABELED = re.compile(r"^(?P<body>.+):Marker (?P<num>\d+)$")


@dataclass
class AssetBody:
    """One rigid body of a `.motive` asset export, body frame, metres."""
    name: str
    marker_pos_body: np.ndarray        # (M,3) m, row j = asset member j+1
    marker_diameter_m: np.ndarray      # (M,)


def parse_bodies(path: str | Path) -> dict[str, AssetBody]:
    """Every rigid body in a `.motive` asset export (from TableMocap `data_io/asset_parser.py:parse_bodies`)."""
    bodies: dict[str, AssetBody] = {}
    for rb in ET.parse(Path(path)).getroot().findall(".//rigid_body"):
        name = next((p.get("value") for p in rb.findall(".//property") if p.get("name") == "NodeName"), None)
        if not name:
            continue
        pos, diam = [], []
        for c in rb.findall(".//constraint"):
            o = c.find("offset")
            if o is None or not o.text:
                continue
            pos.append([float(v) for v in o.text.split(",")])
            d = c.find("diameter")
            diam.append(float(d.text) if d is not None and d.text else np.nan)
        bodies[name] = AssetBody(name, np.asarray(pos, float).reshape(-1, 3), np.asarray(diam, float))
    return bodies


@dataclass
class Body:
    """One body's template in its own frame, mm, compact rows (frames.py's container, unchanged)."""
    name: str
    kind: str                          # 'plate' | 'cap' | 'object' -- always 'object' on this rig
    tk: np.ndarray                     # (m,3) known rows, mm
    known: np.ndarray                  # (m,) asset row of each compact row
    side: str | None = None
    finger: str | None = None
    thin: bool = False
    D: np.ndarray = field(default=None)  # (m,m) pairwise template distances

    def __post_init__(self):
        self.D = np.linalg.norm(self.tk[:, None] - self.tk[None], axis=2)

    @property
    def diam_mm(self) -> float:
        return float(self.D.max()) if len(self.tk) > 1 else 0.0

    def at(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.tk @ R.T + t


@dataclass
class Obs:
    """Every distinct marker observation of the take, sorted by frame."""
    frame: np.ndarray                  # (N,)
    p: np.ndarray                      # (N,3) world mm
    owner: np.ndarray                  # (N,) index into Take.names of Motive's label, -1 none
    row: np.ndarray                    # (N,) Motive's 0-based ASSET row, -1 none
    ptr: np.ndarray                    # (T+1,)
    col: np.ndarray = None             # (N,) CSV marker column it came from -- provenance for the viewer

    def __len__(self) -> int:
        return len(self.frame)

    def at(self, i: int) -> slice:
        return slice(int(self.ptr[i]), int(self.ptr[i + 1]))


@dataclass
class MotivePose:
    """Motive's own rigid-body solve, straight from the CSV. mm, world-from-body."""
    valid: np.ndarray                  # (T,) finite, non-zero quaternion
    R_world_from_body_seq: np.ndarray  # (T,3,3)
    t_world_from_body_mm_seq: np.ndarray  # (T,3)


@dataclass
class Take:
    name: str
    names: list                        # every body name in the CSV; Obs.owner indexes this
    bodies: list                       # [Body] the bodies with a template
    obs: Obs
    frame_idx: np.ndarray              # (T,) Motive's frame numbers
    rate_hz: float
    motive: dict                       # {body: MotivePose}
    asset: dict                        # {body: AssetBody}
    marker_cols: list                  # CSV marker column names, Obs.col indexes this
    plate_motive: dict = field(default_factory=dict)   # unused on this rig (no glove plates)
    cap_anchor: dict = field(default_factory=dict)     # unused on this rig
    report: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frame_idx)

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz

    def frames_of(self, seconds: float) -> int:
        """A duration as a whole number of this take's frames, at least 1."""
        return max(1, int(round(seconds * self.rate_hz)))

    def body_index(self, name: str) -> int:
        return next(k for k, b in enumerate(self.bodies) if b.name == name)


def _quat_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    """(T,4) -> (T,3,3); invalid rows are NaN. Local so a zero quaternion does not raise inside scipy."""
    from scipy.spatial.transform import Rotation
    out = np.full((len(q_xyzw), 3, 3), np.nan)
    n = np.linalg.norm(q_xyzw, axis=1)
    ok = np.isfinite(n) & (n > 0.5)
    if ok.any():
        out[ok] = Rotation.from_quat(q_xyzw[ok]).as_matrix()
    return out


def read_csv(path: Path):
    """(meta dict, type row, name row, sub row, axis row, data (T,C) float with NaN for blanks)."""
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    meta = {head[i]: head[i + 1] for i in range(0, len(head) - 1, 2)}
    typ, name, sub, axis = rows[2], rows[3], rows[6], rows[7]
    body = [r for r in rows[8:] if r]
    width = len(axis)
    data = np.full((len(body), width), np.nan)
    for i, r in enumerate(body):
        for j, x in enumerate(r[:width]):
            if x:
                data[i, j] = float(x)
    return meta, typ, name, sub, axis, data


def _thin(tk: np.ndarray) -> bool:
    c = tk - tk.mean(0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    along = c @ vt[0]
    off = np.linalg.norm(c - np.outer(along, vt[0]), axis=1)
    return bool(np.ptp(along) / max(2 * off.max(), 1e-9) >= THIN_LEVER)


def load_take(csv_path: str | Path, asset_paths: list[str | Path]) -> Take:
    """The take at its native rate, every marker column as an observation, Motive's labels as weak priors."""
    csv_path = Path(csv_path)
    meta, typ, name, sub, axis, data = read_csv(csv_path)
    asset: dict[str, AssetBody] = {}
    for p in asset_paths:
        asset.update(parse_bodies(p))
    rate_hz = float(meta.get("Export Frame Rate", meta.get("Capture Frame Rate", "240")))
    T = data.shape[0]

    names = sorted({name[j] for j in range(2, len(typ)) if typ[j] == "Rigid Body"} | set(asset))
    idx_name = {n: k for k, n in enumerate(names)}
    motive = {}
    for n in names:
        cols = [j for j in range(2, len(typ)) if typ[j] == "Rigid Body" and name[j] == n]
        if not cols:
            continue
        q = {(sub[j], axis[j]): j for j in cols}
        quat = data[:, [q[("Rotation", a)] for a in "XYZW"]]
        pos = data[:, [q[("Position", a)] for a in "XYZ"]]
        R = _quat_to_R(quat)
        motive[n] = MotivePose(np.isfinite(R[:, 0, 0]) & np.isfinite(pos).all(1), R, pos)

    # marker columns come in X,Y,Z triples under one name
    mk = [j for j in range(2, len(typ)) if typ[j] == "Marker" and axis[j] == "X"]
    marker_cols = [name[j] for j in mk]
    P = np.stack([data[:, [j, j + 1, j + 2]] for j in mk], axis=1)          # (T,C,3) mm
    owner = np.full(len(mk), -1)
    row = np.full(len(mk), -1)
    for c, n in enumerate(marker_cols):
        m = _LABELED.match(n)
        if m and m["body"] in idx_name:
            owner[c], row[c] = idx_name[m["body"]], int(m["num"]) - 1
    vis = np.isfinite(P[..., 0])
    F, C = np.nonzero(vis)                                                 # row-major: sorted by frame
    # labels first within a frame, then de-duplicate at 1 um exactly as frames.observations does
    order = np.lexsort((owner[C] < 0, F))
    F, C = F[order], C[order]
    p = P[F, C]
    keep = np.ones(len(F), bool)
    ptr = np.searchsorted(F, np.arange(T + 1))
    for i in range(T):
        a, b = ptr[i], ptr[i + 1]
        if b - a < 2:
            continue
        _u, first = np.unique(np.round(p[a:b], 3), axis=0, return_index=True)
        k = np.zeros(b - a, bool)
        k[first] = True
        keep[a:b] = k
    F, C, p = F[keep], C[keep], p[keep]
    obs = Obs(F.astype(int), p, owner[C], row[C], np.searchsorted(F, np.arange(T + 1)), C)

    bodies = []
    for n in names:
        if n not in asset:
            continue
        tk = asset[n].marker_pos_body * 1000.0
        bodies.append(Body(n, "object", tk, np.arange(len(tk)), thin=_thin(tk)))
    frame_col = np.arange(T)
    report = {"csv": str(csv_path), "frames": T, "rate_hz": rate_hz, "observations": len(obs),
              "marker_columns": len(mk), "labelled_columns": int((owner >= 0).sum()),
              "bodies": {b.name: len(b.tk) for b in bodies},
              "motive_valid": {n: round(float(m.valid.mean()), 4) for n, m in motive.items()}}
    return Take(csv_path.stem, names, bodies, obs, frame_col, rate_hz, motive, asset, marker_cols, report=report)


def asset_paths_for(cfg) -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [root / p for p in cfg.data.assets]


def main() -> None:
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config, strip_overrides
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    take = load_take(args.csv, asset_paths_for(cfg))
    print(json.dumps(take.report, indent=2))
    # Motive's own residual: labelled markers against its RB pose and the asset template
    for b in take.bodies:
        k = take.names.index(b.name)
        mp = take.motive[b.name]
        on = (take.obs.owner == k) & (take.obs.row >= 0)
        f, r = take.obs.frame[on], take.obs.row[on]
        ok = mp.valid[f]
        pred = np.einsum("nij,nj->ni", mp.R_world_from_body_seq[f[ok]], b.tk[r[ok]]) + mp.t_world_from_body_mm_seq[f[ok]]
        e = np.linalg.norm(pred - take.obs.p[on][ok], axis=1)
        print(f"{b.name:22s} labelled obs {on.sum():7d}  |Motive pose x template - marker| mm "
              f"p50 {np.median(e):6.2f}  p90 {np.percentile(e, 90):6.2f}  p99 {np.percentile(e, 99):7.2f}", flush=True)


if __name__ == "__main__":
    main()
