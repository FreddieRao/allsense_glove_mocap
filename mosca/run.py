#!/usr/bin/env python3
"""Stage B: MoSca marker re-identification and rigid-body re-solve for one Motive CSV take.

    take      = load_take(csv)                     every marker column, Motive's labels as weak priors
    seg       = tracklets.link(take.obs)           one marker followed through time
    sc        = resolve(take, seg)                 scaffold: groups -> naming -> LOCAL naming -> Motive seed
                                                   -> pose/verify/split/recover/partner loop
    templates = refine(sc)                         generalised Procrustes on frames with every row accepted
    sc        = resolve(take', seg)                again, with the take's own measured templates
    floating  = floating.name_floating(sc)         the two phalanx markers, by distance windows to the caps

and the poses that come out are `shape.consensus` fits on the scaffold's accepted markers -- Kabsch over the
largest consistent subset, nearest-previous tie-break -- in each body's Motive frame (the template is aligned
back to the asset template, so `R_world_from_<body>` means the same frame Motive's does).

WHY MOTIVE'S POSE IS NOT USED. Measured on calibration_1018, where the hand is rigid in the fixture, the
tip-cap rotation relative to the knuckle cap from Motive's own RB solve deviates from its median by p50 25 deg
and p90 175 deg: the 5-marker caps flip between near-symmetric solutions. Motive's labelled knuckle markers
sit p50 2.83 / p90 6.88 mm from its own pose there (0.21 / 0.40 mm on free_motion). The re-solve's
numbers are in the summary JSON this writes (`flip_check`).

TEMPLATE REFINEMENT. The `.motive` offsets are Motive's estimate at asset-creation time; a 0.5 mm row error
becomes a 0.5 mm floor on every downstream residual. The per-take template is the mean of the accepted
markers mapped into the body frame over frames where every row is accepted (generalised Procrustes,
`template.iterations` rounds), then Kabsch-aligned back onto the asset template so the body frame is kept.

    ~/miniforge3/envs/far_grasp/bin/python -m mosca.run --take calibration
    ~/miniforge3/envs/far_grasp/bin/python -m mosca.run --take free_motion

Writes outputs/<take>/mosca.npz and outputs/<take>/mosca.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config, strip_overrides                         # noqa: E402
from mosca import floating, scaffold, tracklets                         # noqa: E402
from mosca.shape import kabsch, turn_deg                                # noqa: E402
from mosca.take import Body, asset_paths_for, load_take                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def name_local(sc, take, nc, lc) -> dict:
    """Name free segments by template search on the markers AROUND each body's Motive position.

    WHY THE SCAFFOLD'S OWN NAMING IS NOT ENOUGH HERE. `scaffold.name_groups` names a RIGIDITY group, and in
    calibration_1018 the whole hand rests in the fixture: fixture, both caps and both floating markers are
    one rigid group (2 groups >= 2 segments on the take), which is named once -- as the fixture, 18 of its
    markers -- and the caps inside it are never offered to their own templates. `seed_from_motive` cannot
    rescue them either, because Motive swaps the knuckle cap's labels WITHIN one physical track: segment 23
    carries Motive row 4 on 7669 frames and row 1 on 1085, segment 24 rows 0/3 at 3887/3782, so no label
    reaches the 0.8 majority share and the knuckle posed from 3 markers on median.

    So the group is replaced by a spatial neighbourhood: at `lc.frames` frames spread over the take, every
    observation within the body's template radius + `lc.margin_mm` of Motive's body ORIGIN (its position
    only, which a flip barely moves -- never its rotation) is offered to `shape.place` against that body's
    template, prediction-free, exactly as `name_groups` does per frame. Each placed marker votes (body, row)
    for its segment; a segment takes the label when it holds `nc.row_share` of its votes over >= 3 frames.
    """
    from collections import Counter, defaultdict
    from mosca.shape import place
    obs = take.obs
    votes = defaultdict(Counter)
    frames = np.unique(np.linspace(0, take.n_frames - 1, int(lc.frames)).astype(int))
    for k, body in enumerate(take.bodies):
        mp = take.motive.get(body.name)
        if mp is None:
            continue
        reach = float(np.linalg.norm(body.tk, axis=1).max()) + float(lc.margin_mm)
        for f in frames:
            if not mp.valid[f]:
                continue
            sl = obs.at(int(f))
            near = np.flatnonzero(np.linalg.norm(obs.p[sl] - mp.t_world_from_body_mm_seq[f], axis=1) <= reach) + sl.start
            if len(near) < 3:
                continue
            near = near[:25]
            pl = place(obs.p[near], body, nc)
            if pl is None or pl[0] < min(int(nc.min_markers), len(body.tk)):
                continue
            for c, r in zip(pl[4], pl[5]):
                votes[int(sc.seg[near[c]])][(k, int(r))] += 1
    named, conflicts = 0, 0
    for s, cnt in votes.items():
        (key, c), = cnt.most_common(1)
        if c < 3 or c / sum(cnt.values()) < float(nc.row_share):
            continue
        if sc.body[s] >= 0:
            conflicts += int((sc.body[s], sc.row[s]) != key)
            continue
        sc.label(s, key[0], key[1], "local")
        named += 1
    return {"named": named, "group_disagree": conflicts, "segments_voted": len(votes)}


def resolve(take, seg, mc, grps=None, groups_report=None):
    """`scaffold.resolve` with `name_local` between group naming and the Motive seed (see name_local)."""
    sc = scaffold.build(take, seg, mc)
    sc.emf = {}
    if grps is None:
        grps = scaffold.groups(sc, mc.rigidity)
    else:
        sc.report["groups"] = groups_report or {}
    rep = {"naming": scaffold.name_groups(sc, grps, mc.naming)}
    rep["local"] = name_local(sc, take, mc.naming, mc.local)
    if bool(mc.naming.label_prior):
        rep["motive"] = scaffold.seed_from_motive(sc, mc.naming)
    rep["iterations"] = []
    for _it in range(int(mc.iterations)):
        scaffold.solve_poses(sc, mc.pose)
        scaffold.verify(sc, mc.pose)
        it = scaffold.split(sc, mc.split)
        scaffold.solve_poses(sc, mc.pose)
        scaffold.verify(sc, mc.pose)
        it["recovered"] = scaffold.recover(sc, mc.pose, mc.recover)
        it["partnered"] = scaffold.partner(sc, mc.partner, mc.rigidity)
        rep["iterations"].append(it)
    scaffold.solve_poses(sc, mc.pose)
    scaffold.verify(sc, mc.pose)
    sc.report.update(rep)
    sc.groups_cache = grps
    return sc


def refine_templates(sc, take, tc) -> dict[str, tuple[np.ndarray, float]]:
    """{body: (refined tk (m,3) mm, mean |moved| mm)} from frames where every row is accepted."""
    obs = take.obs
    out = {}
    for k, body in enumerate(take.bodies):
        m = len(body.tk)
        ps = sc.poses[k]
        i = np.flatnonzero(sc.acc & (sc.body[sc.seg] == k))
        f, r = obs.frame[i], sc.row[sc.seg[i]]
        Y = np.full((take.n_frames, m, 3), np.nan)
        Y[f, r] = obs.p[i]
        seen = np.isfinite(Y[..., 0])
        fr = np.flatnonzero((seen.sum(1) >= min(int(tc.min_markers), m)) & ps.valid)
        if len(fr) < 50:
            continue
        if len(fr) > int(tc.max_frames):
            fr = fr[np.linspace(0, len(fr) - 1, int(tc.max_frames)).astype(int)]
        tk = body.tk.copy()
        for _it in range(int(tc.iterations)):
            acc, cnt = np.zeros((m, 3)), np.zeros(m)
            for fi in fr:
                rows = np.flatnonzero(seen[fi])
                R, t, _ = kabsch(tk[rows], Y[fi, rows])
                acc[rows] += (Y[fi, rows] - t) @ R           # the markers in the body frame
                cnt[rows] += 1
            tk_new = np.where(cnt[:, None] >= 50, acc / np.maximum(cnt, 1)[:, None], tk)
            R0, t0, _ = kabsch(tk_new, body.tk)               # keep the asset's body frame
            tk = tk_new @ R0.T + t0
        out[body.name] = (tk, float(np.linalg.norm(tk - body.tk, axis=1).mean()))
    return out


def flip_check(take, poses: dict, knuckle: str, tip: str) -> dict:
    """Orientation stability of the tip cap relative to the knuckle cap, Motive vs re-solved.

    On the calibration take the true relative rotation is (near) constant, so the deviation from its median
    IS the error. On free_motion the finger moves, so only the frame-to-frame step is reported."""
    def rel(Rk, Rt, ok):
        Rrel = np.einsum("tji,tjk->tik", Rk[ok], Rt[ok])
        return Rrel

    out = {}
    for src in ("motive", "mosca"):
        Rk, okk = poses[src][knuckle]
        Rt, okt = poses[src][tip]
        ok = okk & okt
        Rrel = rel(Rk, Rt, ok)
        from scipy.spatial.transform import Rotation
        med = Rotation.from_matrix(Rrel).mean().as_matrix()
        dev = np.array([turn_deg(med, R) for R in Rrel])
        step = np.array([turn_deg(a, b) for a, b in zip(Rrel[:-1], Rrel[1:])])
        out[src] = {"frames": int(ok.sum()),
                    "rel_rot_dev_from_mean_deg_p50_p90_p99": [round(float(x), 2) for x in np.percentile(dev, [50, 90, 99])],
                    "step_deg_p99": round(float(np.percentile(step, 99)), 2),
                    "steps_over_20deg": int((step > 20).sum())}
    return out


def run(take_key: str, cfg) -> dict:
    t0 = time.perf_counter()
    mc = cfg.mosca
    csv_path = ROOT / getattr(cfg.data.takes, take_key)
    take = load_take(csv_path, asset_paths_for(cfg))
    seg = tracklets.link(take.obs, take.rate_hz, mc.tracklets)
    trk = tracklets.summary(seg, take.rate_hz)
    sc = resolve(take, seg, mc)
    print(f"[{take_key}] pass 1 {time.perf_counter() - t0:.1f}s  groups {sc.report.get('groups')}", flush=True)

    refined = refine_templates(sc, take, mc.template)
    for name, (tk, moved) in refined.items():
        k = take.body_index(name)
        b = take.bodies[k]
        take.bodies[k] = Body(b.name, b.kind, tk, b.known, thin=b.thin)
        print(f"[{take_key}] template {name:22s} mean |moved from asset| {moved:.2f} mm", flush=True)
    sc = resolve(take, seg, mc, grps=sc.groups_cache, groups_report=sc.report.get("groups"))
    print(f"[{take_key}] pass 2 {time.perf_counter() - t0:.1f}s", flush=True)

    B = cfg.data.bodies
    flab, frep = floating.name_floating(sc, take, B.knuckle, B.tip, mc.floating)
    fl_world = floating.tracks(take, flab)

    obs = take.obs
    body_of_obs = np.where(sc.acc, sc.body[sc.seg], -1)
    row_of_obs = np.where(sc.acc, sc.row[sc.seg], -1)
    arrays = {"frame": obs.frame, "p_world_mm": obs.p.astype(np.float32), "body": body_of_obs.astype(np.int8),
              "row": row_of_obs.astype(np.int8), "float_label": flab.astype(np.int8), "seg": sc.seg.astype(np.int32),
              "motive_owner": obs.owner.astype(np.int8), "float_p_world_mm_seq": fl_world.astype(np.float32),
              "body_names": np.array([b.name for b in take.bodies]), "rate_hz": take.rate_hz}
    per_body, poses = {}, {"motive": {}, "mosca": {}}
    for k, b in enumerate(take.bodies):
        ps = sc.poses[k]
        mp = take.motive[b.name]
        tag = b.name
        arrays[f"{tag}__valid"] = ps.valid
        arrays[f"{tag}__R_world_from_body_seq"] = ps.R.astype(np.float32)
        arrays[f"{tag}__t_world_from_body_mm_seq"] = ps.t.astype(np.float32)
        arrays[f"{tag}__rmse_mm"] = ps.rmse.astype(np.float32)
        arrays[f"{tag}__n"] = ps.n.astype(np.int8)
        arrays[f"{tag}__template_mm"] = b.tk
        arrays[f"{tag}__motive_valid"] = mp.valid
        arrays[f"{tag}__motive_R_world_from_body_seq"] = mp.R_world_from_body_seq.astype(np.float32)
        arrays[f"{tag}__motive_t_world_from_body_mm_seq"] = mp.t_world_from_body_mm_seq.astype(np.float32)
        poses["motive"][b.name] = (mp.R_world_from_body_seq, mp.valid)
        poses["mosca"][b.name] = (ps.R, ps.valid)
        good = ps.valid
        per_body[b.name] = {"posed_frac": round(float(good.mean()), 4),
                            "rmse_mm_p50_p90_p99": [round(float(x), 3) for x in np.percentile(ps.rmse[good], [50, 90, 99])],
                            "markers_p50": float(np.median(ps.n[good])),
                            "template_moved_mm": round(refined[b.name][1], 3) if b.name in refined else None}
    cov = {"observations": len(obs), "assigned_rigid": int((body_of_obs >= 0).sum()),
           "floating": int((flab >= 0).sum()), "static_background": int(sc.static[sc.seg].sum())}
    summary = {"take": take_key, "csv": str(csv_path.relative_to(ROOT)), "frames": take.n_frames,
               "rate_hz": take.rate_hz, "seconds": round(time.perf_counter() - t0, 1), "tracklets": trk,
               "coverage": cov, "bodies": per_body, "floating": frep,
               "flip_check": flip_check(take, poses, B.knuckle, B.tip),
               "scaffold": {k: v for k, v in sc.report.items() if k in ("groups", "naming", "local", "motive", "iterations")}}
    out_dir = ROOT / cfg.data.out_root / take_key
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "mosca.npz", **arrays)
    (out_dir / "mosca.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.tolist()
                                                   if isinstance(o, np.ndarray) else str(o)))
    print(f"wrote {out_dir / 'mosca.npz'}\nwrote {out_dir / 'mosca.json'}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--take", nargs="+", default=["calibration", "free_motion"])
    args = ap.parse_args(strip_overrides())
    cfg = load_config()
    for t in args.take:
        s = run(t, cfg)
        print(json.dumps({k: s[k] for k in ("coverage", "bodies", "floating", "flip_check")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
