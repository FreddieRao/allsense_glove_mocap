#!/usr/bin/env python3
"""The two single large markers on the index phalanges: named by where they sit relative to the two caps.

WHY THE SCAFFOLD CANNOT NAME THEM. A scaffold body needs a template of >= 3 rows to be placed (`shape.place`)
and >= 4 markers to be named (`naming.min_markers`); a single marker glued to a phalanx is neither -- it has no
rigid partner at all, because the phalanx it sits on moves relative to both caps. So after `scaffold.resolve`
they are simply free segments, and this module names them.

THE RULE IS A DISTANCE WINDOW TO BOTH POSED CAPS. The proximal-phalanx marker (between MCP and PIP) stays
near the knuckle cap and far from the tip cap; the middle-phalanx marker (between PIP and DIP) is the reverse.
Measured from the unlabeled tracks (median, mm, to the knuckle / tip cap origin):

    calibration_1018   proximal 39 / 55     middle 67 / 27     (hand rigid: both bands < 8 mm wide)
    free_motion        proximal 45 / 59     middle 74 / 30     (finger moves: bands 9-15 mm wide)

and the nearest foreign markers (the caps' own rows) are within 15 mm of their cap's origin, so a window that
starts above that cannot claim a cap marker. A segment is a candidate for a label when the p5-p95 of BOTH
distances over its posed frames sits inside that label's windows (`mosca.floating.windows_mm`).

ONE OBSERVATION PER LABEL PER FRAME. Where two candidate segments overlap, the frame keeps the one nearest the
label's running median (dk, dt) -- a ghost reflection on the glove sits at a different distance.

A SECOND PASS ONCE MANO EXISTS. The windows only say "on this finger"; after the first MANO fit the predicted
marker position is far sharper, and `reclaim` re-assigns every frame by Hungarian matching against it within
`reclaim_gate_mm`. That is the scaffold's `recover` idea, with MANO as the template.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

LABELS = ("proximal", "middle")


def _cap_dist(sc, take, k: int, f: np.ndarray, p: np.ndarray) -> np.ndarray:
    ps = sc.poses[k]
    out = np.full(len(f), np.nan)
    on = ps.valid[f]
    out[on] = np.linalg.norm(p[on] - ps.t[f[on]], axis=1)
    return out


def name_floating(sc, take, knuckle: str, tip: str, fc) -> tuple[np.ndarray, dict]:
    """(N,) label per observation (-1 none, 0 proximal, 1 middle) and a report."""
    obs = take.obs
    kk, kt = take.body_index(knuckle), take.body_index(tip)
    free = sc.body < 0
    win = fc.windows_mm.to_dict()
    min_n = take.frames_of(float(fc.min_segment_s))
    lab = np.full(len(obs), -1)
    cand = {L: [] for L in LABELS}
    for s in np.flatnonzero(free):
        idx = sc.seg_obs[int(s)]
        if len(idx) < min_n:
            continue
        f, p = obs.frame[idx], obs.p[idx]
        dk, dt = _cap_dist(sc, take, kk, f, p), _cap_dist(sc, take, kt, f, p)
        ok = np.isfinite(dk) & np.isfinite(dt)
        if ok.sum() < min_n:
            continue
        qk, qt = np.percentile(dk[ok], [5, 95]), np.percentile(dt[ok], [5, 95])
        for j, L in enumerate(LABELS):
            wk, wt = win[L]["knuckle"], win[L]["tip"]
            if wk[0] <= qk[0] and qk[1] <= wk[1] and wt[0] <= qt[0] and qt[1] <= wt[1]:
                cand[L].append((int(s), idx, dk, dt))
    rep = {}
    for j, L in enumerate(LABELS):
        if not cand[L]:
            rep[L] = {"segments": 0, "frames": 0}
            continue
        allk = np.concatenate([c[2] for c in cand[L]])
        allt = np.concatenate([c[3] for c in cand[L]])
        ref = np.array([np.nanmedian(allk), np.nanmedian(allt)])
        best = np.full(take.n_frames, np.inf)
        pick = np.full(take.n_frames, -1)
        for _s, idx, dk, dt in cand[L]:
            f = obs.frame[idx]
            cost = np.hypot(np.nan_to_num(dk - ref[0], nan=50.0), np.nan_to_num(dt - ref[1], nan=50.0))
            better = cost < best[f]
            best[f[better]] = cost[better]
            pick[f[better]] = idx[better]
        sel = pick[pick >= 0]
        lab[sel] = j
        rep[L] = {"segments": len(cand[L]), "frames": int(len(sel)),
                  "median_dist_knuckle_tip_mm": [round(float(x), 1) for x in ref]}
    return lab, rep


def reclaim(frame: np.ndarray, p_mm: np.ndarray, n_frames: int, label: np.ndarray,
            p_pred_world_mm_seq: np.ndarray, gate_mm: float, taken: np.ndarray) -> tuple[np.ndarray, dict]:
    """Re-assign the floating labels against a MANO prediction.

    `frame`, `p_mm` (N,), (N,3) the take's observations sorted by frame; `p_pred_world_mm_seq` (2,T,3), NaN
    = no prediction; `taken` (N,) observations a rigid body owns, never claimed. Per frame, Hungarian on
    distance within `gate_mm`. Returns (new label (N,), report)."""
    out = np.full(len(frame), -1)
    ptr = np.searchsorted(frame, np.arange(n_frames + 1))
    for i in range(n_frames):
        a, b = ptr[i], ptr[i + 1]
        cand = np.flatnonzero(~taken[a:b]) + a
        pr = p_pred_world_mm_seq[:, i]
        have = np.isfinite(pr[:, 0])
        if not len(cand) or not have.any():
            continue
        D = np.linalg.norm(pr[have][:, None] - p_mm[cand][None], axis=2)
        r, c = linear_sum_assignment(np.where(D <= gate_mm, D, 1e6))
        Ls = np.flatnonzero(have)
        for rr, cc in zip(r, c):
            if D[rr, cc] <= gate_mm:
                out[cand[cc]] = Ls[rr]
    rep = {L: int((out == j).sum()) for j, L in enumerate(LABELS)}
    rep["window_labels"] = {L: int((label == j).sum()) for j, L in enumerate(LABELS)}
    rep["changed_vs_windows"] = int((out != label).sum())
    return out, rep


def tracks(take, label: np.ndarray) -> np.ndarray:
    """(2,T,3) world mm per label, NaN where unseen."""
    out = np.full((len(LABELS), take.n_frames, 3), np.nan)
    for j in range(len(LABELS)):
        i = np.flatnonzero(label == j)
        out[j, take.obs.frame[i]] = take.obs.p[i]
    return out
