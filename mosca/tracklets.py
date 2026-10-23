#!/usr/bin/env python3
# VENDORED from TableMocap-toolkit-padab/kinematics-replay-bimanual/fitting/reassignment/tracklets.py
# (last change 2edada7, 2026-10-16). Keep in sync by hand; only the import path is changed.
"""Stage 1: one marker followed through time -- MoSca's "curves", which here need no depth lifting.

WHY THIS IS THE FOUNDATION. Motive gives an unlabeled marker no id (`unlabeled_markers[*].id` is None), so
a per-frame assignment has to re-identify every marker every frame from geometry alone. Followed through
time, a marker carries its identity across the frames where its body shows only one or two markers -- the
frames a per-frame placement (>= 3 markers) wastes. Measured on chopstick_14 (120.8 Hz): the median marker
moves 0.29 mm per frame against >= 8 mm between neighbours, and 97% of all observations fall in tracks of
>= 0.25 s. Where Motive's label changes INSIDE such a track the step there is 0.48 mm median: Motive
relabelled, the marker did not jump.

CONSERVATIVE BY CONSTRUCTION. A wrong link merges two markers' identities, which no later stage can see as
cleanly as a cut; a missing link only costs a segment boundary, which the scaffold re-joins through the
template. So a track is cut wherever the match is not clearly unique:

  gate     a constant-velocity prediction k frames ahead misses by at most  3 noise + a_max (k dt)^2,
           and a track seen once (no velocity yet) by  3 noise + v_max dt  -- physical, so 250 Hz and
           120 Hz takes mean the same thing;
  ratio    the matched distance must be <= `ratio` x the runner-up in its row AND its column, unless it is
           within `ratio_floor_mm`, closer than two distinct markers can be;
  gap      a track survives a dropout of at most `max_gap_s`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

G = 9.80665e3              # mm/s^2
BIG = 1e9


def link(obs, rate_hz: float, tc) -> np.ndarray:
    """(N,) segment id of every observation."""
    dt = 1.0 / rate_hz
    max_gap = max(0, int(round(float(tc.max_gap_s) * rate_hz)))
    noise3, a_max, v_max = 3 * float(tc.noise_mm), float(tc.a_max_g) * G, float(tc.v_max_m_s) * 1e3
    ratio, floor = float(tc.ratio), float(tc.ratio_floor_mm)
    seg = np.full(len(obs), -1)
    ids = np.zeros(0, int)
    last_p, vel, last_f, seen = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int)
    nxt = 0
    for i in range(len(obs.ptr) - 1):
        sl = obs.at(i)
        P = obs.p[sl]
        alive = (i - last_f) <= max_gap + 1
        ids, last_p, vel, last_f, seen = ids[alive], last_p[alive], vel[alive], last_f[alive], seen[alive]
        got = np.full(len(P), -1)
        if len(ids) and len(P):
            k = (i - last_f).astype(float)
            pred = last_p + vel * k[:, None]
            gate = np.where(seen > 1, noise3 + a_max * (k * dt) ** 2, noise3 + v_max * dt * k)
            D = np.linalg.norm(pred[:, None] - P[None], axis=2)
            C = np.where(D <= gate[:, None], D, BIG)
            r, c = linear_sum_assignment(C)
            ok = C[r, c] < BIG
            r, c = r[ok], c[ok]
            if len(r):
                row2 = np.partition(D, 1, axis=1)[:, 1] if D.shape[1] > 1 else np.full(len(D), np.inf)
                col2 = np.partition(D, 1, axis=0)[1] if D.shape[0] > 1 else np.full(D.shape[1], np.inf)
                d = D[r, c]
                clear = (d <= floor) | ((d <= ratio * row2[r]) & (d <= ratio * col2[c]))
                r, c = r[clear], c[clear]
                got[c] = r
        new = np.flatnonzero(got < 0)
        seg_now = np.empty(len(P), int)
        m = got >= 0
        seg_now[m] = ids[got[m]]
        seg_now[new] = nxt + np.arange(len(new))
        seg[sl] = seg_now
        # update matched tracks, then append the new ones
        tr = got[m]
        if len(tr):
            k = (i - last_f[tr]).astype(float)
            vel[tr] = (P[m] - last_p[tr]) / k[:, None]
            last_p[tr], last_f[tr], seen[tr] = P[m], i, seen[tr] + 1
        ids = np.r_[ids, seg_now[new]]
        last_p = np.r_[last_p, P[new]].reshape(-1, 3)
        vel = np.r_[vel, np.zeros((len(new), 3))].reshape(-1, 3)
        last_f = np.r_[last_f, np.full(len(new), i)]
        seen = np.r_[seen, np.ones(len(new), int)]
        nxt += len(new)
    return seg


def summary(seg: np.ndarray, rate_hz: float) -> dict:
    """How much of the take lives in long tracks -- the number that says whether 4D identity is available."""
    n = np.bincount(seg)
    n = n[n > 0]
    tot = n.sum()
    return {"segments": int(len(n)), "observations": int(tot),
            **{f"obs_in_tracks_ge_{s}s": round(float(n[n >= s * rate_hz].sum() / tot), 4) for s in (0.25, 1, 5)}}
