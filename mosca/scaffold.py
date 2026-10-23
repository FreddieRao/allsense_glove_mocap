#!/usr/bin/env python3
# VENDORED from TableMocap-toolkit-padab/kinematics-replay-bimanual/fitting/reassignment/scaffold.py
# (last change 2edada7, 2026-10-16). Keep in sync by hand; only the import path is changed.
"""Stages 2-3: give every track segment an identity -- (body, template row) -- that its whole life agrees with.

THE TEMPLATE IS THE SCAFFOLD. MoSca has to learn a deformation graph because it knows nothing about the
scene; every body here has a pre-defined marker template (`data/asset/*.motive`). So MoSca's
as-rigid-as-possible residual between learned neighbours becomes the EXACT rigid residual of a segment
against its row,

    e_s(t) = | R_b(t) x_row + t_b(t) - p_s(t) |,

and the unknowns are a label per SEGMENT (a few hundred per take) plus a pose per body per frame, instead
of a label per marker per frame (~500 000). A segment's identity therefore reaches the frames where its
body shows only one or two markers -- no pose there, but the identity is the segment's.

THE ORDER, and the MoSca idea each step borrows:

    groups      signed rigidity graph between moving segments (MoSca's curve-distance topology). A
                distance that CHANGES proves two bodies (cannot-link); a distance that holds while the
                pair TURNED is evidence for one. A distance that merely holds is not -- a held object and
                a resting one are rigid with the hand for a while, and plain linking percolates into one
                blob (probe: 151k observations, both chopsticks and a plate in one group).
    naming      each group matched to EVERY template in one round, voted over its best-covered frames; a
                cap has to sit near its median place in its OWN plate's frame, which is what keeps a right
                cap from being named as the left hand's mirror cap.
    seeding     a segment no group named takes Motive's majority label over its whole life -- weak per
                frame, strong per segment -- and is then held to the same residual test as any other.
    loop        pose per body per frame (Kabsch, outliers dropped; a thin body from two markers with its
                roll carried) -> accept each observation within gate_b -> one observation per (body, row,
                frame) -> cut a segment where it is rejected for `split.min_run_s` -> a posed body's EMPTY
                rows claim free segments (MoSca's completion of the unobserved) -> a free segment holding
                a template distance to a labelled one is named by that ruler (the chopsticks, which turn
                too little for the rotation test).

FAIL OPEN. Only an unlabelled STATIC segment becomes background; a free segment that nothing names stays
in the pool for the consumers, exactly as `fitting/reassignment` treats a marker it cannot place.
"""
from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from mosca.shape import carry_roll, consensus, fit_two, place, turn_deg

REACH_MM = 200.0           # tip_fusion's max_reach_mm: the fallback cap gate where no anchor exists
VERIFIED_SHARE = 0.8       # a segment whose posed frames pass at this rate carries its identity to the rest


@dataclass
class Pose:
    """A body's per-frame pose. `valid` means a POSE -- position and rotation both measured. `pos_only` is
    the weaker, separate claim: the position is measured, the rotation is not (see `position_only`)."""
    valid: np.ndarray                  # (T,)
    R: np.ndarray                      # (T,3,3)
    t: np.ndarray                      # (T,3) mm
    n: np.ndarray                      # (T,) markers in the fit
    rmse: np.ndarray                   # (T,)
    pos_only: np.ndarray = None        # (T,) position measured from 1-2 markers, rotation NOT measured
    pos_t: np.ndarray = None           # (T,3) mm, that position
    pos_sigma: np.ndarray = None       # (T,) its 1-sigma, LEVER-SCALED (position_only)
    pos_n: np.ndarray = None           # (T,) markers behind it (1 or 2)

    @classmethod
    def empty(cls, T: int) -> "Pose":
        return cls(np.zeros(T, bool), np.full((T, 3, 3), np.nan), np.full((T, 3), np.nan),
                   np.zeros(T, int), np.full(T, np.nan),
                   np.zeros(T, bool), np.full((T, 3), np.nan), np.full(T, np.nan), np.zeros(T, int))


@dataclass
class Scaffold:
    take: object
    seg: np.ndarray                    # (N,) segment of every observation
    seg_obs: list                      # per segment, its observations in frame order
    body: np.ndarray                   # (S,) body index, -1 unlabelled
    row: np.ndarray                    # (S,) compact template row, -1
    src: list                          # (S,) 'group' | 'motive' | 'recover' | 'partner' | ''
    static: np.ndarray                 # (S,) bool
    banned: dict = field(default_factory=lambda: defaultdict(set))   # seg -> {(b, row)} it was cut from
    acc: np.ndarray = None             # (N,) accepted
    resid: np.ndarray = None           # (N,) mm, NaN where the body had no pose
    poses: list = None                 # [Pose] per body
    sigma: np.ndarray = None           # (B,) per-body fit noise, mm
    emf: dict = field(default_factory=dict)   # {cap body: (R (T,3,3), t (T,3) mm, ok (T,))} -- emf.py
    groups_cache: list = None          # pass 1's rigidity graph, reused by pass 2
    report: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------------------- basics
    @property
    def n_seg(self) -> int:
        return len(self.seg_obs)

    def frames(self, s: int) -> np.ndarray:
        return self.take.obs.frame[self.seg_obs[s]]

    def duration_s(self, s: int) -> float:
        f = self.frames(s)
        return (f[-1] - f[0] + 1) / self.take.rate_hz if len(f) else 0.0

    def label(self, s: int, b: int, r: int, how: str) -> None:
        self.body[s], self.row[s], self.src[s] = b, r, how

    def unlabel(self, s: int, ban: bool = True) -> None:
        if ban and self.body[s] >= 0:
            self.banned[s].add((int(self.body[s]), int(self.row[s])))
        self.body[s], self.row[s], self.src[s] = -1, -1, ""

    def cut(self, s: int, take_out: np.ndarray) -> int:
        """Move observations `take_out` (a subset of segment s) into a new, unlabelled segment."""
        new = self.n_seg
        keep = ~np.isin(self.seg_obs[s], take_out)
        self.seg_obs[s] = self.seg_obs[s][keep]
        self.seg_obs.append(np.sort(take_out))
        self.seg[take_out] = new
        self.body, self.row = np.r_[self.body, -1], np.r_[self.row, -1]
        self.static = np.r_[self.static, False]
        self.src.append("")
        self.banned[new] = set(self.banned[s]) | ({(int(self.body[s]), int(self.row[s]))} if self.body[s] >= 0 else set())
        return new

    def emf_pose(self, body: str, f: int):
        """(R, t) the coils imply for this cap at this frame, or None. An INDEPENDENT sensor: unlike Motive's
        label it can contradict a wrong solve instead of agreeing with it."""
        got = self.emf.get(body)
        if got is None or not got[2][f]:
            return None
        return got[0][f], got[1][f]

    def emf_body(self, f: int, p_mm: np.ndarray, gate_mm: float) -> int:
        """The cap body whose coil puts it nearest `p_mm` at frame f, within `gate_mm`, or -1."""
        best, bk = gate_mm, -1
        for k, b in enumerate(self.take.bodies):
            got = self.emf.get(b.name)
            if got is None or not got[2][f]:
                continue
            d = float(np.linalg.norm(got[1][f] - p_mm))
            if d < best:
                best, bk = d, k
        return bk

    def gate(self, b: int, pc) -> float:
        return max(float(pc.gate_floor_mm), float(pc.gate_k) * float(self.sigma[b]))

    def plate_pose(self, side: str, f: int):
        """(R, t) of `side`'s dorsal plate at frame f: this scaffold's own where it solved it, else Motive's."""
        k = next((j for j, b in enumerate(self.take.bodies) if b.kind == "plate" and b.side == side), None)
        if k is not None and self.poses is not None and self.poses[k].valid[f]:
            return self.poses[k].R[f], self.poses[k].t[f]
        pm = self.take.plate_motive.get(side)
        if pm is not None and pm.valid[f]:
            return pm.R_world_from_body[f], pm.t_world_from_body_mm[f]
        return None


def build(take, seg: np.ndarray, sc_cfg) -> Scaffold:
    order = np.argsort(seg, kind="stable")
    counts = np.bincount(seg)
    seg_obs = np.split(order, np.cumsum(counts)[:-1])
    S = len(seg_obs)
    sc = Scaffold(take, seg.copy(), seg_obs, np.full(S, -1), np.full(S, -1), [""] * S, np.zeros(S, bool))
    sc.sigma = np.full(len(take.bodies), 0.5)
    sc.poses = None
    _classify_static(sc, sc_cfg.static)
    return sc


def _classify_static(sc: Scaffold, st) -> None:
    obs = sc.take.obs
    for s in range(sc.n_seg):
        if sc.duration_s(s) < float(st.min_s):
            continue
        p = obs.p[sc.seg_obs[s]]
        sc.static[s] = np.percentile(np.linalg.norm(p - np.median(p, 0), axis=1), 95) <= float(st.drift_mm)


# ----------------------------------------------------------------------------------------- groups
def _subgrid(sc: Scaffold, segs: list, sample_hz: float, only_accepted: bool = False):
    """(X (T_sub, n, 3) float32 NaN where absent, step) -- the segments on a coarse time grid."""
    take = sc.take
    step = max(1, int(round(take.rate_hz / sample_hz)))
    X = np.full(((take.n_frames + step - 1) // step, len(segs), 3), np.nan, np.float32)
    for j, s in enumerate(segs):
        idx = sc.seg_obs[s]
        if only_accepted:
            idx = idx[sc.acc[idx]]
        f = take.obs.frame[idx]
        on = f % step == 0
        X[f[on] // step, j] = take.obs.p[idx[on]]
    return X, step


def _span(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vis = np.isfinite(X[..., 0])
    has = vis.any(0)
    first = np.where(has, vis.argmax(0), 0)
    last = np.where(has, len(vis) - 1 - vis[::-1].argmax(0), -1)
    return first, last


def _pair_stats(X: np.ndarray, a: int, bs: np.ndarray, lo: int, hi: int):
    """(n co-visible, median, band p1-p99, turned p95 deg) between segment a and each of bs over [lo, hi]."""
    V = X[lo:hi + 1, bs] - X[lo:hi + 1, a:a + 1]
    d = np.linalg.norm(V, axis=2)
    m = np.isfinite(d).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns: pairs never co-visible
        med = np.nanmedian(d, 0)
        band = np.nanpercentile(d, 99, 0) - np.nanpercentile(d, 1, 0)
        u = V / d[..., None]
        mu = np.nanmean(u, 0)
        mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        turned = np.nanpercentile(np.degrees(np.arccos(np.clip(np.einsum("tnk,nk->tn", u, mu), -1, 1))), 95, 0)
    return m, med, band, turned


def groups(sc: Scaffold, rc) -> list[np.ndarray]:
    """Rigid groups of moving segments: constrained merge by tested rigidity, never across a cannot-link."""
    take = sc.take
    segs = [s for s in range(sc.n_seg) if not sc.static[s] and sc.duration_s(s) >= float(rc.min_segment_s)]
    if len(segs) < 2:
        return []
    X, step = _subgrid(sc, segs, float(rc.sample_hz))
    hz = take.rate_hz / step
    must_n, cannot_n = max(2, int(float(rc.must_overlap_s) * hz)), max(2, int(round(float(rc.cannot_overlap_s) * hz)))
    first, last = _span(X)
    order = np.argsort(first)
    must, cannot = [], defaultdict(set)
    for ai, a in enumerate(order):
        bs = order[ai + 1:]
        bs = bs[(first[bs] <= last[a]) & (last[bs] >= first[a])]
        if not len(bs):
            continue
        m, med, band, turned = _pair_stats(X, a, bs, first[a], last[a])
        for b in bs[(m >= cannot_n) & (band > float(rc.cannot_band_mm))]:
            cannot[a].add(int(b))
            cannot[int(b)].add(int(a))
        ok = (m >= must_n) & (band < float(rc.must_band_mm)) & (med < float(rc.max_pair_mm)) \
            & (turned >= float(rc.min_rotation_deg))
        must += [(float(m[j] * turned[j]), int(a), int(bs[j])) for j in np.flatnonzero(ok)]
    parent = list(range(len(segs)))
    members = {i: {i} for i in range(len(segs))}
    links = defaultdict(set)
    for _w, a, b in must:
        links[a].add(b)
        links[b].add(a)

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    blocked = 0
    support = int(rc.support)
    for _w, a, b in sorted(must, reverse=True):
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        if any(cannot[x] & members[rb] for x in members[ra]):
            blocked += 1
            continue
        small, big = sorted((members[ra], members[rb]), key=len)
        if len(big) >= support and any(len(links[x] & big) < support for x in small):
            continue
        parent[rb] = ra
        members[ra] |= members.pop(rb)
    out = [np.array([segs[j] for j in ms]) for ms in members.values() if len(ms) >= 2]
    sc.report["groups"] = {"segments": len(segs), "must_links": len(must),
                           "cannot_links": sum(map(len, cannot.values())) // 2, "merges_blocked": blocked,
                           "groups_ge2": len(out)}
    return out


# ----------------------------------------------------------------------------------------- naming
def _frames_to_vote(sc: Scaffold, members: np.ndarray, nc) -> list[int]:
    f = np.concatenate([sc.frames(s) for s in members])
    cnt = np.bincount(f, minlength=sc.take.n_frames)
    sep = sc.take.frames_of(float(nc.min_separation_s))
    out = []
    for i in np.argsort(-cnt, kind="stable"):
        if cnt[i] < int(nc.min_markers) or len(out) >= int(nc.frames):
            break
        if all(abs(int(i) - j) >= sep for j in out):
            out.append(int(i))
    return out


def _cap_ok(sc: Scaffold, body, R: np.ndarray, t: np.ndarray, f: int, nc) -> bool:
    """Is this placement physically available to this cap? The coil's own prediction where it has one --
    a few mm, body-specific -- else the plate-frame anchor, which only says "somewhere on this hand"."""
    e = sc.emf_pose(body.name, f)
    if e is not None:
        return float(np.linalg.norm(t - e[1])) <= float(nc.emf_mm)
    pp = sc.plate_pose(body.side, f)
    if pp is None:
        return False
    a = sc.take.cap_anchor.get(body.name)
    if a is None:
        return float(np.linalg.norm(t - pp[1])) <= REACH_MM
    return float(np.linalg.norm(pp[0].T @ (t - pp[1]) - a)) <= float(nc.anchor_mm)


def _motive_body(sc: Scaffold, members: np.ndarray) -> int:
    """The body Motive labels most of the group's segments for (majority per segment, then over segments),
    or -1 -- a TIE-BREAKER for naming, never a vote on its own."""
    take, obs = sc.take, sc.take.obs
    b_of_name = {b.name: k for k, b in enumerate(take.bodies)}
    per = []
    for s in members:
        own = obs.owner[sc.seg_obs[s]]
        own = own[own >= 0]
        if len(own):
            per.append(b_of_name.get(take.names[Counter(own.tolist()).most_common(1)[0][0]], -1))
    if not per:
        return -1
    (k, c), = Counter(per).most_common(1)
    return k if c >= 0.8 * len(members) else -1


def name_groups(sc: Scaffold, grps: list, nc) -> dict:
    """Name each still-unlabelled group by template votes over EVERY template; returns groups named.

    ONE ROUND OVER ALL KINDS, because offering a group only the plates first names a cap as a plate: on
    book_13 the right pinky's own six-marker group took the right DORSAL template, which fits four of its
    markers within tolerance, and the pinky lost its markers for the whole take. Against every template the
    pinky explains 6 of 6 and the plate 4. A placement must also explain `min_explain` of the group's markers
    at that frame -- a single rigid body is explained whole. Where Motive's segment majority names a body
    that explains within one marker of the winner, it wins (`label_prior`). A cap is gated on its plate's
    pose, Motive's until this scaffold has solved its own.
    """
    obs, bodies = sc.take.obs, sc.take.bodies
    named, votes_geom, votes_prior, votes_emf, groups_prior = 0, 0, 0, 0, 0
    for g in sorted(grps, key=lambda g: -sum(len(sc.seg_obs[s]) for s in g)):
        members = np.array([s for s in g if sc.body[s] < 0])
        if len(members) < int(nc.min_markers):
            continue
        prior = _motive_body(sc, members) if bool(nc.label_prior) else -1
        votes, rows = Counter(), defaultdict(Counter)
        for f in _frames_to_vote(sc, members, nc):
            idx = np.concatenate([sc.seg_obs[s][sc.frames(s) == f] for s in members])[:25]
            P = obs.p[idx]
            fits = {}
            for k, body in enumerate(bodies):
                pl = place(P, body, nc)
                if pl is None or pl[0] < int(nc.min_markers) or pl[0] < float(nc.min_explain) * len(P):
                    continue
                if body.kind == "cap" and not _cap_ok(sc, body, pl[2], pl[3], f, nc):
                    continue
                fits[k] = pl
            if not fits:
                continue
            k = max(fits, key=lambda j: (fits[j][0], -fits[j][1]))
            coil = sc.emf_body(f, P.mean(0), float(nc.emf_mm))
            overrode = False
            if coil in fits:                           # an independent sensor beats both geometry and Motive
                if coil != k:
                    votes_emf += 1
                k = coil
            elif prior in fits and prior != k and fits[prior][0] >= fits[k][0] - 1:
                k, overrode = prior, True              # Motive's majority, within one marker of geometry's best
            votes_prior += overrode
            votes_geom += not overrode
            votes[k] += 1
            for c, r in zip(fits[k][4], fits[k][5]):
                rows[int(sc.seg[idx[c]])][(k, int(r))] += 1
        if not votes:
            continue
        (k, c), = votes.most_common(1)
        if c < 3 or c / sum(votes.values()) < float(nc.body_share):
            continue
        named += 1
        groups_prior += prior == k and votes_prior > 0
        for s in members:
            rv = Counter({r: v for (kk, r), v in rows[int(s)].items() if kk == k})
            if rv and rv.most_common(1)[0][1] / sum(rows[int(s)].values()) >= float(nc.row_share):
                sc.label(int(s), k, rv.most_common(1)[0][0], "group")
    return {"named": named, "frame_votes_geometry": votes_geom, "frame_votes_motive_override": votes_prior,
            "frame_votes_emf_override": votes_emf, "groups_where_motive_overrode": groups_prior}


def _row_of(body) -> np.ndarray:
    out = np.full(int(body.known.max()) + 1, -1)
    out[body.known] = np.arange(len(body.known))
    return out


def _emf_contradicts(sc: Scaffold, idx: np.ndarray, body: str, row: int, veto_mm: float) -> bool:
    """Does the coil put this body's row somewhere else than this segment sits? The check Motive's own label
    cannot provide: a label IS Motive's solve, so a wrong solve and a wrong label agree. Measured case --
    chopstick_14's right dorsal plate holds three foreign markers for 11 s at a 100% label share, 92-127 mm
    off. Only applies to a cap with a usable coil; silent otherwise."""
    obs = sc.take.obs
    got = sc.emf.get(body)
    if got is None:
        return False
    k = sc.take.body_index(body)
    f = obs.frame[idx]
    on = got[2][f]
    if on.sum() < 5:
        return False
    i, fo = idx[on], f[on]
    pred = np.einsum("nij,j->ni", got[0][fo], sc.take.bodies[k].tk[row]) + got[1][fo]
    return float(np.median(np.linalg.norm(pred - obs.p[i], axis=1))) > veto_mm


def seed_from_motive(sc: Scaffold, nc) -> dict:
    """Unlabelled segments take Motive's majority (body, row) where it is a clear majority over their life
    AND the coils do not contradict it.

    A pre-pose template-consistency filter on these seeds was built and REJECTED (2026-10-06): it changed
    posed frames by zero on all three chopstick takes while costing markers per frame, because the post-pose
    cleanup below (consensus -> verify -> split) already removes a bad seed from any body that poses, and for
    a body that never poses, dropping wrong labels does not create right ones. Do not re-add it.
    """
    take, obs = sc.take, sc.take.obs
    b_of_name = {b.name: k for k, b in enumerate(take.bodies)}
    owner_b = np.array([b_of_name.get(n, -1) for n in take.names] + [-1])
    row_of = [_row_of(b) for b in take.bodies]
    min_n = take.frames_of(float(nc.label_min_s))
    seeded, disagree, vetoed = 0, 0, 0
    for s in range(sc.n_seg):
        idx = sc.seg_obs[s]
        ob = owner_b[obs.owner[idx]]
        lab = ob >= 0
        if lab.sum() < min_n:
            continue
        (key, c), = Counter(zip(ob[lab].tolist(), obs.row[idx][lab].tolist())).most_common(1)
        if c / lab.sum() < float(nc.label_share):
            continue
        b, ar = key
        r = int(row_of[b][ar]) if 0 <= ar < len(row_of[b]) else -1
        if r < 0 or (b, r) in sc.banned[s]:
            continue
        if _emf_contradicts(sc, idx, take.bodies[b].name, r, float(nc.emf_veto_mm)):
            vetoed += 1
            continue
        if sc.body[s] >= 0:
            disagree += int((sc.body[s], sc.row[s]) != (b, r))
            continue
        sc.label(s, b, r, "motive")
        seeded += 1
    return {"seeded": seeded, "group_vs_motive_disagree": disagree, "vetoed_by_emf": vetoed}


# --------------------------------------------------------------------------------------- the loop
def _labelled(sc: Scaffold, b: int) -> np.ndarray:
    segs = np.flatnonzero(sc.body == b)
    return np.concatenate([sc.seg_obs[s] for s in segs]) if len(segs) else np.zeros(0, int)


def _plausible(R, t, R_prev, t_prev, k: int, dt: float, pc) -> bool:
    """A pose k frames after the last accepted one, within the physical rate limits."""
    lim_deg = float(pc.max_rate_deg_s) * k * dt + float(pc.rate_slack_deg)
    lim_mm = float(pc.max_speed_m_s) * 1e3 * k * dt + float(pc.step_slack_mm)
    return turn_deg(R_prev, R) <= lim_deg and float(np.linalg.norm(t - t_prev)) <= lim_mm


def _long_axis(tk: np.ndarray) -> np.ndarray:
    return np.linalg.svd(tk - tk.mean(0), full_matrices=False)[2][0]


def _verified_segments(sc: Scaffold, k: int) -> set:
    """Segments of body k that passed the residual test somewhere -- the only evidence a SINGLE marker's
    identity has, since one marker cannot be checked against anything within its own frame."""
    out = set()
    if sc.acc is None:
        return out
    for s in np.flatnonzero(sc.body == k):
        idx = sc.seg_obs[int(s)]
        if np.isfinite(sc.resid[idx]).any() and sc.acc[idx].any():
            out.add(int(s))
    return out


def position_only(body, X: np.ndarray, Y: np.ndarray, R_ref: np.ndarray, sigma_b: float, pc) -> tuple | None:
    """(t_mm, sigma_mm) from `pos_min_markers` or more markers (but fewer than a pose needs), or None --
    position measured, rotation NOT.

    WHY THIS IS A SEPARATE THING AND NOT A POSE. A rigid fit needs three markers; a thin body's `fit_two`
    needs two AND a recent rotation to carry roll from. With fewer, the rotation is simply not in the data --
    but the POSITION is, given any rotation estimate: t = p_world - R x_row. `object_track` already carries
    the pose as position + axis + roll and smooths the three independently, so a measurement that speaks only
    to position is something it can consume natively; what it must NOT be given is a rotation invented here
    and presented as measured.

    THE SIGMA IS THE WHOLE POINT. The rotation reference is wrong by about `sigma_axis`, and that angular
    error lands at the marker's LEVER ARM from the body origin:

        sigma = sqrt( sigma_b^2 + (|x_row| sigma_axis)^2 )

    |x_row| spans 9-239 mm on the chopsticks, so this is 0.15-4.17 mm -- a 28x range that the single
    configured `object_track.sigma.markers.pos_mm` (0.362) would flatten, trusting a far-tip marker 11x more
    than it deserves. Two markers are averaged, so their lever is the mean of the two.
    """
    if len(X) < int(pc.pos_min_markers):
        return None
    lev = np.linalg.norm(X, axis=1)
    if len(X) == 2:
        # the one check a 2-marker frame can make on itself: the pair must be the template distance apart
        d = float(np.linalg.norm(Y[1] - Y[0]))
        if abs(d - float(np.linalg.norm(X[1] - X[0]))) > float(pc.pos_pair_tol_mm):
            return None
    t = Y.mean(0) - R_ref @ X.mean(0)
    sig = float(np.sqrt(sigma_b ** 2 + (lev.mean() * float(pc.pos_sigma_axis)) ** 2) / np.sqrt(len(X)))
    return t, sig


def nearest_true(valid: np.ndarray, reach: int) -> np.ndarray:
    """(T,) the nearest frame where `valid`, at most `reach` away in EITHER direction, -1 for none.
    `object_track.nearest_valid` is the same idea; this package depends only on data_io, so it is local."""
    k = np.flatnonzero(valid)
    out = np.full(len(valid), -1)
    if not len(k):
        return out
    i = np.arange(len(valid))
    hi = np.clip(np.searchsorted(k, i), 0, len(k) - 1)
    lo = np.clip(hi - 1, 0, len(k) - 1)
    j = np.where(np.abs(k[lo] - i) <= np.abs(k[hi] - i), k[lo], k[hi])
    return np.where(np.abs(j - i) <= reach, j, -1)


def _add_position_only(sc: Scaffold, k: int, ps: Pose, body, idx: np.ndarray, pc) -> None:
    """Fill `ps.pos_*` on the frames that have 1-2 markers and no pose.

    A SECOND PASS, not a branch in the causal loop, and that is the difference between 59 frames and all of
    them: the rotation reference does not have to be the PREVIOUS pose. These are long unposed runs -- 1115
    frames on chopstick_14/chopstick02 -- so a 0.1 s backward reach (`hold_s`, which the rate gate needs)
    reaches almost none of them. Once the whole take's poses are known, the nearest pose in EITHER direction
    within `pos_hold_s` is the better reference, and a batch pipeline has no reason to be causal.
    """
    if not len(idx):
        return
    obs = sc.take.obs
    ref = nearest_true(ps.valid, sc.take.frames_of(float(pc.pos_hold_s)))
    verified = _verified_segments(sc, k) if bool(pc.pos_require_verified) else None
    f = obs.frame[idx]
    order = np.argsort(f, kind="stable")
    idx, f = idx[order], f[order]
    starts = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
    for a, e in zip(starts, np.r_[starts[1:], len(idx)]):
        i = int(f[a])
        if ps.valid[i] or not (int(pc.pos_min_markers) <= e - a <= 2) or ref[i] < 0:
            continue
        if verified is not None and not all(int(x) in verified for x in sc.seg[idx[a:e]]):
            continue
        rows = sc.row[sc.seg[idx[a:e]]]
        got = position_only(body, body.tk[rows], obs.p[idx[a:e]], ps.R[ref[i]], float(sc.sigma[k]), pc)
        if got is not None:
            ps.pos_t[i], ps.pos_sigma[i] = got
            ps.pos_only[i], ps.pos_n[i] = True, int(e - a)


def solve_poses(sc: Scaffold, pc) -> None:
    """Each body's pose per frame from its labelled observations: the largest consistent set
    (`shape.consensus`), gated by the physical rate limits against the last accepted pose within `hold_s`.
    A thin body's roll is carried where the measured roll would turn faster than the limit."""
    take, obs = sc.take, sc.take.obs
    T, dt = take.n_frames, take.dt
    hold = take.frames_of(float(pc.hold_s))
    sc.poses = []
    rejected = {}
    for k, body in enumerate(take.bodies):
        ps = Pose.empty(T)
        g = sc.gate(k, pc)
        axis = _long_axis(body.tk) if body.thin else None
        idx = _labelled(sc, k)
        n_rej = 0
        if len(idx):
            idx = idx[np.argsort(obs.frame[idx], kind="stable")]
            f = obs.frame[idx]
            starts = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
            last = -10 ** 9
            for a, e in zip(starts, np.r_[starts[1:], len(idx)]):
                i = int(f[a])
                X, Y = body.tk[sc.row[sc.seg[idx[a:e]]]], obs.p[idx[a:e]]
                recent = i - last <= hold
                fit = None
                if len(X) >= 3:
                    R, t, rmse, keep = consensus(X, Y, g, float(pc.outlier_mm), ps.t[last] if recent else None)
                    if body.thin and recent and not _plausible(R, t, ps.R[last], ps.t[last], i - last, dt, pc):
                        R = carry_roll(R, ps.R[last], axis)
                        t = Y[keep].mean(0) - R @ X[keep].mean(0)
                        rmse = float(np.sqrt(((X[keep] @ R.T + t - Y[keep]) ** 2).sum(1).mean()))
                    if rmse <= float(pc.rmse_max_mm):
                        fit = (R, t, rmse, len(keep))
                elif len(X) == 2 and body.thin and recent:
                    two = fit_two(X, Y, ps.R[last], float(pc.gate_floor_mm))
                    if two is not None:
                        fit = (*two, 2)
                if fit is not None and recent and not _plausible(fit[0], fit[1], ps.R[last], ps.t[last], i - last, dt, pc):
                    fit = None
                    n_rej += 1
                if fit is not None:
                    ps.R[i], ps.t[i], ps.rmse[i], ps.n[i] = fit
                    ps.valid[i] = True
                    last = i
        good = ps.valid & (ps.n >= 4)
        if good.sum() >= 20:
            sc.sigma[k] = max(0.1, float(np.median(ps.rmse[good])))
        if n_rej:
            rejected[body.name] = n_rej
        if bool(pc.pos_only) and body.kind == "object":
            _add_position_only(sc, k, ps, body, idx, pc)
        sc.poses.append(ps)
    sc.report["implausible_poses"] = rejected
    sc.report["position_only"] = {b.name: {"frames": int(p.pos_only.sum()),
                                           "sigma_mm_p50": round(float(np.median(p.pos_sigma[p.pos_only])), 2),
                                           "from_1_marker": int((p.pos_n == 1).sum()),
                                           "from_2_markers": int((p.pos_n == 2).sum())}
                                  for b, p in zip(take.bodies, sc.poses) if p.pos_only.any()}


def verify(sc: Scaffold, pc) -> None:
    """Accept each labelled observation within its body's gate; one observation per (body, row, frame)."""
    take, obs = sc.take, sc.take.obs
    N = len(obs)
    sc.resid, sc.acc = np.full(N, np.nan), np.zeros(N, bool)
    for k, body in enumerate(take.bodies):
        idx = _labelled(sc, k)
        if not len(idx):
            continue
        f = obs.frame[idx]
        ps = sc.poses[k]
        on = ps.valid[f]
        i, fo = idx[on], f[on]
        pred = np.einsum("nij,nj->ni", ps.R[fo], body.tk[sc.row[sc.seg[i]]]) + ps.t[fo]
        sc.resid[i] = np.linalg.norm(pred - obs.p[i], axis=1)
        sc.acc[i] = sc.resid[i] <= sc.gate(k, pc)
    # identity carried to the frames with no pose, by segments that passed where they could be tested
    for s in np.flatnonzero(sc.body >= 0):
        idx = sc.seg_obs[s]
        posed = np.isfinite(sc.resid[idx])
        n_pose = int(posed.sum())
        carry = (n_pose > 0 and sc.acc[idx[posed]].mean() >= VERIFIED_SHARE) or \
            (n_pose == 0 and sc.src[s] in ("motive", "partner"))
        sc.acc[idx[~posed]] = carry
    _exclusive(sc)


def _exclusive(sc: Scaffold) -> None:
    """One accepted observation per (body, row, frame): the smallest residual, else the longest segment."""
    obs = sc.take.obs
    i = np.flatnonzero(sc.acc)
    b, r, f = sc.body[sc.seg[i]], sc.row[sc.seg[i]], obs.frame[i]
    key = (b.astype(np.int64) * 1000 + r) * (sc.take.n_frames + 1) + f
    length = np.array([len(sc.seg_obs[s]) for s in range(sc.n_seg)])[sc.seg[i]]
    order = np.lexsort((-length, np.nan_to_num(sc.resid[i], nan=1e9), key))
    dup = np.r_[False, key[order][1:] == key[order][:-1]]
    sc.acc[i[order][dup]] = False
    sc.report.setdefault("exclusive_dropped", []).append(int(dup.sum()))


def split(sc: Scaffold, sp) -> dict:
    """Cut segments where they are rejected for `min_run_s`; drop the label of a segment rejected mostly."""
    run_n = sc.take.frames_of(float(sp.min_run_s))
    cut, dropped = 0, 0
    for s in list(np.flatnonzero(sc.body >= 0)):
        idx = sc.seg_obs[s]
        posed = np.isfinite(sc.resid[idx])
        rej = posed & ~sc.acc[idx]
        if not rej.any():
            continue
        if rej.sum() > 0.5 * posed.sum():
            sc.unlabel(s, ban=False)
            dropped += 1
            continue
        p = np.flatnonzero(posed)
        bad = rej[p]
        edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
        for a, e in zip(edges[::2], edges[1::2]):
            if e - a >= run_n:
                sc.cut(s, idx[p[a]:p[e - 1] + 1])
                cut += 1
    return {"cut": cut, "unlabelled": dropped}


def recover(sc: Scaffold, pc, rc) -> int:
    """A posed body's empty rows claim free segments' observations nearby; a segment whose claims agree
    over its life takes that identity."""
    take, obs = sc.take, sc.take.obs
    free = sc.body[sc.seg] < 0
    votes = defaultdict(Counter)
    posed_while = defaultdict(Counter)
    for k, body in enumerate(take.bodies):
        ps = sc.poses[k]
        got = sc.emf.get(body.name) if bool(rc.use_emf) else None
        # the coils pose a cap on frames its own markers cannot (< 3 of them), which is exactly where a
        # per-frame placement gives up; `pose_at` prefers the marker fit where there is one
        fr = np.flatnonzero(ps.valid | (got[2] if got is not None else False))
        if not len(fr):
            continue
        def pose_at(f, _ps=ps, _got=got):
            return (_ps.R[f], _ps.t[f]) if _ps.valid[f] else (_got[0][f], _got[1][f])
        have = np.zeros((take.n_frames, len(body.tk)), bool)
        i_acc = _labelled(sc, k)
        i_acc = i_acc[sc.acc[i_acc]]
        have[obs.frame[i_acc], sc.row[sc.seg[i_acc]]] = True
        g = sc.gate(k, pc)
        for f in fr:
            sl = obs.at(int(f))
            cand = np.flatnonzero(free[sl]) + sl.start
            if not len(cand):
                continue
            for s in np.unique(sc.seg[cand]):
                posed_while[int(s)][k] += 1
            miss = np.flatnonzero(~have[f])
            if not len(miss):
                continue
            Rf, tf = pose_at(f)
            pred = body.tk[miss] @ Rf.T + tf
            D = np.linalg.norm(pred[:, None] - obs.p[cand][None], axis=2)
            r, c = linear_sum_assignment(np.where(D <= g, D, 1e6))
            for rr, cc in zip(r, c):
                if D[rr, cc] <= g:
                    votes[int(sc.seg[cand[cc]])][(k, int(miss[rr]))] += 1
    min_v = take.frames_of(float(rc.min_votes_s))
    n = 0
    for s, cnt in votes.items():
        (key, c), = cnt.most_common(1)
        if key in sc.banned[s] or c < min_v or c / sum(cnt.values()) < float(rc.min_share):
            continue
        if c / max(1, posed_while[s][key[0]]) < float(rc.consistency):
            continue
        sc.label(s, key[0], key[1], "recover")
        n += 1
    return n


def partner(sc: Scaffold, ptc, rc) -> int:
    """A free segment holding one of a labelled partner's template distances is named by that ruler."""
    take, bodies = sc.take, sc.take.bodies
    free = [s for s in range(sc.n_seg) if sc.body[s] < 0 and not sc.static[s]
            and sc.duration_s(s) >= float(ptc.min_overlap_s)]
    lab = [s for s in np.flatnonzero(sc.body >= 0) if sc.duration_s(int(s)) >= float(ptc.min_overlap_s)]
    if not free or not lab:
        return 0
    X, step = _subgrid(sc, free + [int(s) for s in lab], float(rc.sample_hz), only_accepted=False)
    nf = len(free)
    lab_cols = np.arange(nf, nf + len(lab))
    hz = take.rate_hz / step
    min_n = max(2, int(float(ptc.min_overlap_s) * hz))
    first, last = _span(X)
    eps, n = float(ptc.eps_mm), 0
    for j, s in enumerate(free):
        bs = lab_cols[(first[lab_cols] <= last[j]) & (last[lab_cols] >= first[j])]
        if not len(bs):
            continue
        V = X[first[j]:last[j] + 1, bs] - X[first[j]:last[j] + 1, j:j + 1]
        d = np.linalg.norm(V, axis=2)
        m = np.isfinite(d).sum(0)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med, std = np.nanmedian(d, 0), np.nanstd(d, 0)
        votes = Counter()
        for q in np.flatnonzero((m >= min_n) & (std <= float(ptc.pair_std_mm))):
            ls = lab[bs[q] - nf]
            k, rl = int(sc.body[ls]), int(sc.row[ls])
            if med[q] > bodies[k].diam_mm + eps:
                continue
            gap = np.abs(bodies[k].D[rl] - med[q])
            gap[rl] = np.inf
            near = np.flatnonzero(gap <= eps)
            if len(near) == 1 and np.partition(gap, 1)[1] > 2 * eps:
                votes[(k, int(near[0]))] += 1
        if not votes:
            continue
        (key, c), = votes.most_common(1)
        if c >= int(ptc.support) and c / sum(votes.values()) >= 0.7 and key not in sc.banned[s]:
            sc.label(s, key[0], key[1], "partner")
            n += 1
    return n


# ------------------------------------------------------------------------------------------ driver
def resolve(take, seg: np.ndarray, cfg, emf: dict | None = None, grps: list | None = None,
            groups_report: dict | None = None) -> Scaffold:
    """The whole scaffold for one take. `emf` is the coils' cap poses (pass 2); `grps` reuses pass 1's
    rigidity graph, which depends only on the tracks and so cannot change between passes."""
    sc = build(take, seg, cfg)
    sc.emf = emf or {}
    if grps is None:
        grps = groups(sc, cfg.rigidity)                    # sets sc.report["groups"]
    else:
        sc.report["groups"] = groups_report or {}          # pass 2 reuses the graph, so carry its report
    rep = {"naming": name_groups(sc, grps, cfg.naming)}
    if bool(cfg.naming.label_prior):
        rep["motive"] = seed_from_motive(sc, cfg.naming)
    rep["iterations"] = []
    for _it in range(int(cfg.iterations)):
        solve_poses(sc, cfg.pose)
        verify(sc, cfg.pose)
        it = split(sc, cfg.split)
        solve_poses(sc, cfg.pose)
        verify(sc, cfg.pose)
        it["recovered"] = recover(sc, cfg.pose, cfg.recover)
        it["partnered"] = partner(sc, cfg.partner, cfg.rigidity)
        rep["iterations"].append(it)
    solve_poses(sc, cfg.pose)
    verify(sc, cfg.pose)
    sc.report.update(rep)
    sc.groups_cache = grps
    return sc


def confidence(sc: Scaffold, cfg) -> np.ndarray:
    """(N,) per accepted observation, in [0, 1]: fit x segment length x segment agreement x source."""
    take = sc.take
    conf = np.zeros(len(take.obs))
    w = cfg.confidence.source.to_dict() if hasattr(cfg.confidence.source, "to_dict") else vars(cfg.confidence.source)
    for s in np.flatnonzero(sc.body >= 0):
        idx = sc.seg_obs[s]
        a = sc.acc[idx]
        if not a.any():
            continue
        g = sc.gate(int(sc.body[s]), cfg.pose)
        e = sc.resid[idx]
        posed = np.isfinite(e)
        ce = np.where(posed, np.clip(1 - (np.nan_to_num(e) / g) ** 2, 0, 1), 0.5)
        share = sc.acc[idx[posed]].mean() if posed.any() else 0.5
        cl = min(1.0, sc.duration_s(int(s)) / float(cfg.confidence.full_s))
        conf[idx[a]] = (ce * share * cl * float(w.get(sc.src[s], 0.5)))[a]
    return conf
