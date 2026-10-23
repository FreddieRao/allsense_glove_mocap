#!/usr/bin/env python3
# VENDORED from TableMocap-toolkit-padab/kinematics-replay-bimanual/fitting/reassignment/shape.py
# (last change 2edada7, 2026-10-16). Keep in sync by hand; only the import path is changed.
"""The geometry the scaffold needs: Kabsch, a fit that drops outliers, a 2-marker thin fit, and the
prediction-free template search that names a group.

THE SEARCH RUNS ON A GROUP, NOT ON A FRAME. `fitting/reassignment/hypotheses.from_shape` must search ~92
candidates per frame (125 000 triangles on a chopstick take); here the candidates are the markers of ONE
rigid group at one frame -- a few to ~20 -- so every candidate triangle is matched against every template
triangle by sorted side lengths with plain broadcasting, then verified by Kabsch and a nearest-row claim.
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
from scipy.optimize import linear_sum_assignment

PERMS = np.array(list(permutations(range(3))))


def kabsch(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(R, t, rmse) minimising |R X + t - Y|, reflection-guarded."""
    mx, my = X.mean(0), Y.mean(0)
    U, _s, Vt = np.linalg.svd((X - mx).T @ (Y - my))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = my - R @ mx
    return R, t, float(np.sqrt(((X @ R.T + t - Y) ** 2).sum(1).mean()))


def kabsch_batch(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R (n,3,3), t (n,3), residual (n,k)) for n paired sets X, Y of shape (n,k,3)."""
    mx, my = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    U, _s, Vt = np.linalg.svd(np.einsum("nki,nkj->nij", X - mx, Y - my))
    d = np.sign(np.linalg.det(np.einsum("nji,nkj->nik", Vt, U)))
    D = np.tile(np.eye(3), (len(X), 1, 1))
    D[:, 2, 2] = d
    R = np.einsum("nji,njk,nlk->nil", Vt, D, U)
    t = my[:, 0] - np.einsum("nij,nj->ni", R, mx[:, 0])
    res = np.linalg.norm(np.einsum("nij,nkj->nki", R, X) + t[:, None] - Y, axis=2)
    return R, t, res


def fit_dropping(X: np.ndarray, Y: np.ndarray, outlier_mm: float) -> tuple:
    """(R, t, rmse, keep) -- Kabsch, then drop the worst pair while it exceeds `outlier_mm` and > 3 remain."""
    keep = np.arange(len(X))
    while True:
        R, t, rmse = kabsch(X[keep], Y[keep])
        r = np.linalg.norm(X[keep] @ R.T + t - Y[keep], axis=1)
        if r.max() <= outlier_mm or len(keep) <= 3:
            return R, t, rmse, keep
        keep = np.delete(keep, int(np.argmax(r)))


def fit_two(X: np.ndarray, Y: np.ndarray, R_prev: np.ndarray, tol_mm: float) -> tuple | None:
    """(R, t, rmse) of a THIN body from two markers: the minimal rotation taking the previous pose's axis
    onto the observed one, so roll -- which two markers cannot measure -- is carried, not invented."""
    vb, vw = R_prev @ (X[1] - X[0]), Y[1] - Y[0]
    if abs(np.linalg.norm(vb) - np.linalg.norm(vw)) > tol_mm:
        return None
    a, b = vb / np.linalg.norm(vb), vw / np.linalg.norm(vw)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999:
        return None
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = (np.eye(3) + K + K @ K / (1 + c)) @ R_prev
    t = Y.mean(0) - R @ X.mean(0)
    return R, t, float(np.sqrt(((X @ R.T + t - Y) ** 2).sum(1).mean()))


def _tri_sides(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    return np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)], 1)


def place(P: np.ndarray, body, nc) -> tuple | None:
    """The best placement of `body`'s template on the points P (k,3): (n, rmse, R, t, ci, ti) with ci
    indexing P and ti the compact template rows, or None. Prediction-free."""
    if len(P) < 3 or len(body.tk) < 3:
        return None
    tol = float(nc.triplet_tol_mm)
    ctri = np.array(list(combinations(range(len(P)), 3)))
    ttri = np.array(list(combinations(range(len(body.tk)), 3)))
    cs, ts = _tri_sides(P, ctri), _tri_sides(body.tk, ttri)
    # every vertex order of each template triangle: side k is opposite vertex k, so permuting the vertices
    # permutes the sides the same way
    tt = ttri[:, PERMS].reshape(-1, 3)
    tss = ts[:, PERMS].reshape(-1, 3)
    hit = np.argwhere((np.abs(cs[:, None] - tss[None]) <= tol).all(2))
    if not len(hit):
        return None
    ci3, ti3 = ctri[hit[:, 0]], tt[hit[:, 1]]
    R, t, res = kabsch_batch(body.tk[ti3], P[ci3])
    good = np.flatnonzero(np.sqrt((res ** 2).mean(1)) <= float(nc.rmse_max_mm))
    best = None
    for g in good[:400]:
        pred = body.at(R[g], t[g])
        D = np.linalg.norm(pred[:, None] - P[None], axis=2)
        r, c = linear_sum_assignment(np.where(D <= float(nc.claim_mm), D, 1e6))
        ok = D[r, c] <= float(nc.claim_mm)
        ti, ci = r[ok], c[ok]
        if len(ti) < 3:
            continue
        Rf, tf, rmse = kabsch(body.tk[ti], P[ci])
        if rmse > float(nc.rmse_max_mm):
            continue
        key = (len(ti), -rmse)
        if best is None or key > best[0]:
            best = (key, (len(ti), rmse, Rf, tf, ci, ti))
    return best[1] if best else None


def consensus(X: np.ndarray, Y: np.ndarray, gate_mm: float, outlier_mm: float, t_prev: np.ndarray | None,
              max_triples: int = 200) -> tuple:
    """(R, t, rmse, keep) -- the LARGEST set of pairs one rigid pose explains within `gate_mm`.

    WHY NOT `fit_dropping` ALONE. Dropping the worst pair is a greedy descent that a coherent wrong set
    defeats: Motive's right dorsal RB locked onto three foreign markers for 11 s of chopstick_14 (rows 1,
    3, 4, 92-127 mm off), and with those three beside the plate's true four the fit flipped between the two
    sets. Here every 3-pair subset is a hypothesis, the one with most inliers wins, and a tie goes to the
    pose nearest the previous one -- the plate does not teleport.
    """
    R, t, rmse, keep = fit_dropping(X, Y, outlier_mm)
    if len(X) <= 3 or np.linalg.norm(X @ R.T + t - Y, axis=1).max() <= gate_mm:
        return R, t, rmse, keep
    tri = np.array(list(combinations(range(len(X)), 3)))
    if len(tri) > max_triples:
        tri = tri[np.linspace(0, len(tri) - 1, max_triples).astype(int)]
    Rb, tb, _res = kabsch_batch(X[tri], Y[tri])
    e = np.linalg.norm(np.einsum("nij,kj->nki", Rb, X) + tb[:, None] - Y[None], axis=2)
    inl = e <= gate_mm
    count = inl.sum(1)
    step = np.linalg.norm(tb - t_prev, axis=1) if t_prev is not None else np.zeros(len(tb))
    best = int(np.lexsort((step, -count))[0])
    keep = np.flatnonzero(inl[best])
    if len(keep) < 3:
        return R, t, rmse, np.arange(len(X))
    R, t, rmse = kabsch(X[keep], Y[keep])
    return R, t, rmse, keep


def turn_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def carry_roll(R: np.ndarray, R_prev: np.ndarray, axis_body: np.ndarray) -> np.ndarray:
    """R with its roll about the body's long axis replaced by the previous pose's: the minimal rotation
    taking R_prev's axis onto R's, applied to R_prev. The axis is measured; the roll was not."""
    a, b = R_prev @ axis_body, R @ axis_body
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999:
        return R
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + K + K @ K / (1 + c)) @ R_prev
