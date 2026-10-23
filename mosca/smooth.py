#!/usr/bin/env python3
"""Temporal smoothing: of the RAW MARKER TRACKS (`smooth.target: markers`, the default -- see
`smooth_marker_tracks`) or of a re-solved rigid-body pose sequence (`target: pose`), by (1) zero-phase
Butterworth or (2) Kalman / RTS.

WHY. After MoSca the caps are flip-free, but each pose is still a per-frame Kabsch on 5 markers ~10 mm apart,
so ~0.1 mm marker noise becomes ~0.4 deg rotation jitter on its median frame (scripts/rb_jitter.py:
free_motion knuckle 0.24 deg / tip 0.42 deg about a 50 ms moving average). Hand motion lives well below
~10 Hz at 240 Hz sampling, so the noise can be filtered offline without lag.

REPRESENTATION. Translation (mm) is filtered directly. Rotation is filtered as the unit quaternion with its sign
made continuous along time (q and -q are the same rotation), then renormalised -- small-noise, wrap-free.

    butter   4th-order Butterworth low-pass, run forward and backward (`filtfilt`: zero phase, no lag),
             cutoff `butter.cutoff_hz`. Applied per contiguous valid run; a run too short to pad is left as is.
    rts      a constant-velocity Kalman filter + Rauch-Tung-Striebel smoother per channel (3 translation +
             4 quaternion), measurement noise PER FRAME from the marker fit: sigma_t = rmse / sqrt(n) and
             sigma_q = sigma_t / (2 * lever) (lever = RMS distance of the template rows from their centroid;
             a quaternion component moves half the angle), floored at `rts.floor_mm`. Process noise is set from
             `rts.bandwidth_hz`: q = sigma_ref^2 (2 pi f)^4, the white-acceleration intensity whose steady-state
             smoother has that bandwidth for the median measurement noise -- comparable to the Butterworth
             cutoff, but frames fitted from 3 noisy markers are trusted less than 5-marker frames, and gaps
             up to `rts.max_gap_s` are predicted through instead of cutting the run.

`valid` is never extended: a frame without a measured pose stays invalid (except bridged gaps, which `rts`
marks in `bridged` and does not count as valid).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    e = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    return list(zip(e[::2], e[1::2]))


def _continuous_quat(R: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T,4) xyzw, sign-continuous over the valid frames, NaN elsewhere."""
    q = np.full((len(R), 4), np.nan)
    i = np.flatnonzero(valid)
    q[i] = Rotation.from_matrix(R[i]).as_quat()
    prev = None
    for k in i:
        if prev is not None and q[k] @ prev < 0:
            q[k] = -q[k]
        prev = q[k]
    return q


def _from_quat(q: np.ndarray, valid: np.ndarray, R_in: np.ndarray) -> np.ndarray:
    R = R_in.copy()
    i = np.flatnonzero(valid & np.isfinite(q[:, 0]))
    qq = q[i] / np.linalg.norm(q[i], axis=1, keepdims=True)
    R[i] = Rotation.from_quat(qq).as_matrix()
    return R


def smooth_butter(R: np.ndarray, t_mm: np.ndarray, valid: np.ndarray, rate_hz: float, cutoff_hz: float,
                  order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    b, a = butter(order, cutoff_hz / (0.5 * rate_hz))
    pad = 3 * max(len(a), len(b))
    q = _continuous_quat(R, valid)
    t_out, q_out = t_mm.copy(), q.copy()
    for s, e in _runs(valid):
        if e - s <= pad:
            continue
        t_out[s:e] = filtfilt(b, a, t_mm[s:e], axis=0)
        q_out[s:e] = filtfilt(b, a, q[s:e], axis=0)
    return _from_quat(q_out, valid, R), t_out


def _rts_channels(z: np.ndarray, r: np.ndarray, dt: float, qa: np.ndarray) -> np.ndarray:
    """Constant-velocity KF + RTS, independent channels. z (T,C) NaN = no measurement (per channel), r (T,C) variance,
    qa (C,) white-acceleration intensity. Returns the smoothed position (T,C)."""
    T, C = z.shape
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Qb = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    x = np.zeros((T, C, 2))
    P = np.zeros((T, C, 2, 2))
    xp = np.zeros((T, C, 2))
    Pp = np.zeros((T, C, 2, 2))
    fin = np.isfinite(z)
    first = np.where(fin.any(0), fin.argmax(0), 0)
    xc = np.stack([np.nan_to_num(z[first, np.arange(C)]), np.zeros(C)], 1)
    Pc = np.tile(np.diag([1.0, 1.0]), (C, 1, 1)) * np.maximum(np.nan_to_num(r[first, np.arange(C)], nan=1.0),
                                                            1e-12)[:, None, None] * 1e3
    for k in range(T):
        if k > 0:
            xc = xc @ F.T
            Pc = F @ Pc @ F.T + qa[:, None, None] * Qb
        xp[k], Pp[k] = xc, Pc
        m = fin[k]                                           # per channel: a marker can be missing alone
        if m.any():
            S = Pc[m, 0, 0] + r[k, m]
            K = Pc[m, :, 0] / S[:, None]
            xc[m] = xc[m] + K * (z[k, m] - xc[m, 0])[:, None]
            Pc[m] = Pc[m] - K[:, :, None] * Pc[m, 0, None, :]
        x[k], P[k] = xc, Pc
    xs = x.copy()
    for k in range(T - 2, -1, -1):
        G = P[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = x[k] + np.einsum("cij,cj->ci", G, xs[k + 1] - xp[k + 1])
    return xs[:, :, 0]


def smooth_rts(R: np.ndarray, t_mm: np.ndarray, valid: np.ndarray, rmse_mm: np.ndarray, n: np.ndarray,
               template_mm: np.ndarray, rate_hz: float, bandwidth_hz: float, floor_mm: float,
               max_gap_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, t_mm, bridged) smoothed. Runs are joined across gaps up to `max_gap_s` (predicted through)."""
    dt = 1.0 / rate_hz
    lever = float(np.sqrt(((template_mm - template_mm.mean(0)) ** 2).sum(1).mean()))
    sig_t = np.maximum(np.nan_to_num(rmse_mm, nan=1e3) / np.sqrt(np.maximum(n, 1)), floor_mm)
    sig_q = sig_t / (2.0 * lever)
    q = _continuous_quat(R, valid)
    z = np.c_[np.where(valid[:, None], t_mm, np.nan), q]
    rv = np.c_[np.repeat(sig_t[:, None] ** 2, 3, 1), np.repeat(sig_q[:, None] ** 2, 4, 1)]
    w = 2.0 * np.pi * bandwidth_hz
    ref = np.median(np.c_[np.repeat(sig_t[valid, None], 3, 1), np.repeat(sig_q[valid, None], 4, 1)], 0)
    qa = ref ** 2 * w ** 4 * dt                                  # discrete-time scaling of the intensity
    # join runs whose gap is short enough to predict through
    gap = int(round(max_gap_s * rate_hz))
    runs = _runs(valid)
    blocks, cur = [], list(runs[0]) if runs else None
    for s, e in runs[1:]:
        if s - cur[1] <= gap:
            cur[1] = e
        else:
            blocks.append(tuple(cur))
            cur = [s, e]
    if cur:
        blocks.append(tuple(cur))
    t_out, q_out, bridged = t_mm.copy(), q.copy(), np.zeros(len(R), bool)
    for s, e in blocks:
        if e - s < 4:
            continue
        xs = _rts_channels(z[s:e], rv[s:e], dt, qa)
        t_out[s:e], q_out[s:e] = xs[:, :3], xs[:, 3:]
        bridged[s:e] = ~valid[s:e]
    return _from_quat(q_out, valid, R), t_out, bridged


def smooth_body(method: str, R: np.ndarray, t_mm: np.ndarray, valid: np.ndarray, rmse_mm: np.ndarray, n: np.ndarray,
                template_mm: np.ndarray, rate_hz: float, sc) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch on `smooth.method` (`none` | `butter` | `rts`). Returns (R, t_mm)."""
    if method == "none":
        return R, t_mm
    if method == "butter":
        return smooth_butter(R, t_mm, valid, rate_hz, float(sc.butter.cutoff_hz), int(sc.butter.order))
    if method == "rts":
        R2, t2, _ = smooth_rts(R, t_mm, valid, rmse_mm, n, template_mm, rate_hz, float(sc.rts.bandwidth_hz),
                               float(sc.rts.floor_mm), float(sc.rts.max_gap_s))
        return R2, t2
    raise ValueError(f"smooth.method {method!r}")


# ------------------------------------------------------------------------------------- raw-marker smoothing
def smooth_marker_tracks(P_mm: np.ndarray, rate_hz: float, method: str, sc) -> np.ndarray:
    """(T,K,3) mm -> (T,K,3) mm: each marker TRACK smoothed on its own, returned only where it was observed.

    The marker, not the pose, is what is measured: the cap poses are per-frame Kabsch fits of these markers and
    the floating markers have no pose at all, so smoothing the tracks reaches every input of the MANO fit
    (tip-cap markers, floating markers, and through the re-solve the knuckle root). Identity comes from
    MoSca (body, row) / the floating labels, so a track is one physical marker."""
    T, K, _ = P_mm.shape
    out = np.full_like(P_mm, np.nan)
    seen = np.isfinite(P_mm[..., 0])
    if method == "butter":
        b, a = butter(int(sc.butter.order), float(sc.butter.cutoff_hz) / (0.5 * rate_hz))
        pad = 3 * max(len(a), len(b))
        for k in range(K):
            for s_, e in _runs(seen[:, k]):
                out[s_:e, k] = filtfilt(b, a, P_mm[s_:e, k], axis=0) if e - s_ > pad else P_mm[s_:e, k]
        return out
    if method == "rts":
        dt = 1.0 / rate_hz
        sig = float(sc.rts.marker_noise_mm)
        z = P_mm.reshape(T, K * 3)
        r = np.full_like(z, sig ** 2)
        qa = np.full(K * 3, sig ** 2 * (2.0 * np.pi * float(sc.rts.bandwidth_hz)) ** 4 * dt)
        xs = _rts_channels(z, r, dt, qa).reshape(T, K, 3)
        out[seen] = xs[seen]
        return out
    raise ValueError(f"smooth.method {method!r}")


def resolve_from_markers(template_mm: np.ndarray, M_mm: np.ndarray, valid: np.ndarray, min_rows: int = 3):
    """(R (T,3,3), t (T,3) mm, valid (T,)) per-frame Kabsch of the template rows onto the (smoothed) markers.
    `valid` can only shrink: a frame MoSca did not pose is not posed here either."""
    from mosca.shape import kabsch
    T = len(M_mm)
    R = np.full((T, 3, 3), np.nan)
    t = np.full((T, 3), np.nan)
    ok = np.zeros(T, bool)
    seen = np.isfinite(M_mm[..., 0])
    for i in np.flatnonzero(valid & (seen.sum(1) >= min_rows)):
        rows = np.flatnonzero(seen[i])
        R[i], t[i], _ = kabsch(template_mm[rows], M_mm[i, rows])
        ok[i] = True
    return R, t, ok
