#!/usr/bin/env python3
"""MANO fit error of the calibrated index finger: cap markers and floating markers, RMS in mm, per take and split.

WHAT IS MEASURED, per marker observation, against the Stage E track (frozen calibration, per-frame pose):

    tip cap       |predicted tip-cap marker - observed|, the 5 rows, through MANO's distal bone and the binding
    knuckle cap   |T_world_from_knucklecap x_row - observed (raw)| with the cap pose the TRACKED ROOT implies
                  (T_world_from_canonical @ inv(T_knucklecap_from_canonical)): the cap's own rigid fit on the default
                  run -- the floor under every other number -- and what wrist smoothing costs on `smooth.wrist` runs
    float_*       |predicted floating-marker centre - observed|, the marker centre on the skin point + height
    float_*_win   the same, against the Stage B distance-window labels (before the MANO re-claim): what the
                  re-claim removed, i.e. how many window labels a MANO prediction cannot explain within 8 mm

SPLITS for free_motion: `fit` (the K frames Stage D fitted), `heldout` (the `fit.holdout_blocks` time blocks --
never seen by Stage D), `rest` (everything else). calibration is reported whole (its 200 fit frames are a
uniform sample of a static hand).

Also: the floating-marker height above the skin (the marker is ~3 mm radius, so the centre should sit >= 3 mm
up) and the floating residual against PIP flexion, which is where skin slides over the joint.

    ~/miniforge3/envs/far_grasp/bin/python -m evaluation.fit_error
Writes notes/2026-10-22_calibration_eval[_<run.tag>].json and prints the markdown tables used in the pipeline note.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.common import ROOT, calib_sfx, load_rig, sfx, write_json                     # noqa: E402
from calibration.fit_hand import holdout_mask                                # noqa: E402
from config import load_config, strip_overrides                              # noqa: E402

NOTE_STEM = "2026-10-22_calibration_eval"


def stats(e_mm: np.ndarray) -> dict:
    e = e_mm[np.isfinite(e_mm)]
    if not len(e):
        return {"n": 0}
    return {"rms_mm": round(float(np.sqrt((e ** 2).mean())), 3), "p50_mm": round(float(np.median(e)), 3),
            "p90_mm": round(float(np.percentile(e, 90)), 3), "p99_mm": round(float(np.percentile(e, 99)), 3),
            "within_3mm": round(float((e <= 3.0).mean()), 4), "n": int(len(e))}


def residuals(take: str, cfg) -> dict:
    """{group: (T,k) mm residual, NaN where not observed} plus helpers."""
    tr = np.load(ROOT / cfg.data.out_root / take / f"mano_track{sfx(cfg)}.npz")
    rig = load_rig(take, cfg)
    kn = rig.bodies["knuckle"]
    # the knuckle cap where the TRACKED root puts it: T_world_from_canonical @ inv(T_knucklecap_from_canonical) --
    # the cap's own re-solve on the default run, the cap implied by the smoothed wrist on `smooth.wrist` runs
    from calibration.common import T_from_json
    calib = json.loads((ROOT / cfg.data.out_root / "calibration" / f"hand_calib{calib_sfx(cfg)}.json").read_text())
    T_wk = tr["T_world_from_canonical"].astype(float) @ np.linalg.inv(T_from_json(calib["T_knucklecap_from_canonical"]))[None]
    pred_kn = np.einsum("tij,mj->tmi", T_wk[:, :3, :3], kn.template_body) + T_wk[:, None, :3, 3]
    win = np.swapaxes(rig.float_world_raw_seq, 0, 1)                   # Stage B window labels, raw
    # ALWAYS against the RAW markers: a smoothed run is fitted to smoothed tracks, but it is judged on what was
    # measured (older tracks without the *_raw arrays were never smoothed)
    raw = lambda k: tr[f"{k}_raw"] if f"{k}_raw" in tr.files else tr[k]                  # noqa: E731
    tip_o, kn_o, fl_o = raw("tip_obs"), raw("knuckle_obs"), raw("float_obs")
    r = {"tip_cap": np.linalg.norm(tr["tip_pred"] - tip_o, axis=2) * 1000.0,
         "knuckle_cap": np.linalg.norm(pred_kn - kn_o, axis=2) * 1000.0,
         "float_proximal": np.linalg.norm(tr["float_pred"][:, 0] - fl_o[:, 0], axis=1) * 1000.0,
         "float_middle": np.linalg.norm(tr["float_pred"][:, 1] - fl_o[:, 1], axis=1) * 1000.0,
         "float_proximal_win": np.linalg.norm(tr["float_pred"][:, 0] - win[:, 0], axis=1) * 1000.0,
         "float_middle_win": np.linalg.norm(tr["float_pred"][:, 1] - win[:, 1], axis=1) * 1000.0}
    valid = tr["valid"]
    for k in r:
        r[k][~valid] = np.nan
    return {"r": r, "valid": valid, "index_pose": tr["index_pose"], "n": rig.n_frames}


def _hp(x: np.ndarray, valid: np.ndarray, half: int) -> np.ndarray:
    """|x - its centred (2 half + 1)-frame moving average| on frames whose whole window is valid."""
    k = np.ones(2 * half + 1) / (2 * half + 1)
    sm = np.stack([np.convolve(np.nan_to_num(x[:, a]), k, "same") for a in range(x.shape[1])], 1)
    cs = np.r_[0, np.cumsum(valid)]
    i = np.arange(half, len(x) - half)
    full = i[cs[i + half + 1] - cs[i - half] == 2 * half + 1]
    return np.linalg.norm(x[full] - sm[full], axis=1)


def output_smoothness(take: str, cfg, valid: np.ndarray) -> dict:
    """Jitter of the MANO OUTPUT about a 50 ms moving average: the index tip vertex (mm, world) and the index
    flexion angles (deg, MCP/PIP/DIP about canonical z) -- what a consumer of the track sees."""
    tr = np.load(ROOT / cfg.data.out_root / take / f"mano_track{sfx(cfg)}.npz")
    half = 6                                                    # 13 frames = 54 ms at 240 Hz
    tip = _hp(tr["tip_vertex_world"].astype(float) * 1000.0, valid, half)
    wr = _hp(tr["joints_world"][:, 0].astype(float) * 1000.0, valid, half)      # MANO wrist joint
    mcp = _hp(tr["joints_world"][:, 1].astype(float) * 1000.0, valid, half)     # index MCP joint
    out = {"tip_vertex_jitter_mm_rms_p95": [round(float(np.sqrt((tip ** 2).mean())), 4), round(float(np.percentile(tip, 95)), 4)],
           "wrist_joint_jitter_mm_rms_p95": [round(float(np.sqrt((wr ** 2).mean())), 4), round(float(np.percentile(wr, 95)), 4)],
           "index_mcp_jitter_mm_rms_p95": [round(float(np.sqrt((mcp ** 2).mean())), 4), round(float(np.percentile(mcp, 95)), 4)]}
    th = np.degrees(tr["index_pose"].astype(float))
    for j, name in enumerate(("mcp", "pip", "dip")):
        e = _hp(th[:, 3 * j + 2:3 * j + 3], valid, half)
        out[f"{name}_flex_jitter_deg_rms_p95"] = [round(float(np.sqrt((e ** 2).mean())), 4), round(float(np.percentile(e, 95)), 4)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(strip_overrides())
    cfg = load_config()
    calib = json.loads((ROOT / cfg.data.out_root / "calibration" / f"hand_calib{calib_sfx(cfg)}.json").read_text())
    fitz = np.load(ROOT / cfg.data.out_root / "calibration" / f"hand_calib{calib_sfx(cfg)}.npz")
    out = {"source": "evaluation/fit_error.py", "calibration": f"outputs/calibration/hand_calib{calib_sfx(cfg)}.json", "run_tag": str(cfg.run.tag or ""), "takes": {}}
    groups = ("tip_cap", "knuckle_cap", "float_proximal", "float_middle", "float_proximal_win", "float_middle_win")
    for take in ("calibration", "free_motion"):
        R = residuals(take, cfg)
        n = R["n"]
        splits = {"all": np.ones(n, bool)}
        if take == "free_motion":
            fit_mask = np.zeros(n, bool)
            fit_mask[fitz["free_idx"]] = True
            held = holdout_mask(n, cfg.fit)
            splits.update({"fit": fit_mask, "heldout": held, "rest": ~held & ~fit_mask})
        res = {}
        for sp, m in splits.items():
            res[sp] = {g: stats(R["r"][g][m]) for g in groups}
            res[sp]["frames"] = int((m & R["valid"]).sum())
        # coverage: frames with a (re-claimed) floating observation among tracked frames
        for g in ("float_proximal", "float_middle"):
            res["all"][g]["coverage"] = round(float(np.isfinite(R["r"][g]).sum() / R["valid"].sum()), 4)
        # residual vs PIP flexion (deg, about canonical z)
        pip = np.degrees(R["index_pose"][:, 5])
        bins = [-20, 10, 30, 50, 70, 130]
        res["float_vs_pip_flexion"] = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pip >= lo) & (pip < hi)
            if m.sum() < 50:
                continue
            res["float_vs_pip_flexion"][f"{lo}..{hi} deg"] = {
                "frames": int(m.sum()),
                "proximal_rms_mm": stats(R["r"]["float_proximal"][m]).get("rms_mm"),
                "middle_rms_mm": stats(R["r"]["float_middle"][m]).get("rms_mm"),
                "tip_rms_mm": stats(R["r"]["tip_cap"][m]).get("rms_mm")}
        res["output_smoothness"] = output_smoothness(take, cfg, R["valid"])
        ip = np.degrees(R["index_pose"][R["valid"]]).reshape(-1, 3, 3)
        res["index_flexion_range_deg_p1_p99"] = {j: np.percentile(ip[:, k, 2], [1, 99]).round(1).tolist()
                                                 for k, j in enumerate(("mcp", "pip", "dip"))}
        out["takes"][take] = res
    out["floating_height_mm"] = {L: round(v["height_mm"], 3) for L, v in calib["floating"].items()}
    out["floating_radius_mm"] = float(cfg.mosca.floating.radius_mm)
    write_json(ROOT / "notes" / f"{NOTE_STEM}{sfx(cfg)}.json", out)

    # ---- markdown tables
    print("\n| take | split | frames | tip cap RMS | knuckle cap RMS | float prox RMS | float mid RMS |")
    print("|---|---|---|---|---|---|---|")
    for take, res in out["takes"].items():
        for sp in ("all", "fit", "heldout", "rest"):
            if sp not in res:
                continue
            g = res[sp]
            print(f"| {take} | {sp} | {g['frames']} | " + " | ".join(
                f"{g[k]['rms_mm']:.2f} (p90 {g[k]['p90_mm']:.2f})" for k in
                ("tip_cap", "knuckle_cap", "float_proximal", "float_middle")) + " |")
    print("\n| take | float label | RMS after re-claim | RMS on window labels | coverage |")
    print("|---|---|---|---|---|")
    for take, res in out["takes"].items():
        for L in ("proximal", "middle"):
            a, b = res["all"][f"float_{L}"], res["all"][f"float_{L}_win"]
            print(f"| {take} | {L} | {a['rms_mm']:.2f} (n {a['n']}) | {b['rms_mm']:.2f} (n {b['n']}) | {a['coverage']:.3f} |")
    print("\nfloat residual vs PIP flexion (free_motion):")
    for k, v in out["takes"]["free_motion"]["float_vs_pip_flexion"].items():
        print(f"  {k:12s} {v}")
    print("flexion range (p1..p99 deg):", out["takes"]["free_motion"]["index_flexion_range_deg_p1_p99"])


if __name__ == "__main__":
    main()
