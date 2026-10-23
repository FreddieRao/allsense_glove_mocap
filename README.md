# allsense_glove: index-finger MANO ↔ Motive calibration

This project calibrates a MANO right-hand model to OptiTrack/Motive mocap of the index finger.

**The rig:**

* a knuckle cap on the hand back (near index MCP)
* a fingertip cap on a glove-wrapped index tip
* two large (~3 mm radius) floating markers on the proximal and middle phalanges
* a printed hand fixture, used for the calibration take

**What the pipeline produces:**

* refined MANO hand shape (betas, plus small index joint-pivot corrections)
* the rigid bindings **fingertip cap ↔ index distal bone** and **knuckle cap ↔ MANO root (wrist)**
* the MANO skin correspondence of both floating markers (face, barycentric point, height above skin)
* a per-frame index-finger track of both takes
* the fit error at every marker, and 3D HTML viewers

## Quick start

```bash
cd /home/ANT.AMAZON.COM/raofuu/Code/allsense_glove
bash run_all.sh                 # everything, ~20 min
bash run_all.sh --skip-mosca    # reuse outputs/<take>/mosca.npz (skips the ~7 min MoSca stage)
```

* **Python:** `~/miniforge3/envs/far_grasp/bin/python` (torch + CUDA, smplx, scipy, trimesh, fast_simplification).
* **MANO:** `/home/ANT.AMAZON.COM/raofuu/Code/TableMocap-toolkit/data/mano/mano_clean/MANO_RIGHT.pkl`
  (`mano.model_dir`).
* **Config:** every knob is in `conf/calibration.yaml`, and any key can be overridden on the command line:

```bash
python -m calibration.fit_hand fit.weights.betas=1.0 smooth.method=butter run.tag=my_variant
```

* **Variants:** `run.tag=<tag>` gives every Stage D–F artefact and viewer a `_<tag>` suffix, so variants sit next to
  the default run. `run.calib_tag=default` makes a tracking variant reuse the default calibration.

## Data (`data/`)

| path | what |
|---|---|
| `data/motive/calibration_1018.csv` | Motive CSV 1.25, 240 Hz: open hand rigid in the fixture, fixture + hand slide on the table |
| `data/motive/free_motion.csv` | fixture static, the hand and index finger move freely |
| `data/motive/jiawei_{hand_fixture,knuckle_00,fingertip_00}_1018.motive` | rigid-body definitions (marker offsets in the body frame, diameters) |
| `data/assets/custom_pose.json`, `custom_pose_metric.obj` | MANO betas + pose of the designed hand (canonical frame) |
| `data/assets/hand_aligned_mm.obj` | the same MANO mesh placed in the fixture CAD frame (the **designed hand**) |
| `data/assets/hand_fixtured.glb` | fixture CAD (m) |
| `data/assets/index_aligned_mm.obj` | scanned glove-wrapped index tip, CAD frame |
| `data/hand_scan/` | a separate hand scan + its MANO fit (compared in `notes/2026-10-22_scan_vs_mocap_mano.md`) |

The Motive data comes from Hugging Face `bby007/allsense_glove`:

```bash
hf download bby007/allsense_glove --repo-type dataset --local-dir data/motive
```

## Pipeline

| stage | code | does | output |
|---|---|---|---|
| A load | `mosca/take.py` | Motive CSV → flat marker observations (Motive labels kept only as weak priors), Motive RB poses, `.motive` templates | – |
| B MoSca | `mosca/run.py`, `mosca/{tracklets,shape,scaffold}.py` (vendored from TableMocap), `mosca/floating.py` | marker tracklets → rigidity groups → template naming → local naming → Motive seed → pose / verify / split / recover loop; per-take template refinement; caps and fixture re-solved per frame (flip-free); floating markers named | `outputs/<take>/mosca.{npz,json}` |
| C init | `calibration/init_rig.py` | fixture → CAD registration; constant relative poses of the calibration take; initial bindings from the designed hand | `outputs/calibration/init_rig.json` |
| D fit | `calibration/fit_hand.py`, `hand_model.py`, `mano_index.py` | joint fit of betas, index pivot corrections, both bindings and both floating skin points | `outputs/calibration/hand_calib.{json,npz}` |
| E track | `calibration/track.py` | per-frame index pose for every frame, calibration frozen; floating labels re-claimed against MANO; optional smoothed wrist | `outputs/<take>/mano_track.{npz,json}` |
| F eval | `evaluation/fit_error.py` | RMS per marker group and split (fit / held-out / rest), against the RAW markers; output jitter | `notes/2026-10-22_calibration_eval.json` |
| viewers | `vis/build_viewer.py`, `vis/build_fixture_viewer.py` | self-contained three.js pages | `notes/figures/*.html` |

### Stage C: fixture → CAD registration

Every fixture marker is snapped either **on top** or **at the side**:

* **Top markers.** Markers 8–17 of the asset are coplanar (±0.06 mm). Their centres go one radius above the CAD's
  top face (z = 35 mm), which fixes tilt and height.
* **Search.** Yaw and x/y are searched with a planar ICP.
* **Polish.** A weighted 6-dof polish finishes it: top markers on the top face, side markers on walls, side markers
  weighted 5× (they set the in-plane direction).
* **Marker positions** are the `.motive` asset's defined offsets, not refined on the take.

### The model (`calibration/hand_model.py`)

```
T_world_from_canonical[i] = T_world_from_knucklecap[i] @ T_knucklecap_from_canonical
tip cap marker r   = T_world_from_canonical[i] @ T_canonical_from_joint[3](betas, theta_i) @ T_distal_from_tipcap @ x_r
floating marker m  = T_world_from_canonical[i] @ ( sum_k bary_mk v_k(betas, theta_i) + h_m n(betas, theta_i) )
```

* **Root.** There is no wrist marker set, so the knuckle cap is the MANO root.
* **Designed-hand prior.** The designed hand (`hand_aligned_mm.obj` through the fixture registration) is the start of
  every binding and a prior of 2 mm / 2° on where the hand sits in the fixture.
* **Index pose.** `theta_i` is 9 numbers per frame (MANO index MCP/PIP/DIP axis-angle); the other joints are held at
  the calibration pose.

### Smoothing (`mosca/smooth.py`)

* **Default.** Every raw marker track is smoothed on its own: RTS, constant velocity, 12 Hz, 0.1 mm marker noise. The
  caps are then re-solved from the smoothed markers. Errors are still measured against the raw markers.
* **Optional smoothed wrist** (`smooth.wrist.enable`, off by default). The fitted root is smoothed at the wrist joint
  (RTS, 4 Hz), and the index is re-tracked against it. `run_all.sh` produces it as the `wristsmooth` variant, overlaid
  in the hand viewers.

## Current results (default run)

These are RMS values over every tracked frame, against the raw markers (`notes/2026-10-22_calibration_eval.json`):

| take | tip cap | knuckle cap | floating proximal | floating middle |
|---|---|---|---|---|
| calibration | 0.31 mm | 0.18 mm | 0.48 mm | 0.34 mm |
| free motion | 0.60 mm | 0.18 mm | 1.86 mm | 1.21 mm |
| free motion, held-out blocks | 0.56 mm | 0.20 mm | 1.36 mm | 1.12 mm |

Other results:

* **Fixture → CAD gap:** 0.13 mm rms for the top markers and 0.27 mm for the side markers.
* **Fitted hand vs the design:** 0.6 mm / 0.3°.
* **Index bone lengths:** proximal 33.1, middle 22.3, DIP–tip 25.6 mm.
* **Floating markers:**
  * proximal phalanx: face 346, vertex 261, 4.0 mm above the skin;
  * middle phalanx: face 53, vertex 86, 5.0 mm above the skin.
* **Smoothed-wrist variant:** wrist jitter 0.100 → 0.025 mm. Finger errors are unchanged; the knuckle cap goes to
  0.24 mm.

## Viewers (`notes/figures/`)

* **`hand_calib_free_motion.html` and `hand_calib_calibration.html`.** These show:
  * markers by MoSca label, and the MANO hand skinned in the page;
  * predicted vs observed cap, floating and knuckle-cap markers, with residual lines;
  * MoSca vs Motive cap axes;
  * the optional smoothed-wrist overlay (`#ovl` in the URL);
  * a residual timeline with the held-out blocks shaded. `#f=<frame>` jumps to a frame.
* **`fixture_registration.html`.** The fixture markers on the CAD at their true radius, coloured by gap, every
  candidate registration, and the designed vs fitted hand.

## Layout

| folder | contents |
|---|---|
| `mosca/` | marker re-identification and rigid-body re-solve (MoSca), smoothing |
| `calibration/` | Stage C–E: init, MANO model, joint fit, tracking |
| `evaluation/` | fit-error metrics |
| `scripts/` | diagnostics: `check_rb_flips.py`, `rb_jitter.py`, `compare_smoothing.py`, `compare_scan_mano.py` |
| `vis/` | viewer builders, JS/CSS, vendored three.js |
| `conf/calibration.yaml` | all configuration |
| `outputs/` | bulk artefacts (npz/json) |
| `notes/` | write-ups and results; each says which script produced it and how to re-run it |

Naming follows TableMocap's `CONVENTIONS.md`:

* transforms are `X_<to>_from_<from>` (e.g. `T_knucklecap_from_canonical`);
* points are named with their frame (`p_world`);
* `_seq` marks a per-frame sequence;
* internals are in metres, with `_mm` only in reports;
* MoSca internals are in mm, because its gates are.

## Notes

| note | about |
|---|---|
| `notes/2026-10-22_calibration_plan.md` | the planned pipeline, as approved |
| `notes/2026-10-22_calibration_pipeline.md` | the final pipeline, deviations from the plan, and limitations (revision 2) |
| `notes/2026-10-22_prior_and_smoothing.md` | prior strength, pose vs marker smoothing, defaults, smoothed wrist |
| `notes/2026-10-22_scan_vs_mocap_mano.md` | hand-scan MANO vs mocap MANO (run on revision 1) |

## Known limitations

* **PIP/DIP split.** With the designed-hand prior and the index pivot corrections (MCP moves ~8.5 mm), tracked PIP
  reaches ~107° while DIP stays within −11…18°. The split between the two joints is not trusted yet.
* **Floating-marker heights** sit at their [3, 5] mm bounds, so MANO's dorsal skin does not match this hand exactly.
* **The knuckle cap is a rigid root** on the hand back; skin slide there is invisible to the model.
* `data/motive/Take 2026-10-17 05.39.06 PM.tak` has no CSV export and is not used.
