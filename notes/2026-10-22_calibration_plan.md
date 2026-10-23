# PLANNED pipeline: index-finger MANO ↔ Motive calibration (allsense_glove)

> The plan as approved on 2026-10-22, before implementation. What was actually built, the deviations from this
> plan and the measured results are in `notes/2026-10-22_calibration_pipeline.md`.

## Context
Goal: a calibrated chain from mocap to the MANO index finger, so that relative finger motion can later be compared
against mocap ground truth. The recordings are from 2026-10-18 (Motive CSV 1.25, 240 Hz, mm, quaternions), in
`/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/data/motive/`.

* **`calibration_1018`** (8802 frames): the open hand rests in the printed fixture, and fixture plus hand slide together on the
  table. The motion is planar (Y span < 6 mm).
  * fixture↔knuckle-cap distance is 63.8 mm (p1–p99 62.1–64.4), knuckle↔tip is 91.7 mm (86.9–92.1).
  * The hand is near-rigid in the fixture, with ~2 mm slip.
* **`free_motion`** (20241 frames): the fixture is static and the hand moves freely; knuckle↔tip ranges 84–108 mm.
* **Rigid bodies:**
  * `jiawei_hand_fixture` (18 markers)
  * `jiawei_knuckle_00` (5 markers, on the hand back near index MCP)
  * `jiawei_fingertip_00` (5 markers, mounted on a glove wrap over the index tip)
* **Motive's RB orientations are unreliable.** In the rigid calibration take, tip-cap rotation relative to the knuckle cap
  jumps by up to 175° (p90 175°), and the tip position in the knuckle frame spans 180 mm. The 5-marker caps flip between
  symmetric solutions, so a MoSca re-solve is required.
* **Two floating markers (r ≈ 3 mm)** appear only as `Unlabeled NNNN` tracks:
  * In calibration, `4383` is on the middle phalanx (27 mm to the tip cap) and `4384` is on the proximal phalanx (39 mm to the knuckle cap).
  * In free motion, `6146` is proximal; the middle-phalanx marker is split into `6147` and `6156`.
* **Assets** (in `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/data/assets/`):
  * `custom_pose.json` holds MANO right-hand betas and pose in MANO canonical (wrist [95.7, 6.4, 6.2] mm, `flat_hand_mean`).
  * `hand_aligned_mm.obj` is that same mesh in the fixture CAD frame. Kabsch residual 0.0 mm gives
    `T_cad_from_canonical`: R=[[0,0,-1],[-1,0,0],[0,1,0]], t=[7.0, 128.79, 24.47] mm.
  * `hand_fixtured.glb` is the fixture CAD (Onshape, m).
  * `index_aligned_mm.obj` is the scanned glove-wrapped index tip in the CAD frame; the cap mounts on top of it.
  * The fixture CAD has no known marker positions, so **`T_fixture_from_cad` is solved from data** (user's choice).

**Deliverables:**
1. Refined MANO betas.
2. `T_tipcap_from_distal` (fingertip cap ↔ index distal bone) and `T_knucklecap_from_canonical` (MCP cap ↔ hand root/wrist).
3. MANO surface correspondence (face + barycentric + normal offset) of the two floating markers.
4. RMS fit error (mm) for the cap markers and for the floating markers.
5. A 3D HTML viewer, plus planned and final pipeline notes.

## Environment and style
* Run with `~/miniforge3/envs/far_grasp/bin/python` (the TableMocap env: torch 2.11 cu128, smplx, scipy, trimesh).
* **MANO model:** `/home/ANT.AMAZON.COM/raofuu/Code/TableMocap-toolkit/data/mano/mano/MANO_RIGHT.pkl`.
  Use the smplx layer with `flat_hand_mean=True` and the full 45-D axis-angle pose (no PCA); the config points at this path.
* **Follow `TableMocap-toolkit-padab/kinematics-replay-bimanual/CONVENTIONS.md`:**
  * `X_<to>_from_<from>` names, `p_<frame>`, `_seq` / `_mm` / `_xyzw` / `_idx`.
  * Internals in metres; mm only in reports and gates.
* **Module and docstring style:**
  * Every module opens with `#!/usr/bin/env python3` and a long docstring: the model as maths, WHY paragraphs with measured
    numbers, and the run command.
  * `from __future__ import annotations`; `@dataclass` results; shape comments like `(T,5,3)`.
  * argparse with `description=__doc__.split("\n")[0]`.
  * YAML config with dotted `key=value` overrides via a small `config.py`, copied from `kinematics-replay-bimanual/config.py`.
  * `print(..., flush=True)` tables, `json.dumps(indent=2)` plus an `.npz` sidecar, then print `wrote <path>`.
* **Folders:**
  * `calibration/`: main code
  * `mosca/`: vendored marker reassignment
  * `vis/`: viewer builder
  * `evaluation/`: metrics
  * `scripts/`: one-off checks
  * `notes/`: write-ups, plus `notes/figures/` for the viewer HTML
  * `outputs/`: bulk artefacts
  * `conf/`: YAML config

## Reuse (copy with a provenance header: source path + commit `387e3c3`)
From `/home/ANT.AMAZON.COM/raofuu/Code/TableMocap-toolkit-padab/kinematics-replay-bimanual/` (KRB):
* **`fitting/reassignment/tracklets.py`** (`link`, `summary`) and **`shape.py`** (`kabsch`, `consensus`, `place`, `fit_dropping`):
  numpy/scipy only, copied verbatim to `mosca/`.
* **`fitting/reassignment/scaffold.py`** (`build`, `groups`, `name_groups`, `seed_from_motive`, `resolve`, `confidence`):
  copied to `mosca/scaffold.py`, with the import path changed. All three bodies are tagged `kind="object"`, which skips the
  cap/plate gating that needs `plate_motive`/`cap_anchor`, and `sc.emf = {}`. The config subset goes into `conf/mosca.yaml`,
  copied from `conf/reassignment.yaml`.
* **`data_io/asset_parser.py:132` `parse_bodies`**: the `.motive` XML parser. The function alone is copied into
  `mosca/take.py`, since `load_asset` requires full glove hands.
* **`frames.py` `Obs`/`Body`/`Take` containers**: re-declared minimally in `mosca/take.py`.
* **`calibration/mano_model.py`**: the wrapper pattern. LBS that returns per-joint global transforms follows
  `hand_scan_processing/fit_mano.py:180`, using `smplx.lbs.batch_rigid_transform`.
* **`calibration/solve_plate_binding.py:151` `procrustes`, `:164` `ball_offset`**: Adam param groups with blocks toggled,
  and betas and offsets never free together.
* **`scripts/finger_bend/cap_cad_fit.py` `offset_surface` / `icp`**: marker centres sit one radius off a CAD surface. Used
  for fixture-markers → `hand_fixtured.glb`.
* **`fitting/hands/mano_fit/finger_chain.py` `range_penalty` / `flexion_deg`**: index joint limits and readouts.
* **Viewer:**
  * the self-contained three.js page pattern from `scripts/mosca_steps/common.py:91-108` (`b64`, `positions`, `page`)
    and `scripts/marker_assignment_mosca/viewer.js`
  * LBS-in-JS from `vis/export_motive_bundle.py:334`: the rest mesh and skin weights are sent once, then 16 bone
    transforms per frame
  * `vis/vendor/three.min.js` copied

## Pipeline and files

### Stage A — load: `mosca/take.py`
* `load_csv(path) -> MotiveCsv` parses the 7-row header and returns:
  * Motive RB poses per body as `R_world_from_body_seq` and `t_world_from_body_seq` (zero/NaN quaternion → invalid)
  * marker observations as a flat `Obs`: `frame`, `p` mm, `owner`/`row` from `Body:Marker NNN` columns, -1 for
    `Unlabeled`; the `Rigid Body Marker` columns are ignored
* `load_take(csv, assets) -> Take`.

### Stage B — MoSca re-identification and RB re-solve: `mosca/run.py`
1. `tracklets.link`, then `scaffold.resolve` labels the cap and fixture marker segments, using Motive labels only as a
   segment-level seed.
2. **Template refinement:** generalised Procrustes over the well-covered frames, starting from the `.motive` offsets.
   Re-solve the poses with `consensus` (Kabsch + outlier drop + nearest-previous tie-break, which removes the flips).
3. **Floating markers** (`mosca/floating.py`): `scaffold` cannot name single markers (bodies need ≥3 rows). The custom
   rule:
   * candidate segments are free segments whose distance band to the posed knuckle cap and tip cap stays inside the
     phalanx windows in `conf/mosca.yaml` (proximal: near the knuckle; middle: near the tip)
   * Hungarian matching per frame, so each frame gets one marker per label
   * split tracks (`6147` + `6156`) are merged under one label
   * after the first MANO fit (Stage D), segments are re-claimed against the predicted positions within a gate, in
     the style of `recover`
4. **Output** in `outputs/<take>/mosca.npz`:
   * labelled marker observations (body, row)
   * re-solved poses `R/t_world_from_<body>_seq`, plus rmse and valid flags
   * floating marker tracks `p_world_seq` with NaN gaps
   * Motive's original poses, for comparison
   * a summary JSON (coverage, flip count vs Motive, per-body rmse)

### Stage C — rigid calibration summary and initialisation: `calibration/init_rig.py`
* **Calibration take:** robust medians of `T_fixture_from_knucklecap`, `T_fixture_from_tipcap`, and the floating
  markers' `p_fixture`. The slip (per-frame deviation) is reported.
* **`T_fixture_from_cad` initialisation:**
  1. Kabsch on 4 anatomical correspondences: knuckle-cap centroid ↔ dorsal index-MCP surface, tip-cap centroid ↔
     dorsal tip of `index_aligned_mm.obj`, and the floating markers ↔ dorsal midpoints of the proximal and middle
     phalanges. All of these come from `hand_aligned_mm.obj` in the CAD frame.
  2. Refine by ICP of the 18 fixture marker centres onto the CAD surface offset by each marker's radius (diameters from
     the `.motive` file).
* **Initial bindings:**
  * `T_knucklecap_from_canonical = inv(T_fixture_from_knucklecap) @ T_fixture_from_cad @ T_cad_from_canonical`
  * `T_tipcap_from_distal`: from the MANO distal-bone global transform at the custom pose
  * floating anchors: the nearest dorsal surface points in the proximal/middle bone frames

### Stage D — joint MANO fit: `calibration/mano_index.py`, `calibration/fit_hand.py`
* **`mano_index.py`:** `ManoIndex` (smplx right layer) with:
  * `forward(betas, hand_pose) -> vertices (B,778,3), T_canonical_from_joint (B,16,4,4)`
  * index joints 1/2/3 = MCP/PIP/DIP; the tip vertex is 320 (KRB) — check which tip vertex index is correct (KRB uses
    320, `has/mano.py` uses 317)
  * vertex normals; point-to-surface distance restricted to index-finger faces
* **Shared unknowns:**
  * `betas` (init and prior: custom_pose `final_betas`)
  * `T_knucklecap_from_canonical`, `T_tipcap_from_distal` (6 each; translation via `ball_offset`)
  * `T_fixture_from_cad` (6)
  * `p_floatprox_bone1`, `p_floatmid_bone2` (3 each)
* **Per-pose unknowns:**
  * `θ_cal`: one 45-D pose for the calibration take, prior to the custom pose
  * free motion: index joints 1–3 (9-D) per frame for K ≈ 600 frames, chosen by farthest-point sampling on the
    tip-cap-in-knuckle-cap pose; the other fingers are frozen at `θ_cal`
  * free-motion root: `T_world_from_canonical_seq = T_world_from_knucklecap_seq @ T_knucklecap_from_canonical`
  * calibration-take root: `T_world_from_fixture_seq @ T_fixture_from_cad @ T_cad_from_canonical`
* **Residuals** (Huber/Geman–McClure, mm-scaled):
  * **Tip cap markers:** `T_world_from_canonical @ T_canonical_from_distal(θ) @ inv(T_tipcap_from_distal) @ x_row` minus the
    observed marker. Used in both takes.
  * **Knuckle cap markers:** calibration take only. In free motion they define the root, so they are not a residual
    there.
  * **Floating markers:** the predicted bone-frame point minus the observed marker.
  * **Floating-marker surface term:** `|dist(p_float, posed MANO surface) − r|`, with r = 3 mm + δ, δ ∈ [0, 2] mm. This is
    what pins the "correspondence on the hand back".
  * **Fixture term:** fixture markers on the offset CAD surface, which keeps `T_fixture_from_cad` honest.
  * **Priors:** index joint range penalty, `θ_cal` → custom pose, betas → init, and temporal smoothness when fitting
    full sequences.
* **Schedule** (as in `solve_plate_binding._one_start`, with Adam param groups whose lr is set to 0 to freeze a block):
  1. poses only
  2. bindings + floating anchors + poses
  3. betas + poses (bindings frozen)
  4. alternate 2 and 3 ×2
  5. LBFGS polish
* **Outputs:**
  * `outputs/calibration/hand_calib_1018.json`: betas, all transforms as R/t plus `q_xyzw`, floating correspondences
    (nearest face id, barycentric, vertex id, normal offset mm), index bone lengths before/after, and the model
    equations as strings
  * an `.npz` sidecar

### Stage E — sequence tracking with the frozen calibration: `calibration/track.py`
* Per-frame index pose for every frame of both takes, batched on GPU, warm-started frame to frame, with smoothness.
* Writes `outputs/<take>/mano_track.npz` with pose, joints, bone transforms and predicted markers.

### Stage F — evaluation: `evaluation/fit_error.py`
* RMS and p50/p90/p99 (mm) per group: tip-cap markers, knuckle-cap markers (calibration), floating proximal, floating
  middle.
* Split into fit frames (K) vs held-out frames (all other free-motion frames, plus contiguous time blocks held out
  from the fit).
* Reports the floating-marker surface offset vs the expected ~3 mm, the betas change, index bone lengths, and the RB
  flip count Motive vs MoSca.
* Writes `notes/2026-10-22_calibration_eval.json` and the table into the final note.

### Viewer: `vis/build_viewer.py`, `vis/viewer.js`, `vis/viewer_body.html`, `vis/viewer.css`, `vis/vendor/three.min.js`
* A self-contained three.js page per take, output to `notes/figures/hand_calib_<take>.html`.
* **Embedded data:**
  * markers coloured by MoSca label, as int16 tenths of a mm + `ptr`
  * re-solved vs Motive RB axes (to show the flips)
  * the MANO mesh via LBS-in-JS (rest verts + top-4 skin weights once, 16 bone transforms per frame)
  * predicted cap/floating markers with residual lines
* Downsampled to 60 Hz to keep the file small.
* Controls: timeline scrub, play/speed, layer toggles, orbit camera (ported from `viewer.js`).

### Notes
* `notes/2026-10-22_calibration_plan.md`: the planned pipeline (this plan, as a design doc). Written first.
* `notes/2026-10-22_calibration_pipeline.md`: the final pipeline with stages, frames and transforms, measured results
  tables, re-run commands, outputs list, and known limitations.
* `scripts/check_rb_flips.py`: the Motive-vs-MoSca orientation-jump diagnostic; its numbers go into the final note.
* `run_all.sh`: runs stages A→F plus the viewer.

## Risks / judgement calls
* **Knuckle cap as a rigid root.** It sits near the MCP, so skin slide over the MCP during flexion will appear as
  knuckle residual. Report the knuckle-in-calibration residual separately, and check the free-motion floating-proximal
  residual against MCP flexion.
* **Shape observability.** Only the index finger and palm are observed, so the other fingers' shape is held by the
  betas prior. Bone lengths are the readout.
* **Floating-marker naming in free motion** relies on the distance windows and then the MANO-predicted re-claim. Report
  the coverage per label.

## Verification
1. **Stage A:** confirm the frame count, the 3 bodies, and the marker counts against the numbers above.
2. **Stage B:** the RB rmse per body is < 0.5 mm and the orientation jumps (p99 relative tip-to-knuckle rotation in the
   calibration take) drop from ~175° to a few degrees. Floating labels cover ≥ 95% of the frames where the marker is
   visible.
3. **Stage C:** the residual of the `T_cad_from_canonical` Kabsch is 0.0 mm. Fixture ICP gives a marker-to-offset-surface
   RMS of about 1 mm.
4. **Stage D/E:** the losses decrease. Target RMS: tip-cap markers ≲ 2 mm, floating markers ≲ 3 mm (the 3 mm marker
   radius sets the scale), with held-out RMS close to the fit RMS.
5. Run `evaluation/fit_error.py` and inspect the HTML viewer in a browser. MANO should stay on the markers through the
   free motion, and the caps should no longer flip.
