# FINAL pipeline: index-finger MANO ↔ Motive calibration (2026-10-22, revision 2)

> **Revision 2.**
> * The fixture → CAD registration is now plane-constrained on the `.motive` asset's DEFINED marker
>   positions: 0.20 mm rms, unambiguous.
> * It is held fixed.
> * The designed hand (`hand_aligned_mm.obj`) is the start of every binding and a soft prior.
> * Hand shape and both bindings were recalibrated on top of this. Every number below is from this revision.
>
> The revision-1 numbers (unconstrained fixture ICP) are kept only where marked.

## Question

Given the two Motive takes recorded on 2026-10-18 (`calibration_1018`: the open hand rests in the printed fixture
and slides with it; `free_motion`: the fixture is static and the index finger moves freely), find:

1. refined MANO hand-shape parameters (betas);
2. the rigid transforms fingertip cap ↔ index distal bone and knuckle (MCP) cap ↔ MANO root (wrist);
3. the MANO skin correspondence of the two large floating markers (proximal and middle phalanx);

then report the RMS of the MANO fit at the cap markers and at the floating markers (the floating markers are about
3 mm in radius). The planned version is `notes/2026-10-22_calibration_plan.md`.

## Re-run

```bash
cd /home/ANT.AMAZON.COM/raofuu/Code/allsense_glove
bash run_all.sh                 # A-F + all viewers, ~15 min (MoSca ~7 min, fit ~3 min, tracking ~5 min)
bash run_all.sh --skip-mosca    # reuse outputs/<take>/mosca.npz, ~8 min
```

* **Python:** `~/miniforge3/envs/far_grasp/bin/python` (the TableMocap env: torch 2.11 cu128, smplx, scipy, trimesh).
* **Config:** `conf/calibration.yaml`. Any key can be overridden on the command line, e.g. `fit.weights.betas=1.0`.

## Pipeline

| stage | code | does | output |
|---|---|---|---|
| A load | `mosca/take.py` | Motive CSV 1.25 → flat marker observations (Motive labels as weak priors), Motive RB poses, `.motive` templates | – |
| B MoSca | `mosca/run.py` (+ `mosca/{tracklets,shape,scaffold}.py` vendored from TableMocap, `mosca/floating.py`) | tracklets → rigidity groups → template naming → **local naming** → Motive seed → pose/verify/split/recover/partner loop; per-take template refinement for the POSE solve; floating markers named by distance windows to both caps | `outputs/<take>/mosca.{npz,json}` |
| C init | `calibration/init_rig.py` | constant relative poses in the calibration take (and their slip); `T_cad_from_canonical` (exact); **plane-constrained fixture → CAD fit on the asset's defined marker positions**; initial bindings from the DESIGNED hand | `outputs/calibration/init_rig.json` |
| D fit | `calibration/fit_hand.py`, `calibration/hand_model.py`, `calibration/mano_index.py` | joint fit of betas, both bindings and both floating skin points over 200 calibration + 600 FPS free-motion frames; fixture registration held, designed hand as soft prior | `outputs/calibration/hand_calib.{json,npz}` |
| E track | `calibration/track.py` | per-frame index pose for every frame of both takes, calibration frozen; MANO re-claim of the floating labels | `outputs/<take>/mano_track.{npz,json}` |
| F eval | `evaluation/fit_error.py`, `scripts/check_rb_flips.py` | RMS per marker group / split; Motive vs MoSca pose stability | `notes/2026-10-22_calibration_eval.json`, `notes/2026-10-22_rb_flips.json` |
| viewers | `vis/build_viewer.py`, `vis/build_fixture_viewer.py` (+ `vis/*.js`, `viewer.css`, `viewer_body.html`) | self-contained three.js pages (TableMocap pattern) | `notes/figures/hand_calib_<take>.html`, `notes/figures/fixture_registration.html` |

### The model (`calibration/hand_model.py`)

The model for frame i of either take, in metres, in MANO's canonical frame (smplx right, `mano_clean`,
`flat_hand_mean=True`, 45-D axis-angle):

```
T_world_from_canonical[i] = T_world_from_knucklecap[i] @ T_knucklecap_from_canonical
tip cap row r  = T_world_from_canonical[i] @ T_canonical_from_joint[3](betas, theta_i) @ T_distal_from_tipcap @ x_r
floating m     = T_world_from_canonical[i] @ ( sum_k bary_mk v_k(betas, theta_i) + h_m n(betas, theta_i) )

designed hand  : T_fixture_from_canonical_design = T_fixture_from_cad @ T_cad_from_canonical
start          : T_knucklecap_from_canonical_0   = inv(T_fixture_from_knucklecap) @ T_fixture_from_canonical_design
prior          : T_fixture_from_knucklecap @ T_knucklecap_from_canonical ~ T_fixture_from_canonical_design  (sigma 5 mm, 5 deg)
```

* **`T_world_from_knucklecap[i]`** is the MoSca re-solve.
* **`T_canonical_from_joint[3]`** is MANO's index DIP bone frame.
* **`theta_i`** is 9 numbers per frame (MANO index joints 1–3).
* **Floating markers** are bound to the SKIN (face, barycentric point, height h along the interpolated normal).
* **`T_cad_from_canonical`** is exact: `hand_aligned_mm.obj` is `custom_pose_metric.obj` moved into the CAD,
  with a Kabsch residual of 0.0 mm.

## Results

### 1. Motive's rigid bodies flip; the MoSca re-solve does not (`scripts/check_rb_flips.py`)

Marker residual is `|pose × template − accepted marker|`, in mm (p50 / p90 / p99):

| take | body | Motive's pose | MoSca re-solve |
|---|---|---|---|
| calibration | fixture | 0.13 / 0.37 / 0.80 | 0.11 / 0.30 / 0.73 |
| calibration | knuckle cap | **10.75 / 14.88 / 15.13** | 0.12 / 0.22 / 0.46 |
| calibration | tip cap | 0.11 / 2.37 / 8.55 | 0.09 / 0.17 / 0.29 |
| free_motion | fixture | 0.02 / 0.07 / 0.56 | 0.02 / 0.09 / 0.54 |
| free_motion | knuckle cap | 0.14 / 0.32 / 2.77 | 0.13 / 0.26 / 0.42 |
| free_motion | tip cap | 0.16 / **12.90 / 19.60** | 0.13 / 0.27 / 0.52 |

The tip cap's rotation relative to the knuckle cap:

* On the calibration take (hand rigid) it deviates from its mean by p50 / p90 / p99:
  * Motive: 24.5 / 174.8 / 176.5°
  * MoSca: 1.9 / 3.0 / 3.9°
* Frame steps over 20° at 240 Hz:
  * calibration: 558 (Motive) vs 0 (MoSca)
  * free_motion: 60 (Motive) vs 0 (MoSca)

### 2. Fixture → CAD registration (`calibration/init_rig.py`, viewer `notes/figures/fixture_registration.html`)

* **Which marker positions.** The fixture markers are the `.motive` asset's DEFINED offsets, not refined on the
  calibration take (`init.fixture_template: asset`). Each marker's radius is the asset's diameter / 2
  (3.0–3.6 mm).
* **The flatness.** In the asset, markers 8–17 are coplanar to 0.06 mm. The other 8 all sit on one side of that
  plane: 24.8, 13.9, 22.7, 13.4, 6.9, 25.0, 7.7, 14.8 mm away, on the fixture's lower steps.
* **The fit.** The ten coplanar centres go one radius above the CAD's largest upward-facing flat level
  (z = 35.0 mm, 7227 mm² of +z faces). That fixes tilt and height exactly. Yaw and x/y are then searched from a
  start every 2°, with a planar ICP over all 18 markers. A free 6-dof polish finishes it.

| fit | rms gap (marker ball ↔ CAD surface) | max | notes |
|---|---|---|---|
| **plane-constrained + polish (used)** | **0.20 mm** | 0.39 mm | polish moved 0.13° / 0.02 mm |
| next plane-constrained basins | 0.73 / 0.79 / 0.91 mm | 1.8–2.0 mm | 2–5° / 2–5 mm away |
| revision 1, unconstrained ICP basins | 1.23–1.80 mm | 2.4–4.2 mm | 5–17° apart; markers on side walls and the underside |

The registration is therefore certain, and Stage D holds it fixed (`fit.freeze_fixture_cad`).

### 3. MANO fit error (`evaluation/fit_error.py`, Stage E track, every frame)

Values are RMS in mm, with p90 in parentheses. "heldout" is time blocks 2 and 7 of 10, which never entered Stage D.

| take | split | frames | tip cap | knuckle cap* | floating proximal | floating middle |
|---|---|---|---|---|---|---|
| calibration | all | 8801 | **0.27** (0.42) | 0.16 (0.22) | **0.34** (0.56) | **0.27** (0.43) |
| free_motion | all | 20164 | **0.42** (0.63) | 0.17 (0.26) | **1.02** (1.57) | **0.97** (1.66) |
| free_motion | fit | 600 | 0.46 (0.68) | 0.17 (0.26) | 1.07 (1.69) | 1.06 (1.79) |
| free_motion | heldout | 4048 | 0.40 (0.59) | 0.19 (0.29) | 0.93 (1.50) | 0.75 (1.30) |
| free_motion | rest | 15516 | 0.42 (0.64) | 0.17 (0.25) | 1.04 (1.58) | 1.02 (1.72) |
| *revision 1, free_motion all* | | | *0.84* | *0.17* | *2.32* | *1.58* |

\* The knuckle cap IS the MANO root, so its number is its own rigid fit (the floor), not a MANO error.

* **Held-out error:** on the held-out blocks the error is no worse than on the fit frames.
* **Within the marker radius:** every tip-cap observation is within 3 mm. So are 99.4% (proximal) and 97.5% (middle)
  of the free-motion floating observations, and 100% of the calibration take.
* **Floating labels:** coverage after the MANO re-claim is 100% / 99.9%. Only 152 window labels changed.
* **Floating error vs flexion:** it is still largest with the finger extended. At PIP < 10° it is proximal 1.19 /
  middle 1.27 mm; at PIP 50–70° it is 0.67 / 0.44 mm.
* **Index flexion range** (p1–p99): MCP 8–83°, PIP −10–59°, DIP −15–78°.

### 4. Calibration deliverables (`outputs/calibration/hand_calib.json`)

**Betas** (start and prior: `custom_pose.json`, prior weight 0.1):

* fitted: `[0.029, -0.071, -0.028, -0.111, 0.004, -0.012, 0.020, 0.187, -0.094, -0.039]`
* index bone lengths at zero pose: wrist–MCP 90.45, proximal 32.94, middle 22.16, DIP–tip 25.49 mm
* initial lengths: 90.60 / 33.01 / 22.21 / 25.54 mm

The shape barely moves from `custom_pose.json`. Placing the hand correctly in the fixture removed the need for the
larger betas change of revision 1.

**Bindings** (translation in mm, `q_xyzw` scipy order; `distal` = MANO joint 3, the index DIP bone frame):

| transform | t (mm) | q_xyzw |
|---|---|---|
| `T_knucklecap_from_canonical` | [14.75, −32.92, −27.14] | [−0.1152, −0.6173, −0.0439, 0.7770] |
| `T_canonical_from_knucklecap` | [25.30, 25.43, 27.41] | [0.1152, 0.6173, 0.0439, 0.7770] |
| `T_distal_from_tipcap` | [−17.81, 17.60, −14.33] | [0.0693, 0.9476, 0.0585, 0.3062] |
| `T_tipcap_from_distal` | [−25.44, −13.41, −2.27] | [0.0693, 0.9476, 0.0585, −0.3062] |

Stage D moved them from the designed-hand start by 13.4 mm / 2.9° (knuckle) and 21.1 mm / 7.8° (tip).

**Where the hand actually sat in the fixture.** Relative to the designed `hand_aligned_mm.obj` the hand sits
13.4 mm / 2.9° away. The mean vertex offset in the CAD frame is:

* palm: [−6.1, −7.7, +7.1] mm
* index: [−15.0, −9.7, +7.0] mm

So the hand rests about 7 mm higher than designed (not pressed fully into the cavity) and about 8 mm toward the
wrist. The data prefers this over the design even against the 5 mm / 5° prior, so I take it as the real posture,
not a fitting artefact.

**Floating markers** (MANO right mesh, 778 vertices / 1538 faces):

| marker | face | barycentric | nearest vertex | height above skin | along its bone |
|---|---|---|---|---|---|
| proximal phalanx | 134 | [0.034, 0.731, 0.235] | 176 | 2.93 mm | 0.58 |
| middle phalanx | 301 | [0.248, 0.389, 0.363] | 222 | 5.00 mm | 0.62 |

Both sit on the dorsum, about 60% of the way along their phalanx. The height is bounded to [3, 5] mm (radius 3 mm
plus up to 2 mm of base); the proximal marker sits at the lower bound and the middle one at the upper.

**Calibration take posture:** the index mean pose in the fixture is within 4.8° of flat on every axis
(std ≤ 2.8°).

### 5. Ablation: betas prior weight (revision 1, kept for reference)

| `fit.weights.betas` | free tip | free float prox / mid | calibration float mid | max \|β\| |
|---|---|---|---|---|
| 1.0 | 0.71 | 2.09 / 2.06 | 0.67 | 0.05 |
| **0.1 (chosen)** | 0.70 | 2.04 / 1.98 | 0.35 | 0.38 |
| 0.01 | 0.67 | 1.92 / 1.92 | 0.72 | 1.42 |

These are revision 1 fit-frame RMS values. The hand-scan comparison (`notes/2026-10-22_scan_vs_mocap_mano.md`) was
also run on revision 1 and has not been redone.

## Deviations from the plan, and why

1. **Local naming was added to MoSca** (`mosca/run.py:name_local`).
   * In the calibration take, fixture + caps + floats form ONE rigid group, which `scaffold.name_groups` names as
     the fixture.
   * Motive swaps the knuckle cap's labels inside one physical track, so `seed_from_motive` cannot rescue it.
   * `name_local` runs `shape.place` on the observations around each body's Motive *position*. Knuckle coverage went
     from 3 to 5 markers on the median frame.
2. **The MANO root is the knuckle cap in both takes**, not the fixture.
   * The fixture CAD now enters as the designed hand placement: the binding start, plus a soft prior (5 mm / 5°).
   * The fixture → CAD registration is held fixed.
3. **The fixture → CAD fit is plane-constrained on the asset's defined marker positions** (user decision,
   revision 2). The unconstrained fit of revision 1 was ambiguous (1.23–1.8 mm basins) and wrong: it put markers on
   walls and the underside.
4. **The calibration take gets one index pose per frame**, each pulled to the flat hand. The hand moves about 1–3°
   in the fixture.
5. **The flexion start is unwrapped** (`hand_model.init_index_pose`). A full fist turns the tip cap about 190°, past
   where a rotation vector wraps. Before the fix this caused a 0.55 s tracker run at 13 mm error.
6. **Stage D keeps its best phase end, and resets the floating-marker Adam moments on every re-attach.**
   * A later phase can raise the objective. In revision 2 the polish raised the float term 0.53 → 1.10 mm², so the
     second `betas` phase is kept.

## Known limitations

* **Knuckle cap as a rigid root.** It sits on the hand back, so skin slide there is invisible to the model.
* **Floating-marker heights sit at their bounds** (2.93 at 3.0, 5.00 at 5.0 mm). MANO's dorsal skin over the
  phalanges does not match this hand exactly. The remaining floating error is largest with the finger extended
  (about 1.2 mm).
* **Hyperextension limit.** PIP touches its −10° limit on some frames.
* `Take 2026-10-17 05.39.06 PM.tak` has no CSV export and is not used.

## Outputs

| file | what |
|---|---|
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/outputs/calibration/hand_calib.json` | the calibration (betas, bindings, floating correspondences, fit history) |
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/outputs/calibration/init_rig.json` | Stage C: slip, `T_cad_from_canonical`, plane-constrained fixture registration, starts |
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/outputs/<take>/mosca.npz` | labelled markers, re-solved and Motive poses, floating tracks |
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/outputs/<take>/mano_track.npz` | per-frame index pose, predicted markers, world joints, `T_world_from_canonical` |
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/2026-10-22_calibration_eval.json` | every number in Results §3 |
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/2026-10-22_rb_flips.json` | every number in Results §1 |
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_calibration.html` | 3D viewer, calibration take |
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_free_motion.html` | 3D viewer, free motion |
| `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/fixture_registration.html` | fixture markers on the CAD, all candidate registrations, designed vs fitted hand |
