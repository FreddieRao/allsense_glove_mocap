# Designed-hand prior strength, and rigid-body smoothing (2026-10-22)

## Question

1. Should the designed-hand prior be 1 mm / 1° or a relaxed 2 mm / 2°?
2. On the better of the two, how do the two pose smoothers compare: (1) a zero-phase Butterworth and (2) a
   Kalman / RTS smoother?

## Common setup

Every variant shares this setup:

* the top/side fixture → CAD registration: gap RMS 0.20 mm, held fixed;
* index MCP/PIP/DIP pivot corrections inside a 10 mm ball (`fit.joint_offset_mm: 10`);
* betas prior weight 0.1;
* the same Stage D schedule.

## Re-run

```bash
PY=~/miniforge3/envs/far_grasp/bin/python
P="fit.fixture_sigma_t_mm=2.0 fit.fixture_sigma_r_deg=2.0 fit.weights.fixture=1.0"
for m in none butter rts; do
  T=prior2w1; [ $m != none ] && T=prior2w1_$m
  O="run.tag=$T smooth.method=$m $P"
  $PY -m calibration.fit_hand $O && $PY -m calibration.track $O && $PY -m evaluation.fit_error $O && $PY -m vis.build_viewer $O
done
$PY -m scripts.compare_smoothing      # rigid-body jitter / fidelity -> notes/2026-10-22_smoothing_rb.json
```

The current default (1 mm / 1°, weight 10) is the untagged run. The "2 mm / 2°, weight 10" row is `run.tag=prior2`
with the same two sigmas as above.

## 1. Prior strength

The prior is `fixture weight × ((dt / σ_t)² + (dr / σ_r)²)`, where dt and dr are the hand's offset from the
designed placement through the fixture. RMS is in mm (tip cap / floating proximal / floating middle):

| prior | calibration | free motion, all | free motion, held-out | hand vs design | MCP pivot correction |
|---|---|---|---|---|---|
| 1 mm / 1°, weight 10 (current) | 0.31 / 0.48 / 0.32 | 0.59 / 1.94 / 1.21 | 0.55 / 1.43 / 1.13 | 0.05 mm / 0.03° | [5.5, 6.7, 2.4] mm |
| 2 mm / 2°, weight 10 | 0.31 / 0.48 / 0.33 | 0.59 / 1.93 / 1.21 | 0.55 / 1.42 / 1.13 | 0.07 mm / 0.05° | [5.5, 6.6, 2.4] mm |
| **2 mm / 2°, weight 1** | 0.31 / 0.49 / 0.35 | **0.58 / 1.86 / 1.20** | **0.54 / 1.37 / 1.11** | 0.62 mm / 0.33° | [5.3, 6.2, 2.4] mm |

* **The weight matters, the sigma doesn't.** At weight 10, 2 mm / 2° still pins the hand: 1 mm off costs 2.5 mm²,
  more than the whole data term.
* **Relaxed, the hand still stays close to the design:** about 0.6 mm / 0.3° off, with a slightly better fit.
  With the pivot correction allowed, the data and the designed placement agree.
* **The posture caveat holds in all three:** PIP reaches about 107° while DIP stays between −11 and 18°.
  Revision 2, with a free root and no pivot correction, had DIP up to 78°. This looks like the pivot correction
  trading DIP flexion into PIP. Not resolved.

## 2. Smoothing, on 2 mm / 2°, weight 1

Both smoothers run at 12 Hz. Rigid-body jitter is measured against a 50 ms moving average. Fidelity is the
distance from the smoothed pose to the accepted markers (`scripts/compare_smoothing.py`).

| free motion | none | (1) Butterworth | (2) RTS |
|---|---|---|---|
| tip cap rotation jitter, RMS | 0.66° | 0.23° | **0.20°** |
| knuckle cap rotation jitter, RMS | 0.37° | 0.086° | **0.083°** |
| tip cap fidelity, p50 / p95 | 0.13 / 0.34 mm | 0.17 / 0.44 mm | 0.17 / 0.51 mm |
| knuckle cap fidelity, p50 / p95 | 0.13 / 0.32 mm | 0.14 / 0.33 mm | 0.13 / 0.34 mm |

End to end through fit, track and evaluation. RMS is in mm (tip / prox / mid); output jitter is about a 54 ms
moving average:

| | calibration | free motion | held-out | index tip vertex jitter | flexion jitter MCP / PIP / DIP |
|---|---|---|---|---|---|
| none | 0.31 / 0.49 / 0.35 | 0.58 / 1.86 / 1.20 | 0.54 / 1.37 / 1.11 | 0.276 mm | 0.31 / 0.19 / 0.18° |
| (1) Butterworth | 0.30 / 0.47 / 0.33 | 0.58 / 1.85 / 1.21 | 0.54 / 1.37 / 1.11 | 0.258 mm | **0.115** / 0.180 / 0.122° |
| (2) RTS | 0.30 / 0.47 / 0.32 | 0.58 / 1.84 / 1.21 | 0.54 / 1.36 / 1.12 | 0.258 mm | 0.119 / 0.181 / **0.121°** |

* **Fit error:** neither smoother hurts it, and both lower the calibration-take error a little.
* **Where the gain is:** MCP flexion jitter drops 2.7× and DIP 1.5×. Smoothing the knuckle cap (the MANO root)
  removes root noise from the MCP angle.
* **What doesn't change:** the index-tip jitter barely moves (0.276 → 0.258 mm). The distal bone is driven by the
  tip-cap MARKERS, which neither smoother touches. Smoothing the tip-cap markers, or raising `track.smooth`, would
  be the next step for the tip.
* **Butterworth vs RTS:** RTS is marginally smoother at the rigid-body level, and the two are indistinguishable
  after MANO. RTS also bridges short gaps and trusts 3-marker frames less. Butterworth is simpler.

## Renders

* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_free_motion_prior2w1_rts.html`
* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_free_motion_prior2w1_butter.html`
* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_free_motion_prior2w1.html`
* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_calibration_prior2w1_rts.html`
* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/fixture_registration_prior2w1_rts.html`
  (fitted vs designed hand: median vertex distance 1.4 mm)

## 3. Smoothing the RAW MARKERS instead of the pose (user decision, 2026-10-22)

`smooth.target: markers` (`mosca/smooth.py:smooth_marker_tracks`):

* Every raw marker track is smoothed on its own: cap markers by MoSca (body, row), floating markers by label (after
  the MANO re-claim in Stage E). Smoothed values exist only on frames where the marker was observed.
* The cap poses are then re-solved by Kabsch from the smoothed markers.
* The MANO fit and tracker see smoothed tip-cap and floating markers. Errors are still measured against the RAW
  markers (`tip_obs_raw` / `float_obs_raw` in `mano_track_*.npz`).

**Hyper-parameters, the same 12 Hz for both:**

* Butterworth: 4th order, `filtfilt` (zero phase), 12 Hz cutoff.
* RTS: constant velocity, marker noise σ = 0.1 mm, white-acceleration intensity q = σ² (2π · 12 Hz)⁴ · dt.

Prior 2 mm / 2°, weight 1. RMS is in mm against the RAW markers (tip / prox / mid); output jitter is about a
54 ms moving average:

| smoothing | calibration | free motion | held-out | index tip vertex jitter | flexion jitter MCP / PIP / DIP |
|---|---|---|---|---|---|
| none | 0.31 / 0.49 / 0.35 | 0.58 / 1.86 / 1.20 | 0.54 / 1.37 / 1.11 | 0.276 mm | 0.308 / 0.191 / 0.176° |
| pose Butterworth | 0.30 / 0.47 / 0.33 | 0.58 / 1.85 / 1.21 | 0.54 / 1.37 / 1.11 | 0.258 mm | 0.115 / 0.180 / 0.122° |
| pose RTS | 0.30 / 0.47 / 0.32 | 0.58 / 1.84 / 1.21 | 0.54 / 1.36 / 1.12 | 0.258 mm | 0.119 / 0.181 / 0.121° |
| marker Butterworth | 0.31 / 0.48 / 0.33 | 0.59 / 1.86 / 1.21 | 0.55 / 1.36 / 1.11 | 0.214 mm | 0.108 / 0.153 / 0.113° |
| **marker RTS** | 0.30 / 0.48 / 0.33 | 0.60 / 1.86 / 1.21 | 0.56 / 1.36 / 1.12 | **0.198 mm** | **0.102 / 0.148 / 0.109°** |

* **Marker RTS is the smoothest output.** Index-tip jitter is −28% (pose smoothing gave −7%), and MCP / PIP / DIP
  flexion jitter is 3.0× / 1.3× / 1.6× lower.
* **Error against the raw markers is unchanged within 0.02 mm.**
* **Why marker smoothing helps the tip:** the distal bone is driven by the tip-cap markers, which only marker
  smoothing touches.

Re-run (as in §2, with `smooth.target=markers`):

```bash
O="run.tag=prior2w1_mrts smooth.method=rts smooth.target=markers fit.fixture_sigma_t_mm=2.0 fit.fixture_sigma_r_deg=2.0 fit.weights.fixture=1.0"
$PY -m calibration.fit_hand $O && $PY -m calibration.track $O && $PY -m evaluation.fit_error $O && $PY -m vis.build_viewer $O
```

Renders:

* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_free_motion_prior2w1_mrts.html`
* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_free_motion_prior2w1_mbutter.html`
* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/figures/hand_calib_calibration_prior2w1_mrts.html`

## 4. Defaults, and the optional smoothed wrist (2026-10-22)

**Defaults** (`conf/calibration.yaml`, the untagged run, re-run with `bash run_all.sh --skip-mosca`):

* `smooth.method: rts`, `smooth.target: markers`
* fixture prior 2 mm / 2° at weight 1

Only the default viewers are kept in `notes/figures/`.

**Optional smoothed wrist** (`smooth.wrist`, `calibration/track.py:smooth_wrist`):

* There is no wrist marker set, so the MANO root is the knuckle cap through its binding.
* The fitted root is smoothed AT THE WRIST JOINT: world wrist position plus root rotation, RTS at 4 Hz.
* The knuckle-cap pose is re-derived from the smoothed root, and the index is re-tracked against it with the same
  calibration (`run.tag=wristsmooth run.calib_tag=default smooth.wrist.enable=true`, in `run_all.sh`).
* It runs in about 5 min for both takes.

| free motion | default | smoothed wrist |
|---|---|---|
| wrist joint jitter (RMS about 54 ms) | 0.100 mm | **0.025 mm** |
| root rotation jitter | 0.093° | **0.022°** |
| index MCP joint jitter | – | 0.026 mm |
| tip cap / float prox / float mid RMS | 0.60 / 1.86 / 1.21 mm | 0.60 / 1.86 / 1.21 mm |
| **knuckle-cap RMS (cap where the tracked root puts it)** | 0.18 mm | 0.24 mm (calibration take 0.22 mm) |

Smoothing the wrist costs 0.06 mm at the knuckle cap and nothing on the finger markers.

**Viewers.** Both hand viewers now show:

* the knuckle-cap error: predicted cap markers from the MANO root as blue rings, residual lines, a per-frame value,
  and a timeline row at 1 mm full scale;
* an optional blue overlay of the smoothed-wrist hand, with its own knuckle-cap error (`#ovl` in the URL turns it
  on).

## 5. Distal-bone direction term (2026-10-27)

**The constraint.** `distal_dir` (`calibration/hand_model.py:distal_dir_deg2`; Stage C stores `distal_dir_tipcap` in
`init_rig.json`):

* The designed hand's distal bone (DIP joint → tip vertex of `hand_aligned_mm.obj`) is expressed in the tip-cap frame
  through the certain fixture registration: [0.888, −0.074, −0.454].
* In every frame of both takes, Stage D and E penalise the squared angle to `R_world_from_tipcap @ distal_dir_tipcap`
  (weight 0.05 per deg², `fit.weights.distal_dir`, `track.distal_dir_weight`).
* The scan's long axis (`distal_dir.source: scan`) is weakly defined and sits 7° off the design, so `design` is the
  default.

Default run, RMS (tip / prox / mid in mm; distal direction in deg):

| | calibration | free motion | free held-out | distal-direction error, free motion (RMS / p95) | DIP range |
|---|---|---|---|---|---|
| before | 0.31 / 0.48 / 0.34 | 0.60 / 1.86 / 1.21 | 0.56 / 1.36 / 1.12 | 5.26° / 9.21° | −11…20° |
| after | 0.32 / 0.52 / 0.27 | 0.58 / 1.93 / 1.26 | 0.54 / 1.41 / 1.10 | 1.74° / 3.01° | −10…28° |

* **The constraint holds:** the distal bone now follows the cap within about 1.7°.
* **It barely changes the PIP/DIP split:** PIP reaches 103°, DIP 28°. The split is set by the middle phalanx, which
  only the middle floating marker and the ~8.5 mm MCP pivot correction constrain.
