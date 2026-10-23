# Hand-scan MANO fit vs the mocap-calibrated MANO (2026-10-22)

## Question

How does the MANO fit to the hand scan (`data/hand_scan`, from `SECOND_PERSON_HAND_MODEL.zip`) compare with the
betas our mocap calibration fitted (`outputs/calibration/hand_calib.json`)? Would the mocap data accept the
scan's shape?

**Same model:** the scan fit uses smplx MANO right, `use_pca=False`, `flat_hand_mean=True`. Our `ManoIndex`
(`mano_clean`) reproduces the scan's `rest_vertices_mm` from its betas to 8e-6 mm, so the numbers below share one
model and one canonical frame. SCAN1 and SCAN2 share one betas vector.

**Assumption:** the scan README says "the second person from hand_20261018". I assumed that is the same subject
as the 2026-10-18 mocap takes, and did not verify it.

## Re-run

```bash
PY=~/miniforge3/envs/far_grasp/bin/python
$PY -m calibration.fit_hand --tag scan_frozen fit.betas_init=data/hand_scan/SCAN2_MANO_PARAMETERS.npz fit.freeze_betas=true
$PY -m calibration.fit_hand --tag scan_init   fit.betas_init=data/hand_scan/SCAN2_MANO_PARAMETERS.npz
$PY -m scripts.compare_scan_mano          # -> notes/2026-10-22_scan_vs_mocap_mano.json
```

## Results

### Betas

| | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 | ‖β‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| custom_pose (our init) | 0.005 | −0.029 | −0.034 | −0.009 | 0.007 | −0.001 | −0.019 | 0.012 | −0.000 | −0.011 | 0.05 |
| mocap fit (ours) | 0.338 | −0.253 | −0.343 | −0.024 | 0.084 | −0.127 | −0.071 | 0.188 | 0.048 | 0.060 | 0.61 |
| scan fit | −0.563 | −0.095 | −0.700 | −0.121 | −0.379 | −0.371 | −0.250 | 0.194 | −0.329 | 1.461 | 1.86 |

The vectors look very different, but betas are not a good comparison on their own. The geometry below is what
matters.

### Shape at rest (zero pose, mm)

| measure | custom_pose | mocap fit | scan fit | scan − mocap |
|---|---|---|---|---|
| index wrist–MCP | 90.60 | 88.94 | 92.41 | +3.47 |
| index proximal | 33.01 | 32.73 | 34.27 | +1.54 |
| index middle | 22.21 | 22.10 | 23.37 | +1.27 |
| index DIP–tip | 25.54 | 25.49 | 26.82 | +1.33 |
| middle proximal / middle / DIP–tip | 31.74 / 23.29 / 25.11 | 31.39 / 23.14 / 24.99 | 33.12 / 24.51 / 26.30 | +1.7 / +1.4 / +1.3 |
| thumb proximal / middle / IP–tip | 30.79 / 27.10 / 34.72 | 30.43 / 26.87 / 34.46 | 31.89 / 27.97 / 36.18 | +1.5 / +1.1 / +1.7 |
| palm width (index–pinky MCP) | 66.82 | 64.82 | 68.39 | +3.57 |
| hand length (wrist–middle tip) | 173.92 | 171.72 | 179.37 | +7.65 |

The scan hand is about 4–5% larger everywhere. Each index segment is 1.3–1.5 mm longer, and the hand is 7.7 mm
longer overall. Vertex distance between the two rest meshes:

* raw: p50 3.1 mm, max 5.1 mm
* aligned on the palm: p50 3.7 mm, max 6.9 mm
* aligned on the palm, index finger only: p50 5.3 mm

### Index posture

Values are axis-angle components in degrees: z is flexion, y is abduction.

| joint | scan2 (on the scanner) | mocap calibration take (in the fixture) |
|---|---|---|
| MCP | y 12.7, z −8.7 | y 2.0, z 0.8 |
| PIP | z −7.6 | z 0.6 |
| DIP | z −5.8 | z −0.3 |

On the scanner the index is slightly hyperextended and abducted. In the fixture it is flat.

### Does the mocap data accept the scan shape?

This is the Stage D fit-frame RMS in mm, with everything else (bindings, floating skin points, poses) refitted:

| shape | cal tip | cal prox | cal mid | free tip | free prox | free mid | float height prox / mid (mm) |
|---|---|---|---|---|---|---|---|
| mocap fit (ours) | 0.21 | 0.52 | 0.73 | 0.70 | 2.03 | 2.01 | 2.90 / 4.88 |
| scan betas as start + prior | 0.20 | 0.47 | 0.34 | 0.69 | 2.02 | 2.00 | 2.91 / 4.91 |
| **scan betas, held fixed** | 0.20 | 0.47 | 0.33 | 0.70 | 2.08 | 2.07 | 2.90 / 4.92 |

**Yes.** With the scan's shape held fixed, every marker group is within 0.06 mm of our fitted shape. The
calibration take actually fits better (the middle float improves). The mocap data cannot tell a hand 4–5% larger
from ours: it sees only the index finger relative to a cap on the hand back, and the bindings absorb the scale.
The scan measures the whole surface directly.

The calibration middle-float RMS of our fit moves between runs (0.35 mm earlier, 0.73 mm in this re-run). That
term is weakly determined, so a 0.4 mm difference there is not meaningful.

## Recommendation

Use the scan betas, held fixed (`fit.betas_init=data/hand_scan/SCAN2_MANO_PARAMETERS.npz fit.freeze_betas=true`),
as the calibration shape. They cost nothing in mocap fit error and come from a direct surface measurement. The
floating-marker heights are unchanged (2.9 / 4.9 mm), so the residual skin mismatch over the middle phalanx is
not a scale effect.

## Outputs

* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/notes/2026-10-22_scan_vs_mocap_mano.json`: every number above
* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/outputs/calibration/hand_calib_scan_frozen.json`: the calibration with the scan betas frozen
* `/home/ANT.AMAZON.COM/raofuu/Code/allsense_glove/outputs/calibration/hand_calib_scan_init.json`: the calibration with the scan betas as start + prior
