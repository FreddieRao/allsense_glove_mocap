#!/usr/bin/env bash
# allsense_glove index-finger calibration, stages A-F + viewer. Re-run from the repo root:  bash run_all.sh
# Pass --skip-mosca to reuse outputs/<take>/mosca.npz (Stage B is ~3-7 min per take).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/far_grasp/bin/python}
if [[ "${1:-}" != "--skip-mosca" ]]; then
  $PY -m mosca.run --take calibration free_motion        # A+B: load CSV, MoSca re-identification, RB re-solve
fi
$PY -m calibration.init_rig                              # C: rigid calibration summary + initial bindings
$PY -m calibration.fit_hand                              # D: joint fit (betas, bindings, floating skin points)
$PY -m calibration.track --take calibration free_motion  # E: per-frame index pose with the frozen calibration
$PY -m evaluation.fit_error                              # F: RMS of cap / floating markers -> notes/
# optional variant: the same calibration tracked with the smoothed wrist, overlaid (toggle) in the hand viewers
WS="run.tag=wristsmooth run.calib_tag=default smooth.wrist.enable=true"
$PY -m calibration.track $WS && $PY -m evaluation.fit_error $WS
$PY -m scripts.check_rb_flips                            #    Motive vs MoSca orientation stability -> notes/
$PY -m vis.build_viewer --take calibration free_motion   #    3D HTML -> notes/figures/
$PY -m vis.build_fixture_viewer                         #    fixture asset on the CAD -> notes/figures/
