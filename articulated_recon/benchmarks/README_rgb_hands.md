# Fresh RGB-only hand fitting

This path creates new hand inputs from recorded RGB and model predictions. The
older `mano/scripts/prepare_full_mano_pipeline.py` imports supplied calibration,
depth and earlier triangulation, so it remains a calibrated/RGB-D diagnostic.

The new preparation consumes:

- An isolated `rgb_frontend.py` source bundle and its learned UniDepth geometry.
- All-frame `sam3_rgb_masks.py` candidates for the task object, hand and tabletop.
- Fresh `rgb_hand_landmarks.py` predictions, including missing/multiple hands.
- User-licensed, converted MANO left/right files outside the repository.

It never reads archived `triangulation.npz`, `initialization.npz` or hand fits.
Model weights and datasets are not distributed with this package.

## Stages

1. `rgb_hand_landmarks.py` runs MediaPipe on native RGB plus expanded/rotated SAM3
   hand crops. It saves every distinct candidate, relative predicted 3D joints,
   image landmarks and handedness. Candidate rows are not temporal identities.
2. `rgb_mano_initialize.py` associates a hand using task-object image proximity
   and wrist continuity. Translation comes from learned depth near the detected
   wrist, restricted to predicted hand pixels and excluding object masks.
   Missing or dispersed depth is rejected. Padded optimizer seeds are never
   labeled observed.
3. Both MANO sides are fitted to predicted relative 3D and image landmarks. The
   default `rgb_evidence` policy combines reprojection, relative-3D residual and a
   soft handedness prior. This is a declared heuristic, not calibrated inference.
   `reprojection_only` reproduces the earlier diagnostic that sometimes selected
   the wrong chirality. Ambiguous rankings stop instead of forcing a side.
4. The mechanically ported Amani `RobotFitter` consumes the fresh initialization,
   learned depth, hand foreground points and continuing 2D landmark evidence.
   One hand shape is shared across time. Verified inputs explicitly name their
   handedness and coordinate frame; RGB preparation uses `source_camera`.

The first pass does **not** enforce fingertip/object contact or estimate robot
registration. A good source-view overlay is insufficient evidence of correct
hidden finger geometry.

## Example commands

Run from `articulated_recon/`, with dependencies installed in an isolated runtime.
The tested detector version is MediaPipe `0.10.32`; its hand-landmarker task asset
must be supplied locally. GPU fitting additionally needs the package's existing
Torch/SMPL-X dependencies and licensed MANO files.

```sh
python benchmarks/rgb_hand_landmarks.py \
  --requests detector-requests.json \
  --detector-model /private/models/hand_landmarker.task

python mano/scripts/bootstrap_runtime.py \
  --run runs/rgb-mano --upstream . --models /private/models/mano

python benchmarks/rgb_mano_initialize.py \
  --requests hand-requests.json --run runs/rgb-mano \
  --models /private/models/mano --device cuda

PYTHONPATH=mano python mano/scripts/run_robotfitter_mano.py \
  --run runs/rgb-mano --case bottle --models /private/models/mano \
  --input-manifest runs/rgb-mano/inputs/bottle/input-manifest.json \
  --iterations 100

python benchmarks/preview_rgb_mano.py \
  --run runs/rgb-mano --models /private/models/mano --cases bottle
```

`detector-requests.json` is a list of objects with `bundle`, `masks`, and `output`
directory paths. `hand-requests.json` is a list such as:

```json
[
  {
    "case": "bottle",
    "geometry": "runs/bottle/unidepth-v1",
    "landmarks": "runs/bottle/mediapipe",
    "masks": "runs/bottle/sam3",
    "object_prompt": "pump bottle",
    "side_policy": "rgb_evidence"
  }
]
```

Use a fresh run directory for each changed hypothesis. The 100-iteration example
is a reduced-budget diagnostic; the port's default is 300. Receipts report actual
iterations, visibility, initialization choices and hashes. Preview metrics are
agreement with **predicted** landmarks, not ground-truth hand/contact accuracy.

## Generated-view diagnostics

`da3_helper_views.py` checks that source view0 exactly matches the deterministic
crop/resize of recorded RGB, then evaluates matched timestamps with source-only
DA3, generated-helper DA3 and uncertain requested-camera conditioning. Unposed
DA3 Giant scale is arbitrary and must be anchored explicitly to learned metric
source evidence before metric reconstruction.

`helper_camera_audit.py --object-prompt 'pump bottle'` excludes the exact bottle
category and hands from background correspondences. Camera requests are
conditioning hypotheses, not measured generated-view calibration. Its parallax
test measures internal consistency using learned source depth/K, not GT accuracy.
