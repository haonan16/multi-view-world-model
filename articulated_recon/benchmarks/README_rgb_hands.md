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

## Fresh object mesh tracking

`source_mesh_track.py` consumes that verified hand result plus fresh SAM3 masks,
SAM3D mesh and the identical learned source geometry. It authenticates the
upstream hand's original manifest and output hashes before merging downstream
ancestry. RGB, depth, cameras and native timestamps must match exactly.

The first pass fits 13 uniformly sampled native frames with Amani's existing
mask association and silhouette/depth pose objective. `--association rgb_identity`
instead associates unique SAM3 candidates through every native RGB frame using
image continuity, then fits the same keyframes. Ambiguous detections remain
unknown. A valid image mask survives a missing or inconsistent depth centroid.
This variant also excludes predicted hand pixels from object foreground points,
depth evidence and the outside-silhouette penalty; it does not label those pixels
as free space. Both object and hand candidate thresholds are 0.3.

Only the first frame can
initialize/refine the shared mesh shape; moving frames are not combined under
one pose. The SAM3D mesh topology is retained, including the bottle nozzle, with
no primitive fallback. Its initial support placement is an explicit learned
table prior. Subsequent pose fitting has no assumed-rest attraction.

Single-view attachment remains unknown because the existing human attachment
test requires two observed views. No rigid-grasp constraint or executable plan
is asserted; `planning_ready` stays false. A fit can complete while contact and
target-robot registration remain unresolved.

```sh
python benchmarks/source_mesh_track.py \
  --geometry runs/bottle/unidepth-v1 --masks runs/bottle/sam3 \
  --asset runs/bottle/sam3d --hand-run runs/rgb-mano \
  --case bottle --output runs/bottle/source-mesh-track \
  --keyframes 13 --shape-iters 250 --pose-iters 150 --samples 1500
```

## Measured bottle diagnostic, 19 November 2026

The source is 120 recorded HO-Cap RGB frames from one camera at 30 fps. All
depth, intrinsics, masks and hand initialization in these runs are model
estimates. Dataset calibration, sensor depth and simulator feedback are not
inputs. These are single-source-view diagnostics, not a validated multiview
reconstruction or execution result.

The handedness-prior variant selected the right hand. After the same
100-iteration port budget, agreement with predicted image landmarks improved
from 2.737 px mean in the earlier left-hand diagnostic to 1.744 px. Source
overlays improved, but the independent display camera still reveals fingers
that look too spread/flat around the bottle. This score does not establish
correct hidden fingers or physical contact.

For the mesh, the same fresh SAM3D asset, right-hand result, 13 fitting frames
and 250/150 shape/pose iteration budgets were reused:

| Source-only measurement | Author 3D association, v1 | RGB identity + hand occlusion, v2 |
| --- | ---: | ---: |
| Fitting frames with object association | 4/13 | 13/13 |
| Visible-mask IoU, all 13 frozen poses | 0.306 | 0.779 |
| Visible-mask IoU, four already-supported frames | 0.774 | 0.795 |
| Visible-mask IoU, nine recovered frames | 0.098 | 0.771 |
| Median agreement with predicted source depth | 38.5 mm | 7.08 mm |
| Object fitting CPU time | 29.8 s | 36.4 s |

The four common-supported native frames are 0, 10, 20 and 119. All-pose scores
include v1's explicitly unobserved, frozen fallback poses. The principal gain
is recovering the lift segment, rather than a large improvement on frames v1
already tracked. The actual SAM3D topology/nozzle is retained in both fits.

This is a **combined variant**, not a clean test of depth bias alone: v1 applies
its fixed 5 cm 3D association gate at the 13 sampled frames, whereas v2 tracks
image identity over all 120 native frames and adds explicit hand occlusion.
All 13 v2 keyframes have usable predicted-depth centroids. Ordinary motion
between sparse frames and depth bias have not yet been separated as causes of
the original gate rejection. The metrics compare against SAM3/UniDepth model
outputs; they do not establish ground-truth 3D accuracy. Contact and robot
registration remain unresolved, and neither fit is marked planning-ready.

`mesh_track_source_metrics.py` computes this common reference comparison.
`preview_source_mesh.py` renders recorded RGB, the actual fitted hand/object,
and a display-camera 3D view at the native 30 fps. It labels pose interpolation
between fitting frames, hides/labels missing associations, and identifies the
table footprint as illustrative rather than reconstructed table dimensions.

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

The completed bottle helper run used the correct author-encoded prompt,
"Pick up the pump bottle," and 13 frames of the source motion. Its three helper
views changed the actor's appearance but did not follow the requested camera
parallax: median static-background shifts were 0.37, 0.36 and 0.60 px versus
requested shifts of 60.3, 43.6 and 58.7 px under the learned source geometry.
This rejects treating those requested poses as verified helper calibration
under the tested conditioning. Cold scheduler wall time was 16m14s on an A100
3g.20gb MIG slice; the timed model stage was 363.8s, peak allocation 7.61 GB.
