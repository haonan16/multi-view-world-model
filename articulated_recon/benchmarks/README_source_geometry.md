# Source RGB block initializer

This baseline consumes new UniDepthV2 depth/intrinsics and SAM3 masks from the
same isolated source RGB sequence. It uses no measured depth, dataset camera
calibration, cached object mesh or simulator. It is **a fresh shared cuboid and
translation approximation**, not SAM3D or full articulated reconstruction.

The two fixed confidence variants are `all` and `rank50`. `rank50` retains
pixels with within-frame inverse predicted-error rank at least 0.5. This is a
ranking, not a calibrated probability. Both variants perform exactly the same
RGB-mask association; depth confidence cannot select a different target object.

The initializer:

1. Associates prompted block candidates using detector score, mask size, table
   bounding-region overlap and previous-mask continuity. Close competing scores
   remain ambiguous. Candidate row IDs are never treated as temporal identities.
2. Fits a robust table plane from model-depth pixels inside table masks, excluding
   every hand/block candidate. It assumes a static source camera and orients the
   plane toward that camera. The camera identity is a coordinate gauge, not a
   measured camera-to-robot transform.
3. Fits one cuboid to first-frame visible points. Its bottom is placed on the
   **predicted** table under an explicit initial-support hypothesis. This is a
   task/algorithm assumption, not an inferred contact or supplied 3D measurement.
4. Keeps that geometry and orientation fixed. Object translations follow changes
   in the median visible depth surface. Later frames are never lifted back onto
   the table. Depth/visibility bias can therefore produce below-table tracks,
   which the diagnostics report rather than hide.
5. Records estimated support/touch intervals and unknown contact intervals.
   `touch_only` is the default. The optional `proximity_motion_grasp` policy adds
   hypothesized grasps only when adjacent frames show proximity and similar hand
   and object displacement. Missing evidence splits events; no continuous grasp
   is invented to make the planner run.

Missing/ambiguous object frames retain a placeholder pose with `observed=false`.
They cannot become planning evidence. A missing or ambiguous first frame rejects
the initializer because no shared shape is available. This code is block-only;
it must not be presented as reconstructing a bottle body or nozzle.

From `articulated_recon`:

```bash
python benchmarks/source_geometry.py config --confidence-mode all \
  --output /path/all-config.json
python benchmarks/source_geometry.py config --confidence-mode rank50 \
  --output /path/rank50-config.json
python benchmarks/source_geometry.py run \
  --depth-bundle /path/unidepth-output \
  --mask-bundle /path/sam3-output \
  --config-path /path/all-config.json --output /path/new-initializer-output
```

`scene-sequence.json` stores the shared cuboid, estimated table, actual timestamps,
fixed-orientation trajectory, observations and contact hypotheses in
`source_camera` coordinates. `input-manifest.json` binds the scene to the exact
RGB/depth/camera/mask artifacts, retaining unresolved robot registration.
`diagnostics.json` includes mask alternatives, per-frame camera-Z dispersion,
depth point counts, table penetration, timing and limitations. An initializer
failure writes a rejected diagnostic instead of an executable plan.

The default configurations must be fixed before comparing output metrics.
Changes after an initial run require separately named variants and reporting all
results. Synthetic unit tests validate array behavior and lineage; they do not
measure reconstruction accuracy on the real demonstration.
