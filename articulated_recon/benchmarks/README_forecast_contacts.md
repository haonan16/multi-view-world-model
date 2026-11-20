# Generated masks, camera audit and contact hypotheses

These adapters consume the strict single-image forecast described in
`README_strict_forecast.md`. They do not consume recorded future RGB, sensor
depth, GT contacts or simulator feedback. CPU fixtures verify their interfaces;
they do not establish real forecasting, reconstruction or execution quality.

## Framewise SAM3 on the generated sequence

```sh
python articulated_recon/benchmarks/forecast_masks.py \
  --helper /path/forecast-output --condition /path/strict-condition \
  --checkpoint /path/cached-facebook-sam3-snapshot \
  --output /path/new-forecast-masks --preflight
```

Remove `--preflight` for inference using the cached checkpoint. Default task
categories are `wooden block on patterned mat`, `hand`, and `tabletop`.
Thresholds default to 0.3 for detections and 0.5 for mask pixels. Every prompt is
processed independently on every frame/view image. This is **framewise
segmentation**, not tracking or automatic object relevance discovery.

`masks.npz` stores packed candidate masks, scores/boxes, prompt indices `p`,
forecast frame indices `t`, view indices `v`, timestamps, source-frame indices,
the observed-image mask and per-image pixel hashes/timings. Candidate row IDs
identify storage rows, not objects over time. Empty detections stay empty.
Only frame 0/view 0 is observed; all remaining RGB images are generated.

`source-only/masks.npz` contains the independently computed view-0 candidates.
Its manifest consumes the generated source stream without the generated-helper
artifact. This permits a matched source-only comparison without rerunning SAM3
or pretending that filtering an all-view reconstruction removes its dependence
on helper views. Products use `forecast_sam3_masks` and
`forecast_sam3_weights`, leaving initial-image mask ancestry intact.

## Initial camera-response diagnostic

```sh
python articulated_recon/benchmarks/forecast_camera_audit.py \
  --helper /path/forecast-output --condition /path/strict-condition \
  --output /path/new-initial-camera-audit
```

The audit uses only t0, initial predicted depth/K and initial hand/object masks.
It compares static-feature displacement against both requested-camera parallax
and copying the source view. Raw decoder pixels must match the completed
generation receipt. `initial-camera-response.png` and `report.json` are internal
consistency diagnostics. Too few matches remain unmeasured; this does not prove
camera correctness, future motion quality or physical feasibility.

## Contact stage after shared-scene fitting

Save the configuration before looking at execution outcomes:

```sh
python articulated_recon/benchmarks/forecast_contacts.py config \
  --output /path/contact-config.json
python articulated_recon/benchmarks/forecast_contacts.py run \
  --scene-bundle /path/forecast-shared-scene \
  --masks /path/forecast-masks/source-only \
  --condition /path/strict-condition \
  --config-path /path/contact-config.json \
  --output /path/new-contact-scene
```

The current contact estimator uses source-view evidence. It requires one unique
object detection and an unambiguous nearest hand candidate. Proximity and motion
thresholds are normalized by object image size. A grasp hypothesis needs
contiguous proximity, hand/object co-motion, estimated lift and recent support.
An interval ending before the prediction horizon requires a supported release.
With the explicit `allow_terminal_hold=true` configuration, an otherwise valid
grasp continuing through the final supported frame is retained as a terminal
hold, labelled `release_unobserved_beyond_horizon`. No release is invented, and
a missing/unsupported final frame cannot be bridged to qualify. Release evidence
combines hand separation, a stationary
mask sequence and the fitted cuboid bottom near the estimated table. Support
windows adapt deterministically to the declared frame cadence; they require
multiple actual predicted samples rather than interpolated mask evidence.

The existing initial support hypothesis may apply at t0 only if the initial
scene explicitly declared it and the fitted bottom remains near the plane. It
is labelled separately and never propagated through missing future evidence.
All contacts remain `hypothesized` with null, uncalibrated confidence. Missing
masks or unsupported predicted poses produce unknown intervals. Multiple
separate grasps remain separate; there is no first-to-last fallback.

The output `scene-sequence.json` preserves fitted geometry, timestamps, observed
flags and prediction-support flags. It adds contact hypotheses and a new bound
manifest artifact. `diagnostics.json` reports candidate intervals, rejection
reasons, support evidence and assumptions. When an all-view fit already contains
another `forecast_sam3_masks` artifact, the independent source subset is aliased
to `forecast_contact_source_masks` in the merged graph. Original mask files and
manifests remain unchanged; both hashes are retained.

The agent should inspect the generated movie, the fitter's `mask-overlays/` and
`overlays/` images, and these diagnostics before interpreting a phase as a pick
rather than a push. This visual review does not turn predictions into measured
contacts. A numerical hypothesis does not prove hand identity, force closure or
a stable robot grasp. Registered TCP planning may still reject geometry,
unsupported motion, split grasps or insufficient clearance. No scene geometry
is changed in this contact stage to make a plan pass.
