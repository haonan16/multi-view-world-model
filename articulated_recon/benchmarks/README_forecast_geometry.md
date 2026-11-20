# Shared block geometry from generated forecast views

`forecast_geometry.py` is a numerical initializer for the strict single-image
forecast pipeline. It fits **one cuboid shape and a complete SE(3) pose track**
to DA3 depth/cameras and SAM3 object masks on generated RGB. It is not an agent
result, a forecast accuracy measurement, or an executed manipulation trial.

The observed input remains one initial RGB image. The adapter reuses the strict
preparation validator, checks generated pixel/timestamp identity, and preserves
future `source_frame_indices=-1`. No recorded future, sensor depth, evaluator
geometry, contact annotation or simulator enters fitting.

## Matched branches and scale

| DA3 branch | Geometry inputs | Scale treatment |
| --- | --- | --- |
| `source_unposed` | observed initial/source-camera generated future | initial-image UniDepth/DA3 ratio; separate estimated scale per future timestamp |
| `helper_unposed` | the same source stream plus generated helper views | initial-image ratio; separate estimated scale per synchronized view group |
| `helper_requested_cameras` | generated multi-view RGB with model-designed camera requests | DA3 depth/camera baseline scale retained unchanged |

DA3 unposed inference operates independently at each timestamp. One first-frame
ratio therefore does **not** establish later metric depth. The fitter estimates
later scales jointly with one shared shape and the object poses. The first
scale is fixed by the initial predicted-depth ratio; later scales have an
explicit weak log-scale prior. This uses object rigidity and initial shape as
model assumptions, not new metric observations. `geometry.npz` stores the
estimated scale vector, scaled depth and scaled camera translations. Requested
camera scale remains inherited from the first-image model-designed baseline,
which can itself be wrong. All branches retain `metric_scale.status=estimated`.

The source-only branch requires `forecast-masks/source-only`, whose independent
source-view SAM3 outputs have no generated-helper ancestry. Filtering rows out
of an all-view artifact without preserving the proper lineage is rejected.

## Fit and uncertainty

The objective combines robust point-to-cuboid surface residuals with projected
silhouette coverage and model-outside-mask distances. A single half-extent
vector is shared across all frames. The initial pose is the common first-image
initializer; future translations and rotations are fitted freely. Fixed weak
shape/pose priors regularize ambiguities. Cuboid rotational symmetries remain
unresolved, and no articulated hand is reconstructed by this adapter.

An opt-in `initial_rest_prior` can hold the initial pose through one contiguous
stationary **source-view** prefix. It requires unique sufficiently confident
object masks, stable centroid/area/silhouette relative to the initial mask, a
detected distant hand, and bounded timestamp gaps. Significant hand approach,
image motion, area/silhouette change, missing/ambiguous masks or a time gap ends
the prefix permanently. A missing hand is unknown, not evidence of no contact.
The default maximum centroid shift is 0.1 object diagonals, area change 15%,
minimum initial-mask IoU 0.7, minimum hand gap 0.75 diagonals, maximum hand-gap
drop 0.25 diagonals and maximum sampling interval 0.5 seconds. These are fixed
model-hypothesis thresholds, not calibrated confidence or contact truth.

Anchored poses are removed from the optimizer variables; this is a fit
constraint rather than a post-fit trajectory/plan snap. Later poses remain
free, and missing-depth frames do not gain `prediction_supported` merely from
this prior. The selector reads masks/timestamps only, never depth, fitted
positions, contact outcomes or simulator feedback. The selected frames, exact
measurements and first stop reason are recorded in fit diagnostics and the
scene's forecast metadata. Shared shape is still one global fitted shape.

The option is off unless explicitly added. `default_config()` and previously
saved baseline config keys are unchanged; `config --initial-rest` emits the
extra complete policy block. Existing frozen files are never rewritten. The
same source-mask policy can be used across source/helper conditions.

This variant preserves every supplied DA3 K/c2w estimate. Per-frame focal
estimates can drift even though the physical source camera is assumed fixed.
Replacing them with first-frame K would alter the unprojection/projection
hypothesis and requires a separately declared camera ablation; it is not done
silently by the rest prior. Stationary image evidence itself cannot prove a
unique stationary 3D pose, so this constraint remains an explicit hypothesis.

Object association accepts one distinct sufficiently large SAM3 candidate per
view after removing near-duplicate masks. Ambiguous or missing identities are
unknown. Predicted hand pixels are excluded from object fitting and silhouette
penalties; missing masks are never empty-space observations. Unsupported future
poses carry forward only as explicit placeholders and cannot become supported
merely because the optimizer produced a number.

The initial predicted support plane and `initial_support_prior`, when present,
are copied with initial-image-only scope. The default fit imposes no table
constraint, no stationary-object prior, no grasp, and no hand attachment.
`contacts` stays empty with explicit unknown-contact metadata. Shared geometry
changes can alter initial clearance despite the fixed first pose; the receipt
reports the estimated clearance for every frame and all negative-clearance
indices.

An optional soft nonpenetration penalty can be frozen before execution. It
penalizes negative box-to-initial-plane clearance inside the objective, with no
post-fit plane snap or hard physical guarantee. It may trade silhouette/depth
agreement against a predicted support plane. Report it as a separate prior
variant, together with residual penetration; do not select it using simulator
outcomes.

## Commands

From `articulated_recon`, using a NumPy/SciPy/Pillow runtime:

```bash
python benchmarks/forecast_geometry.py config --output fit-baseline.json
python benchmarks/forecast_geometry.py config \
  --nonpenetration-weight 10 --output fit-soft-plane.json
python benchmarks/forecast_geometry.py config \
  --initial-rest --output fit-rgb-initial-rest.json

python benchmarks/forecast_geometry.py run \
  --condition /path/to/strict-condition \
  --helper /path/to/generated-rgb \
  --predictions /path/to/da3/helper_requested_cameras \
  --masks /path/to/forecast-masks \
  --config-path fit-baseline.json --output /path/to/new-helper-fit

python benchmarks/forecast_geometry.py run \
  --condition /path/to/strict-condition \
  --helper /path/to/generated-rgb \
  --predictions /path/to/da3/source_unposed \
  --masks /path/to/forecast-masks/source-only \
  --config-path fit-baseline.json --output /path/to/new-source-fit
```

The output directory must be new. The branch name comes from the authenticated
DA3 execution receipt. Use the same saved fitter config for matched source and
helper comparisons. No model job or agent inference is launched by this script.

## Outputs and planner interface

- `scene-sequence.json`: shared cuboid, rigid track, first-image support and unknown contacts.
- `geometry.npz`: generated RGB, scaled predicted depths/cameras, masks, observed-image flags and scale vector.
- `input-manifest.json`: forecast ancestry and exact output hashes.
- `execution.json` and `diagnostics.json`: `status=complete`, `scene_sha256`, optimizer status, scale/clearance diagnostics, timings and source-agreement metrics.
- `overlays/`: fitted block on every generated view; `mask-overlays/`: selected object/hand masks.

The object has `observed=[true,false,...]` and
`prediction_supported=[false,<supported future estimates>...]`. Its
`prediction_provenance` explicitly declares `origin=model_prediction`,
`generated=true`, the active generated RGB/depth/camera/mask artifact IDs and
the estimation method. A completed numerical fit can still contain unsupported
future frames; planner readiness remains a separate decision.

Per-frame/view diagnostics include point counts, mask rows, source silhouette
and depth agreement, hand-to-object centroid distance, point-distance 10th
percentile, minimum signed hand-to-box distance and image mask gap. Missing
evidence yields null distances. These are model-prediction proximity measures,
**not contact events**. A later agent must inspect actual generated motion and
create any contact hypothesis separately with evidence and uncertainty.

Tests exercise free rigid lift/rotation, changing unposed scales, unknown masks
and occlusion, optional soft-plane/rest fitting, motion/approach/gap termination,
depth/focal drift with unchanged camera inputs, and the real strict producer/consumer
contract for all three branches:

```bash
python -m pytest -q mano/tests/test_forecast_geometry.py
```

The generated test fixtures establish software behavior only. Recorded-case
quality, forecast accuracy, contact validity and execution success remain
unmeasured until their respective stages actually run.
