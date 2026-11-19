# Source-only block reconstruction refinements

These experiments consume one recorded RGB sequence, UniDepth-predicted depth/K,
and source SAM3 masks. They do not consume sensor depth, dataset calibration,
evaluation meshes, or simulator feedback. This is a fixed-orientation shared
cuboid baseline, not full object rotation tracking or SAM3D mesh reconstruction.

## Tested sequence of changes

1. `source_geometry.py`: initializes a shared cuboid and estimates translations
   from visible predicted-depth points. `all` retains points; `rank50` selects the
   upper half of global uncertainty ranks. A rank is not a probability.
2. `source_joint_fit.py`: jointly fits dimensions and all translations using
   source depth, projected masks, temporal smoothness, and soft support inferred
   from stationary RGB object masks with a distant detected hand. Missing hand
   detections are unknown. `object_rank50` ranks within each object, while
   `soft_rank` retains weak points with reduced weights.
3. `source_contact_phases.py`: estimates contiguous grasp hypotheses from
   observed RGB proximity, hand/object co-motion, predicted lift, and independent
   support/release evidence. It does not join old intervals or fill missing masks.
4. `static_depth_calibration.py`: keeps the frame-zero learned metric gauge and
   estimates a multiplicative depth correction per frame from a source-predicted
   table region. Hand/object masks are excluded separately in each frame. The
   static-camera assumption is checked heuristically through RGB stability;
   disjoint background pixels assess plane consistency. Raw predictions remain
   unchanged. A fresh shape is initialized on the SAME frame-zero plane.
5. `source_nonpenetrating_fit.py`: runs the same joint residuals while restricting
   all poses to nonnegative clearance above the one estimated table. Dimensions
   and all translations are optimized together. Initial infeasible clearances are
   projected only to start the constrained optimizer; all such changes are
   reported. There is no post-fit endpoint clamp or planner lifting.

The constrained fit uses the same 0.15 m trust bound in table-tangent coordinates
and clearance rather than world-axis coordinates. It reports that change and
individual depth, silhouette, support, acceleration, and shape-prior costs.
Nonpenetration relative to an estimated plane is not proof of correct geometry.

## Reproducible artifacts

Every run writes a scene, materialized RGB-only input manifest, configuration,
and diagnostics; completed variants are kept. The static correction is a compact
recipe, **not a raw geometry archive**. Generic consumers must explicitly call:

```python
from benchmarks.static_depth_calibration import materialize_recipe
geometry = materialize_recipe(raw_geometry_path, depth_calibration_path)
```

That reader verifies the raw archive hash and timestamps and preserves frame-zero
scale. It also preserves `confidence_available=False`; neutral stored weights
must not be treated as model uncertainty. `source_joint_fit` supports `all`
without rank filtering/weighting and rejects rank policies when confidence is
explicitly unavailable. The initializer similarly rejects `rank50` in that case.

Fixed deployment registration and the unchanged planner are applied after
reconstruction. A rejected plan stays rejected. Neither a generated TCP plan nor
passing estimated-scene checks demonstrates independent physics success.

## Current measured evidence

The 80-frame public RH20T block run lost 17 manipulation frames under global
rank50 filtering. Per-object ranking and soft weights retained all 80 frames.
Joint fitting improved source silhouette agreement, but initially made mean
estimated-table support agreement worse. Static background calibration reduced
mean held-out plane residual from 17.20 to 3.65 mm and reduced mean support error
from 20.41 to 2.97 mm (per-object rank) or 18.72 to 2.54 mm (soft rank). Endpoints
still penetrated the estimated table by 7.69/6.29 mm. Those versions were rejected
by the unchanged planner, before simulation. These are source-consistency
measurements, not ground-truth geometry accuracy or execution success.

The separate global nonpenetration fit then produced two TCP plans through those
same guards. It reduced support residuals to 0.835/1.355 mm but increased depth
surface residuals from 2.38/2.42 to 2.92/3.00 mm and made the inferred cuboids
flatter. Source bbox errors changed from 0.292/0.299 to 0.307/0.314 px. This exposes
a real tradeoff: a physically feasible explanation of biased observations is not
necessarily a better shape. Independent physics, Panda IK, and GT geometry
accuracy remain separate checks; passing the geometric planner is not success.
