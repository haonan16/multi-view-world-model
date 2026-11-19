# SAM3D aspect-ratio ablation

`source_aspect_fit.py` reads the fresh RGB-derived canonical SAM3D mesh. Its XYZ
bounding-box extents are divided by their geometric mean; only these proportions
enter reconstruction. One positive metric size is fitted jointly with all object
translations. SAM3D's MoGe scale, camera-space mesh, and camera pose are excluded
from the method's active input lineage.

The source initializer's table/PCA orientation is retained: canonical Z maps to
the estimated table normal and canonical XY to the inherited tabletop basis. This
is an aspect-constrained cuboid experiment, not full SAM3D mesh tracking. Neither
shape proportions nor metric size are known ground truth.

The comparison preserves source RGB, masks, frame-zero depth calibration, loss
weights, global nonpenetration, contact phase estimator, fixed deployment, and
planner guards. Raw inputs and earlier reconstructions remain unchanged. The
new code was run from an isolated `code-variants/source-geometry-v2` directory;
no frozen producer files were replaced. The experiment did not access simulator
outputs or evaluation geometry.

## Observed source-consistency tradeoff

For the 80-frame public RH20T block sequence, the actual canonical mesh ratios
were `[1.00125, 0.99632, 1.00244]`. The image-derived shape prior prevents the free
fit from flattening its cuboid, while worsening agreement with predicted depth:

| Policy | Shape before → after (mm) | Depth surface error before → after (mm) | Bbox error before → after (px) | Support residual before → after (mm) |
|---|---|---|---|---|
| Within-object rank | 51.2×39.8×35.7 → 42.2×42.0×42.2 | 2.92 → 3.41 | 0.307 → 0.396 | 0.835 → 0.187 |
| Soft rank weights | 53.2×39.6×33.8 → 42.0×41.8×42.1 | 3.00 → 3.95 | 0.314 → 0.375 | 1.355 → 0.040 |

Both fits retained all 80 observed frames, remained above the estimated table,
and produced TCP plans through the unchanged guards. Optimization took 1.54 and
1.50 seconds, excluding SAM3D generation and other pipeline stages. This does not
establish better true geometry, valid Panda IK, grasp quality, or execution
success. All physical evaluation remains separate and was hidden from this fit.

## Reproduction

Write a configuration before running:

```bash
python benchmarks/source_aspect_fit.py config \
  --point-policy object_rank50 --output aspect-config.json
```

Run with the static-calibrated initializer, fresh SAM3D asset bundle, and matched
free-aspect nonpenetrating comparison. A new output directory is required:

```bash
python benchmarks/source_aspect_fit.py run \
  --initializer-bundle /path/to/static-depth/initializer \
  --asset-bundle /path/to/fresh-sam3d-block \
  --comparison-bundle /path/to/nonpenetrating/object_rank50 \
  --config-path aspect-config.json --output /path/to/new-result
```

The producer verifies RGB/mask identity, first-frame object association, source
artifact hashes, and identical comparison loss settings. It exports the actual
scene, active lineage, configuration, dimensions, source residual families,
optimizer diagnostics, and feasible initialization changes.
