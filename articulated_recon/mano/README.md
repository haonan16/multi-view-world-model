# MANO hand reconstruction

Human-hand counterpart to the robot reconstruction code in `../artirecon`.
This package collects the existing MANO implementation and experiment tools for
the RH20T button/block and HO-Cap bottle cases. It makes those tools available
in the shared repository; it does not claim a new benchmark or a completed
robot/hand pipeline unification.

## What is included

- A differentiable MANO surface backend, wrist translation/rotation, finger PCA
  coordinates and one shared hand-shape vector across the sequence.
- The direct adaptation of Amani's `RobotFitter`: multistart initialization,
  forward/backward tracking and joint sequence refinement. The build scripts
  record exact source diffs and hashes and reject unrecognized source changes.
- RGB landmark and hand-evidence constraints, including fingertip evidence and
  visibility handling. The 45-component mode preserves the PCA convention of
  the smaller saved fits.
- The human interaction adapter for `ObjectsV2`, SAM3/SAM3D stage wrappers,
  SAM3 video tracking and object refitting.
- CPU geometry/rendering utilities, regression tests and the existing
  plan-freezing/execution utilities under `actions/`.

`mano_recon/fitting.py` is the original standalone sequence fitter. The direct
Amani port is built as `runtime/amani_comparison/fit_robot_mano.py`; these are
different fitters. The direct port removes the fixed Panda base search and tied
fingers and adds free wrist coordinates and shared MANO shape. It does not fit
robot joints to a human hand.

## Install and licensed assets

Use a separate environment from the world-model training environment:

```bash
cd articulated_recon/mano
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Choose a PyTorch build compatible with your GPU before installing on a cluster.
The package dependency ranges are compatibility declarations, not a frozen
replication environment for the historical GPU runs.

MANO assets are **not distributed here**. Obtain them under your own
[MANO license](https://mano.is.tue.mpg.de/). If you already have our numeric
conversion locally, set `MANO_MODEL_DIR` to the directory containing
`MANO_RIGHT.npz` and `MANO_LEFT.npz`. Otherwise convert your authorized archive:

```bash
python -m pip install -e '.[model-preparation]'
python scripts/prepare_models.py /path/to/mano_v1_2.zip
export MANO_MODEL_DIR="$PWD/assets/mano"
```

Conversion reads the original trusted model pickles; fitting uses numeric NPZ
files with pickle disabled. Do not redistribute model files or commit the
`assets/` directory.

## Build the direct RobotFitter port

The sibling `articulated_recon` checkout provides the upstream code. No archived
experiment code is required to build the runtime:

```bash
python scripts/bootstrap_runtime.py \
  --run /path/to/working-run \
  --models "$MANO_MODEL_DIR"
```

`--upstream /path/to/articulated_recon` overrides the sibling checkout.
`--models` creates a local symlink into the working run, not a copy or upload.
The run contains `upstream/`, generated `runtime/amani_comparison/`, and
provenance records. Input observations and model checkpoints are supplied
separately.

## Prepared observations and stages

The direct pipeline currently accepts a prepared `inputs/<case>/` bundle,
where `<case>` is `button`, `block` or `bottle`:

| File | Contents |
| --- | --- |
| `obs.npz` | `rgb[T,V,H,W,3]`, camera-Z depth in metres as `depth_m[T,V,H,W]`, camera intrinsics `K[V,3,3]`, OpenCV world-from-camera matrices `c2w[V,4,4]` |
| `hand-input.npz` | RGB-derived `initial_q[T,3+P]`, `initial_translation[T,3]`, foreground `points[T,N,3]`, `point_views[T,N]`, valid `counts[T]` |
| `protocol.json` | `fps`, table `normal` and `offset`, plus input/source provenance |
| `ann.json` | Object names, roles, text prompts and a reviewed RGB instance click, using the upstream annotation format |

`prepare_full_mano_pipeline.py` migrates historical prepared detector/triangulation
bundles. It is not a raw-video detector entry point. Provide actual observations
and initialization; the bootstrap does not invent them or retrieve private data.

For the standalone fitter, the separate `mano_recon.observations.Observations`
loader documents its strict NPZ schema; its depth field is `depth`, not
`depth_m`. Do not interchange the two formats without conversion.

Run the direct hand fitter:

```bash
python scripts/run_robotfitter_mano.py \
  --run /path/to/working-run --case block --models "$MANO_MODEL_DIR" --device cuda
```

For the object stages, use the existing upstream SAM3 and SAM3D environments.
Those gated models and their dependencies are not installed by this package.
The SAM3D wrapper expects a directory containing `sam3d_mesh.py` and
`repo/checkpoints/hf/pipeline.yaml`, matching Amani's setup:

```bash
# Run each command with its corresponding model environment's Python.
python scripts/run_sam_comparison_networks.py \
  --run /path/to/working-run --case block --stage sam3
python scripts/run_sam_comparison_networks.py \
  --run /path/to/working-run --case block --stage sam3d \
  --sam3d-root /path/to/sam3d-wrapper
python scripts/run_full_pipeline_case.py \
  --run /path/to/working-run --case block --models "$MANO_MODEL_DIR" --device cuda
```

The final command starts **after** the SAM3/SAM3D stages. It fits the hand,
fits objects, runs the SAM3 video tracker, and refits the tracked objects.
The tracker needs CUDA even when the geometry fitter is run on CPU. Mesh
simplification additionally needs `fast-simplification` in the SAM3D environment.
The runners retain per-stage timing, status, input hashes and generated outputs.

Hand outputs include `hand-results/<case>/fit.npz`, `hand-sequence.npz` and
`inputs/<case>/frozen-hand.npz` for the object stage. Object outputs live in
`results/`. These fitted/keyframe sequences are reconstructions, not physics
rollouts. The `actions/` tools form a separate downstream stage with their own
prepared scene/plan inputs; running reconstruction does not execute a task.

## Input and evaluation boundary

The recorded-data experiments use sensor depth and supplied camera calibration.
The generated-view experiments elsewhere in the project use predicted depth,
but also retain calibration assumptions. **This package is not a demonstrated
strict RGB-only end-to-end system.** GT hand-pose annotations are not required
to initialize the direct MANO fitter. An instance click is a human-provided RGB
annotation, not automatic task-relevance selection.

Depth, masks, calibration and initial poses must be labeled by source when
preparing a bundle. Keep measured-depth/GT-geometry oracles separate from
model-derived inputs. Held-out cameras must be excluded from initialization as
well as optimization. Good render agreement does not establish contact accuracy
or execution success. The frozen-plan utilities reject post-freeze changes;
evaluation scene feedback must not be used to refine those commands.

## Validation

```bash
MANO_MODEL_DIR=/path/to/licensed/numeric-models python -m pytest -q
```

Tests requiring licensed MANO data skip if it is absent. Historical-result
checks also skip when their external bundles are absent. CPU unit/regression
tests and runtime-build checks do not establish that the complete SAM3/SAM3D
GPU pipeline has been rerun on this branch. The original research recordings,
results, environments and presentation media are deliberately excluded.

MANO: Romero, Tzionas and Black, *Embodied Hands: Modeling and Capturing Hands
and Bodies Together*, SIGGRAPH Asia 2017. Loader: [smplx](https://github.com/vchoutas/smplx).
