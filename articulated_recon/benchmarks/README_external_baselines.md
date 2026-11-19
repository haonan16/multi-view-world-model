# External RGB-only baselines

`external_baselines.py` provides launch preparation and artifact checking for two
external methods. It does not download models, install their environments, run a
robot policy, or claim that either method has been reproduced on our clips.

| Method | Pinned source | Input | Native output |
| --- | --- | --- | --- |
| [SceneRig](https://github.com/Red-Fairy/SceneRig) | `2defb4b112d5abce28ab1f2bea8aac014b31533e` | First observed RGB frame | Blender scene, object meshes and physical-settling records |
| [OVOW](https://github.com/SparcAI-Inc/OVOW) | `0078fa2bab641f6dd334b80381fced05a273bb10` | Entire observed monocular RGB clip | Animated mesh scene and comparison videos |

Heads and public entry points were checked on November 19, 2026. This is an
asymmetric input comparison: SceneRig's first-frame reconstruction does not
recover the demonstrated motion. Score its initial geometry against our
first-frame result. OVOW is the sequence reconstruction comparison. A shared
frozen robot planner still needs a native-output conversion for both baselines;
the adapter does not manufacture a trajectory from a static reconstruction.

## Input boundary

Each launch references the shared `artirecon.input_protocol` manifest with
`condition=rgb_only`. The selected artifact must be `kind=rgb`,
`origin=observed_rgb`, `generated=false`, declare its time range, be listed in
`method_inputs`, and have a verified file hash. SceneRig accepts an extracted
image with a single timestamp; OVOW accepts the observed video with a nonzero
time span. The command builder has no depth, camera, CAD, scene-initialization
or arbitrary-argument fields. Unknown fields are rejected.

The main method's predicted-depth ablations remain separate benchmark variants.
For these external baselines, let the released method estimate its own geometry.
Do not pass the evaluation scene or sensor depth through optional upstream
arguments. The adapter verifies declared input provenance and its arguments; it
is **not an operating-system sandbox**. External code/configuration must still
be reviewed. OVOW sources its checkout's `.env` during execution.

## Environment preparation

Use cluster storage and reuse existing model weights where the upstream software
supports them. Do not install these environments on the laptop by default.

- SceneRig's [setup](https://github.com/Red-Fairy/SceneRig#setup) describes roughly
  150 GB for environments/weights, Linux, NVIDIA CUDA and Blender/Isaac. SAM3 and
  SAM3D access is separately gated. Its agent uses API credits. We call the
  Python runner directly so its shell wrapper cannot evaluate key exports from
  a user's shell configuration.
- OVOW's [setup](https://github.com/SparcAI-Inc/OVOW#reproduce-from-zero-to-results)
  describes roughly 120 GB, several environments and a recommended 40 GB GPU.
  Use its localization/setup procedure and review its research-use component
  licenses. Local Qwen is supported by the upstream configuration, avoiding an
  obligatory remote labeling call.

Check out the pinned commits before environment setup. OVOW's path localization
can modify tracked files. If those changes are reviewed, explicitly set
`allow_localized_checkout=true`; the adapter records a digest of the tracked
diff. Otherwise a dirty external checkout fails preflight.

## Launch request

Create a JSON request alongside your input manifest. Paths are relative to the
request file, except `required_paths`, which are relative to the external
checkout. No API-key values belong in this file.

```json
{
  "schema_version": 1,
  "baseline": "scenerig",
  "checkout": "/path/to/SceneRig",
  "input_manifest": "input-manifest.json",
  "rgb_artifact_id": "source_first_frame",
  "input_scope": "first_frame",
  "output_dir": "/path/to/new-run/scenerig",
  "python": "/path/to/SceneRig/.venv/bin/python",
  "gpu": "0",
  "timeout_seconds": 1800,
  "minimum_free_gb": 20,
  "required_paths": [
    "lib/utils/third_party/sam3/checkpoints/sam3.pt",
    "lib/utils/third_party/sam3d/checkpoints/hf/pipeline.yaml"
  ]
}
```

For OVOW use `baseline=ovow`, `input_scope=full_clip`, the video artifact ID and
an OVOW checkout. Omit `python`, `blender` and `model`. Set `required_paths` to
the actual installed runtime/checkpoint locations for that checkout. At least
one dependency path must be declared, and all declared paths must exist.
These presence checks do not replace the upstream smoke test or validate every
weight file. The preflight records that limitation instead of claiming a fully
validated installation. It also checks the pinned commit, Linux, the requested
GPU, output-directory freshness and free disk.

```bash
# Default is read-only launch planning: no inference, API use or data upload.
python articulated_recon/benchmarks/external_baselines.py request.json \
  --report /path/to/preflight.json

# Run only after the external environment and budget are ready.
python articulated_recon/benchmarks/external_baselines.py request.json --execute
```

No automatic retries are made. The output directory is created exclusively;
reuse is rejected. Each run writes a command/provenance receipt, process log,
wall-clock duration and hashes of required native artifacts. Timeout kills the
run's process group. A zero exit status without the expected nonempty artifacts
is a failure. `complete` means the native artifact contract passed, **not** that
the geometry is accurate, motion is physically valid, or a robot task succeeded.

SceneRig can use its own reconstructed simulator to refine scene geometry. This
is distinct from giving it the independent evaluator's scene. Record this
difference when comparing its agent budget to a method with no reconstruction
simulation. Our downstream robot plan remains frozen before independent
execution, with no evaluator feedback sent back for reconstruction or planning.
