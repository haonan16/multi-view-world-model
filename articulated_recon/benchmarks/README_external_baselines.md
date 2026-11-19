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

## Cluster readiness audit — November 19, 2026

**Neither baseline has been reproduced or is ready to launch.** This was a
read-only check of the pinned upstream documentation, known project roots,
named model caches and installed-package metadata. No environments, models,
containers or datasets were downloaded; no GPU or paid inference was started.
"Not found" below means not found in those bounded locations, not a global
search of the shared filesystem.

| Component | What was actually found | Consequence |
| --- | --- | --- |
| SceneRig / OVOW checkouts and native environments | Neither checkout or named environment in the checked roots | Installation and upstream smoke tests remain necessary |
| SAM3 | Two caches of the same Transformers `model.safetensors` revision, 3.44 GB each | Neither is the native `sam3.pt` expected by these releases; compatibility/conversion is not established |
| SAM3D | Native checkpoint directory and torch 2.5.1+cu121 environment | Candidate reuse for SceneRig; its vendored code/environment still needs validation. Two zero-byte encoder files prevent a blanket completeness claim |
| MoGe | Native MoGe-v1 `model.pt`, 1.257 GB; MoGe-2-normal `model.pt`, 1.324 GB | Candidate native reuse for OVOW and SceneRig respectively; the latter exactly matches SceneRig's declared model ID |
| Qwen | Qwen2.5-VL and unversioned Qwen-Image-Edit caches | Neither proves availability of OVOW's Qwen3-VL-8B or SceneRig's Qwen-Image-Edit-2509; do not substitute them silently |
| OVOW reconstruction and deformation | No OnePoseViaGen/Hi3DGen, FoundationPose, SpaTrackerV2, RoMaV2, Motion324 or Hunyuan3D-2 caches found | Most native baseline dependencies remain missing |
| SceneRig remaining backends | No MolmoPoint-8B, SHARP, LanPaint runtime or Isaac environment found | Substantial installation remains |
| Runtime | Login node glibc 2.28; no Blender on PATH; Singularity exists | SceneRig's Isaac requires glibc ≥2.35. Compute-node/container compatibility is untested; presence of Singularity is not an installed solution |

The existing `test2` torch 2.9.0+cu129 environment is not OVOW's released
three-environment stack (opg 2.4.1+cu121, SAM3 2.7.0+cu118, motion 2.4.1+cu124).
Finding a working SAM3/SAM3D setup for our method does not establish that either
external pipeline runs unchanged.

**Preferred first external run: OVOW, one complete block clip, native default
`uplegacy` backbone and local Qwen3-VL.** It supplies the sequence geometry and
motion comparison, avoids a required remote labeling API, and has a smaller
advertised installation than SceneRig. Do not replace its mesh generator with
our cached SAM3D, hand-author object labels, suppress the hand's deformable
branch, or skip full-chain artifacts and call that the released baseline.
SceneRig remains a later first-frame geometry comparison, not a motion baseline.

Upstream resource figures, not measurements on our data:

- [OVOW README](https://github.com/SparcAI-Inc/OVOW/blob/0078fa2bab641f6dd334b80381fced05a273bb10/README.md):
  approximately 120 GB installed, roughly 55 GB checkpoint downloads,
  30–60 minutes for environment creation, ≥40 GB VRAM recommended, and
  19–22 minutes for its 108-frame 768×768 full-video example. The
  [download script](https://github.com/SparcAI-Inc/OVOW/blob/0078fa2bab641f6dd334b80381fced05a273bb10/download_ckpts.sh)
  lists about 41 GB for Hunyuan3D-2 alone. Qwen3-VL is not downloaded by that
  script, so its additional local footprint must be checked before reserving
  storage. Our clip runtime is unknown; do not scale this figure linearly by
  frame count. No TRELLIS2 add-on is needed for the native default backbone.
- [SceneRig README](https://github.com/Red-Fairy/SceneRig/blob/2defb4b112d5abce28ab1f2bea8aac014b31533e/README.md):
  approximately 150 GB installed and a required Claude agent API even for
  preprocessing. A trustworthy per-scene runtime/API cost was not given in
  the inspected setup documentation. Cached weights can reduce downloads
  only after native compatibility is verified.

The filesystem reported 292.77 GB free at this audit. That is a changing shared
capacity snapshot, not a storage allocation. Exact **additional** disk and
GPU/API budgets remain unresolved; installing both now would be inappropriate.

The portable [OVOW request template](ovow-block.request.template.json) is
intentionally incomplete and cannot run as-is. First provision the pinned
checkout/native dependencies; review localized paths and set native
`QWEN_BACKEND=local`; materialize a source-only video manifest with verified
hashes/timestamps; then replace the template paths. Its 30-minute timeout is a
proposed cap, not a measured ETA. Run the adapter without `--execute` and the
upstream artifact-checking smoke test before launching our block clip. Model
imports, native-checkpoint compatibility, GPU driver and full-chain output
completeness remain untested. No simulator plan can yet be attributed to either
external method.
