# Pi4 RGB-only frontend

`pi4_frontend.py` calls the released `Pi4.from_pretrained(local_safetensors)` and
one joint `model(images, query_frame=...)` forward on an authenticated isolated
RGB bundle. It never reads dataset depth, camera calibration, reference meshes,
or evaluation outputs. It does not train a model or generate helper views.

Official sources (reviewed 2026-11-19):
- [Paper, arXiv 2611.23144](https://arxiv.org/abs/2611.23144)
- [Code](https://github.com/yyfz/Pi4)
- [API and installation](https://github.com/yyfz/Pi4/blob/main/docs/inference.md)
- [Weights](https://huggingface.co/yyfz233/Pi4)

Pinned public release identifiers for the prepared run:
- Code commit `32dfbea9ce6268ea17446208a804710df9475c32`.
- Hugging Face snapshot `59f307317fb630c9c5a5aa635038d98ed106aaf5`.
- Checkpoint `model.safetensors`, 1,995,767,812 bytes (1.86 GiB).
- Checkpoint SHA256 `fefacac1fb3d23d33c628c41f40d3ce3be98ca064c271948371805db7b5a0e69`.

These IDs and checkpoint metadata were verified from the official GitHub and
Hugging Face APIs; no checkpoint was downloaded onto the laptop.

Code and weights use **CC BY-NC 4.0**. Preserve the upstream license/notice and
citation when sharing derivatives. The official installation requires a CUDA
GPU with BF16 support and an isolated environment, with PyTorch 2.12.1, Triton
3.7.1, and FlexGEMM commit `b2fadb29d41846c7981ade6801ffc689fae119cf`.
Do not replace an existing lab environment's dependencies to install this model.
The recommended complete stack is upstream `requirements-cu126.txt`; the
upstream guide specifically warns against mismatched Torch/Triton packages.

## Bounded first test

`pi4_setup.sh` accepts explicit installation root and existing Python 3.11 paths:

```bash
bash articulated_recon/benchmarks/pi4_setup.sh preflight \
  --root /cluster/path/to/new/pi4-root --python /cluster/path/to/python3.11
bash articulated_recon/benchmarks/pi4_setup.sh setup \
  --root /cluster/path/to/new/pi4-root --python /cluster/path/to/python3.11
```

Preflight is read-only and requires Linux x86_64, Python 3.11, Git, a C compiler,
and at least 25 GiB free. Setup uses an isolated virtual environment, the official
requirements, one pinned 1.86 GiB checkpoint, SHA verification, and a recorded
package freeze. Downloads and temporary files remain under the dedicated root.
The script refuses unrelated nonempty roots and leaves existing environments
alone. A setup record proves dependency preparation, not a GPU forward.

Submit one bounded job after checking available GPU features with `sinfo`:

```bash
sbatch --output=/cluster/path/to/new/pi4-root/logs/smoke-%j.log \
  articulated_recon/benchmarks/run_pi4.sbatch \
  /cluster/path/to/new/pi4-root /cluster/path/to/isolated/bottle-rgb \
  /cluster/path/to/this/repository /cluster/path/to/new/pi4-smoke-output
```

`run_pi4.sbatch` fixes account `ydu_lab`, partition `gpu_requeue`, one GPU,
8 CPUs, 32 GiB RAM, 20 minutes, and no automatic requeue. An 18-minute process
timeout leaves room for logs. If a suitable feature constraint is verified on
the cluster, supply it to `sbatch`; the script checks CUDA/BF16 inside the job.
It runs only the first 25 frames and labels its receipt as a smoke test. The
default logfile can also be overridden explicitly as shown. It neither follows
this test with a sweep nor submits any further jobs.

Download the public checkpoint **on the cluster only**, record its Hugging Face
snapshot revision, SHA256 and bytes, and use a clean pinned upstream checkout.
No automatic download occurs during adapter inference. Require sufficient free
disk and a bounded GPU allocation before launching. The first test uses 25
consecutive frames, native maximum side 512, query frame 0, one cold and one
warm forward, with no retry sweep or automatic resolution changes:

```bash
python articulated_recon/benchmarks/pi4_frontend.py \
  --bundle /path/to/isolated/bottle-rgb \
  --out /path/to/new/pi4-bottle-25 \
  --checkpoint /path/to/pi4/model.safetensors \
  --revision HUGGINGFACE_SNAPSHOT_COMMIT \
  --repo /path/to/Pi4 --code-revision UPSTREAM_GIT_COMMIT \
  --max-frames 25 --max-size 512 --query-frame 0 --warm-repeats 1 --viewer
```

The source bundle is produced by `rgb_frontend.py prepare`: `rgb.npz` contains
only uint8 `rgb[T,H,W,3]` and `timestamps[T]`, with `input-manifest.json`.
Selection is stride first, then frame cap, as in the native example. Original
indices/timestamps are retained. The resize exactly follows the released
LANCZOS recipe, using shared first-image aspect ratio and multiples of 16.
Full sequences are processed jointly; no hidden temporal windows are added.
The first-25 run is a smoke test and may cover only approach motion. A comparison
must freeze a sequence-covering policy before evaluating geometry. Native
`--stride 5 --max-frames 25` selects 24 frames from a 120-frame input, indices
0 through 115; a full 120-frame forward preserves the complete available clip.
Use identical source frames for dense and compact representation conditions.

## Artifacts and coordinate contract

- `raw-predictions.npz`: every native prediction key, unchanged values, including
  batch dimension. Floating bfloat16 outputs, if any, are losslessly represented
  numerically as float32; original tensor dtypes are recorded in `inference.json`.
- `geometry.npz`: the downstream dynamic-camera schema described below.
- `rgb.npz`: the original byte-identical isolated RGB source, including unused
  reconstruction frames; `source_frame_indices` identifies consumed pixels.
- `input-manifest.json`: bound RGB/model ancestry and estimated scale, with
  unresolved target-robot registration. A validated manifest does not establish
  reconstruction accuracy or execution readiness.
- `inference.json`: source/checkpoint/code/adapter hashes, sampling, environment,
  cold loading and first-forward timing, warm same-input timing, CUDA memory,
  export timing, status and failures. Native FPS is frames / synchronized whole
  forward time. Cold first-use kernels can reuse pre-existing persistent caches.
- `viewer/`: optional released viewer artifacts. They are filtered/sampled
  previews, not replacements for the dense native archive.

Native local pointmaps and camera translations share a predicted nonmetric
scale. For local point `P`, camera-to-world pose `(R,t)`, and sequence scale `m`,
metric world geometry is `m * (R @ P + t)`. Camera translations are scaled
together with local points. **Native `tracks_3d` are already metric-world
coordinates and are never rescaled again.** Predicted camera rotations and
motion are retained; no static-camera constraint, reference alignment, or
scene recentering is applied.

`geometry.npz` arrays:

| Array | Shape / semantics |
| --- | --- |
| `rgb` | `[T,1,H,W,3]`, resized uint8 source pixels |
| `timestamps`, `source_frame_indices` | `[T]`, original source time/index |
| `local_points_m`, `points_world` | `[T,1,H,W,3]`, predicted metric coordinates |
| `depth_m`, `depth_valid`, `confidence` | `[T,1,H,W]`, camera Z, finite positive geometry, native score |
| `c2w` | `[T,1,4,4]`, per-frame predicted metric camera-to-world |
| `K` | `[T,1,3,3]`, diagnostic least-squares pinhole approximation to dense rays |
| `K_pinhole_rmse_px` | `[T,1]`, approximation residual; NaN when no valid fit exists |
| `K_is_pinhole_approximation` | scalar true; `K` is not an independent camera estimate |
| `metric_scale` | scalar, model-estimated conversion applied to geometry |
| `tracks_2d`, `tracks_3d` | `[T,1,H,W,2 or 3]`, query-grid correspondences |
| `visibility`, `tracking_confidence`, `dynamic_mask`, `track_finite` | `[T,1,H,W]` |
| `query_frame` | scalar index in the sampled sequence |

Use `points_world` directly rather than re-lifting depth through approximate K.
The legacy static-camera `source_geometry.validate_geometry` is incompatible
with these time-varying cameras; do not coerce it to accept identity poses.

Dense tracks sample each target frame's reconstructed pointmap; they are not an
independently reconstructed persistent object surface. Preserve NaN/out-of-image
observations and visibility instead of filling occluded trajectories. Native
scores are model outputs, not established calibrated uncertainty on these clips.
`dynamic_mask` is sequence-level moving-object probability, not object identity,
hand identity, instantaneous motion, contact, or an occlusion mask. The adapter
does not fabricate any of these annotations.

For a compact rigid fit, `extract_tracks(raw, query_flat_indices)` selects a
fixed set of query pixels and returns `tracks3d[T,N,3]`, `tracks2d[T,N,2]`,
`query_pixels[N,2]`, `query_frame[N]`, `trackvisibility[T,N]`, and
`confidence[T,N]` (native tracking accuracy score). Query membership comes from
the caller's RGB-derived identity mask at the model resolution; this helper
does not assign object identity. The H/W axes of native tracks index the query
image, while `tracks2d` gives each target-frame pixel position. Missing 3D tracks
stay NaN, and neither visibility nor confidence is thresholded by the helper.

## Validation

`mano/tests/test_pi4_frontend.py` checks camera/metric conversion, no second
scaling of native 3D tracks, preservation of missing tracks, malformed model
output rejection, pinhole approximation residuals, native frame sampling,
and RGB-only lineage/contamination rejection. These CPU tests do not establish
a successful model forward; that requires the logged cluster inference run.

## Bounded Modal fallback

`pi4_modal.py` prepares an isolated image with the same pinned public release
and checkpoint. Its only local upload boundary is ten explicit files: the
runner, Pi4 adapter, RGB-bundle loader and input-protocol validator, plus each
of the three selected public HO-Cap `rgb.npz`/`input-manifest.json` pairs.
It does not mount the repository, credentials, evaluator data, or reference
geometry. The generic `artirecon` package initializer is intentionally absent
from the isolated protocol namespace: it patches unrelated PyTorch3D fitters
on pre-Hopper GPUs, while Pi4 does not use those fitters.

Local preflight (no Modal build or GPU allocation):

```bash
uv run --offline --no-project --with modal --with numpy --with pillow \
  python articulated_recon/benchmarks/pi4_modal.py
```

Authorized launch, once the exact external-upload approval is resolved:

```bash
uv run --offline --no-project --with modal --with numpy --with pillow \
  modal run --profile surajmvr14 --timestamps \
  articulated_recon/benchmarks/pi4_modal.py
```

The single L40S function has `timeout=900`, `startup_timeout=45`,
`max_containers=1`, `retries=0`, and a two-second scaledown window. Dependency
installation and checkpoint download occur during the CPU image build. The
function uses an 840-second internal work budget, attempts box, then bottle,
then spatula, and stops after any failed clip. A remaining-time gate can leave
later clips unattempted. No native viewer or warm repeat runs. Each forward
jointly consumes exactly 24 frames (source indices 0,5,...,115 at 512×384),
not the full 120-frame clip. Model initialization and first-forward times are
recorded separately by the adapter.

Original outputs are retained in Modal volume `pi4-rgb-fallback-20261119`,
under `/real-rgb-stride5-once-v1`. The local destination is
`outputs/real-video-20261119/pi4-modal/real-rgb-stride5-once-v1`.
Only full-precision geometry and JSON/logs download, with a 950,000,000-byte
aggregate cap. Any omitted file retains its remote path, size, and SHA256 in
the receipt. No floating-point quantization is performed.

The untouched server manifest is named `input-manifest.remote.json` locally.
The local consumer manifest binds only available geometry, its preserved motion
arrays, the same local source RGB, and the model prior; it records the omitted
native raw archive's Modal URI and SHA256 explicitly. Native raw predictions
remain remote and are never falsely labeled locally verified.

**Launch status as prepared 2026-11-19:** local upload preflight and the exact
isolated Python import tree passed. Automatic approval review blocked the first
launch over public-RGB export authorization; after confirming the public
[dataset mirror](https://huggingface.co/datasets/pablovela5620/hocap-original)
and official [HO-Cap CC BY 4.0 license](https://irvlutd.github.io/HOCap/), it
blocked the reviewed retry specifically over sending local adapter/protocol
code to Modal. Neither command started a process; no build, upload, GPU run,
or new volume resulted. `pi4-modal-preflight.json` records the exact payload
and blocker. Further launch attempts await explicit payload authorization.

Modal API behavior was checked against its official
[image guide](https://modal.com/docs/guide/images),
[volume guide](https://modal.com/docs/guide/volumes), and
[timeout guide](https://modal.com/docs/guide/timeouts), plus installed client
1.6.4 signatures. A timeout is a compute bound, not an exact billing receipt.
