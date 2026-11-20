# One observed image and a task: strict forecast adapter

The intended primary experiment is one observed RGB image plus an explicit task
instruction → generated multiview motion → a shared 3D scene and motion estimate
→ a frozen robot plan → independent simulator execution. Recorded full-video
reconstruction remains a diagnostic.

This module implements the **input/output boundary for the existing RGB Wan
adapter**, with CPU tests. It does not establish successful GPU forecasting,
accurate generated views, an agent-reconstructed simulator, or robot execution.
The adapter does not export native world-model depth. DA3 depth is a separate
model estimate when that optional downstream adapter is used.

## Preparation

Use `rgb_frontend.py prepare --task-mode forecast` to export a numeric source NPZ
containing exactly `rgb[1,H,W,3]` and `timestamps[1]`. Run the existing image depth
estimator and initial scene estimator using only that packet. Both resulting
manifests must retain `condition=rgb_only`, `task_mode=forecast`, and
`observed_until_s=t0`. An initial object pose is used to design camera requests;
this is **not** a future motion plan or a recorded full-sequence reconstruction.

The preparation checks hashes and declared ancestry, then checks actual array
contents. The initial geometry bundle must contain one source image, predicted
depth `[1,1,H,W]`, estimated K, an identity source-camera pose, and one timestamp.
The shared scene must contain one timestamp, use `source_camera`, and explicitly
identify a supported initial object pose. Its metric scale is estimated, not
known. A full-clip scene or camera bundle cannot become a first-image input by
changing the outer manifest's label. Supplied calibration and sensor/GT depth
are rejected by this strict RGB-only path.

From the repository root:

```sh
python articulated_recon/benchmarks/world_model_views.py prepare-forecast \
  --geometry-bundle /path/to/first-image-depth \
  --scene-path /path/to/first-image-scene/scene-sequence.json \
  --scene-manifest /path/to/first-image-scene/input-manifest.json \
  --object-id block \
  --task-text 'Pick up the small block on the left and move it to the right' \
  --prompt-key task_0004 \
  --frames 17 --dt-s 0.3333333333333333 --angle-degrees 20 \
  --output /path/to/new-forecast-condition
```

The caller must explicitly choose the prediction cadence and horizon before inference. Times are
`t0 + arange(frames)*dt_s`; they are prediction labels, not timestamps read from a
future recorded clip. These timestamps do **not** condition the sampler's FPS.
The documented robot+human checkpoint includes single-view first-frame training;
its data recipe uses 17 frames at 3 fps for human clips and 1.5 fps for robot
clips. The example above declares the human cadence, without claiming that
changing the labels changes generated dynamics. The current virtual rig uses the initial object's estimated
position, a declared angle, and a centred 2:1 crop resized to 320×160. It has source
identity and three distinct nonzero-baseline requests. **Distinct requests do not
prove distinct or correct generated views.** Earlier virtual-rig diagnostics
produced near-copies; see `README_camera_audit.md`.

## Generation

```sh
python articulated_recon/benchmarks/world_model_views.py generate \
  --condition /path/to/new-forecast-condition \
  --repository /path/to/reviewed-author-checkout \
  --config /path/to/reviewed-config.yaml \
  --checkpoint /path/to/existing-checkpoint.ckpt \
  --checkpoint-receipt /path/to/existing-checkpoint-receipt.json \
  --prompt-key task_0004 --conditioning strict_forecast \
  --text-guidance 2 --hist-guidance 1 --seed 20261119 \
  --output /path/to/new-forecast-output
```

The command is a runnable capability; it does not schedule a job, fetch a model,
or claim these example settings have been tested for forecast quality. Existing
reviewed sampler/VAE source hashes and checkpoint receipt checks remain enforced.
GPU inference additionally verifies the observed source latent stays clamped.

The task text and key declared during preparation must match the **actual cached
embedding dictionary**, or a custom task/null dictionary supplied through
`--text-embeddings` and `--text-embedding-receipt`. Custom embedding receipt checks
remain unchanged. Task text and embedding hash are recorded; changed files are
rejected after inference. Null-text mode is rejected for a strict task forecast.
Zero text guidance still uses the task embedding; it removes CFG amplification,
not the task itself.

Only the initial source image enters the conditioning tensor. Unknown target and
future input pixels are zero. Output restoration replaces only frame 0/view 0
with the observed RGB; generated future source pixels remain generated. The
source, initial scene, camera requests and task embedding are rechecked before
writing output provenance with `strict_forecasting_claim=true`. That flag means
the **declared input boundary passed**, not that forecasting or planning worked.

## Output contract and downstream depth

| Packet | Arrays / meaning |
|---|---|
| `condition.npz` | `rgb[1,160,320,3]`, `K[4,3,3]`, `c2w[4,4,4]`, prediction `timestamps[T]`, `source_frame_indices=[0,-1,…]` |
| `generated.npz` | Raw decoded `rgb[T,4,160,320,3]`; requested K/c2w and prediction timestamps; observed mask identifies conditioned locations, not raw decoder pixel fidelity |
| `observations.npz` | Same generated sequence with only the initial source RGB restored; `observed_mask[T,4]` true only at `[0,0]`; no recorded future indices |

The output manifest stays `task_mode=forecast`. Artifact
`generated_source_forecast` has role `source_forecast`, selects view 0 and frames
1 onward, and binds `observations.npz`. Artifact `generated_helper_rgb` has role
`helper`, selects views 1–3 at all predicted times, and binds the same packet.
Both are generated model predictions. `forecast_camera_requests` remains in
their first-image-only ancestry. A mask on the combined packet distinguishes its
single observed image from generated pixels.

The existing `da3_helper_views.py` CLI autodetects the forecast manifest:

```sh
python articulated_recon/benchmarks/da3_helper_views.py \
  --helper /path/to/new-forecast-output \
  --condition /path/to/new-forecast-condition \
  --checkpoint /path/to/existing-da3-checkpoint \
  --output /path/to/new-predicted-depth --preflight
```

Remove `--preflight` only for an authorized inference run. The unchanged CLI runs
the source-only, helper-unposed and helper-requested-camera conditions. In a
forecast, `source_unposed` means the generated source-view future plus the initial
observation; it never means a fully observed source clip. The requested-camera
condition inherits estimated first-image metric scale. Unposed DA3 Giant has
unresolved metric scale at each sampling timestamp; DA3 alone does not create a
temporally rigid shared scene.

An agent/scene fitter must next turn predicted geometry and motion into shared
objects, contacts and support, preserve uncertainty, and resolve fixed robot
registration. Existing registered TCP planning and frozen independent evaluation
can then be reused. This new strict forecast → depth → scene → robot chain has
**not been run end to end** by the CPU tests.

## Validation

`mano/tests/test_world_model_forecast.py` checks first-image-only camera design,
declared future times, zero unknown conditioning pixels, rejection of future RGB
and scene/camera ancestry, packet/rig/task tampering, exact task text binding, and
the generated source/helper provenance. Existing reconstruction and calibrated
diagnostic tests remain unchanged. File hashes and declarations establish an
auditable boundary; they cannot prove a producer truthfully described all inputs.
