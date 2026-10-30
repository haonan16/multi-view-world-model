# RH20T multi-view world model

## Shared presentation (Claude and Codex)

Use the existing deck at `slides/rh20t_unseen_views/index.html` and its sibling
`assets/` directory as the single presentation source. Each slide is a `<section>`;
chart values are in the `curves` JSON script block. Both Claude and Codex should
edit this source rather than create separate online decks. Read the current files
before editing and coordinate edits when another session is working on the deck.

The intended publication target is the `rh20t_unseen_views/` directory on the
`gh-pages` branch of `haonan16/multi-view-world-model`, served at
<https://haonan16.github.io/multi-view-world-model/rh20t_unseen_views/>.
Verify deployment before describing that URL as live. The empty Claude Slides
and ChatGPT Pages presentations are not the shared deliverable.

## Agreed project goal

The user-confirmed goal is to build a world model that takes point-cloud
observations, predicts how the scene evolves over time, and generates consistent
views of the predicted scene from multiple cameras. Predictions should preserve
observed geometry and spatial relationships, remain coherent over time, and
depict the same scene across cameras.

This is the shared goal reference for Claude and Codex. The concrete state
representation, point-cloud encoder, and rendering architecture remain open.
Whether future 3D states are explicit outputs or represented internally has not
been decided. Static novel-view rendering alone does not satisfy the goal.

### Existing baselines and missing capabilities

The implementation described below predicts future RGB frames from synchronized
RGB history and calibrated camera rays using Wan2.1 in `fastvideo/train/`.
Camera raymaps encode camera geometry; they do not supply observed scene depth
or point clouds. The original RGB baseline does not condition on scene depth.

The unseen-view extension (`rh20t_unseen_pc.yaml`) now loads episode depth from
the older `HaojunChen/RH20T` mirror, fuses the two context cameras' frame-0 RGB-D
into a point cloud, and conditions generation on its per-view render and coverage
mask. `rh20t_unseen_rgb.yaml` is the corresponding RGB-only comparison. This is
static observed-geometry conditioning; explicit prediction of future 3D states
and robot-action conditioning remain unimplemented.

The existing configs use four views, 13 frames at 3 fps (a four-second span),
and one clean history latent frame. These are baseline settings, not confirmed
requirements for the point-cloud model. Existing PSNR/SSIM evaluation measures
image reconstruction quality; it does not establish 3D or cross-view consistency.

### Proposed next milestone

First establish the point-cloud data source, timestamps, coordinate system,
units, and calibration, and verify alignment by projecting an observed cloud
into synchronized camera images. Use only observations available at prediction
time as conditioning; future observations belong to targets or evaluation.

Then define a small end-to-end baseline that adds geometry conditioning to the
existing multi-view predictor, with held-out future clips, comparison against
the RGB-only and copy-first-frame baselines, and geometry-aware cross-view and
temporal evaluation. This is a proposal, not an approved architecture or a claim
that the point-cloud model already works.

Open decisions: measured versus reconstructed point clouds; availability of
depth and aligned RGB; action conditioning; observation history and prediction
horizon; whether arbitrary novel camera poses are required; explicit future 3D
outputs; and quantitative acceptance thresholds.

## RGB multi-view baseline

Multi-view future-frame prediction on [RH20T](https://rh20t.github.io/) with
Wan2.1-T2V-1.3B in the modular trainer (`fastvideo/train/`). This is a port of
akiruga's `ego_world_model` multi-view ablation (`algorithms/wan/wan_multiview.py`,
`datasets/rh20t.py`), comparing two ways of feeding V camera views to one DiT:

| design | packing | tokens / sample (4 views, 13 frames, 160x320) |
|--------|---------|------|
| `tokens` | each view VAE-encoded at 160x320; all views' tokens in one sequence, identical RoPE per view | 4 x 4 x 10 x 20 = 3200 |
| `tiled` | views area-downsampled into one 2x2 160x320 canvas, encoded as one video | 4 x 10 x 20 = 800 |

Both designs condition on per-view Plucker raymaps (6 channels concatenated to the
latent before the patch embedding, zero-initialised so step 0 is the pretrained T2V
model) and keep latent frame 0 of every view clean as history (`p_no_hist` = 0.1 of
samples train unconditionally). The loss is flow-matching MSE on the noised frames.

## Code map

| file | role |
|------|------|
| `fastvideo/models/wan/multiview_transformer.py` | `MultiViewWanTransformer3DModel`: `[B, V, C+6, F, h, w]` in, per-view RoPE, per-(view, frame) timestep modulation |
| `fastvideo/dataset/rh20t_multiview.py` | `RH20TMultiViewDataset`: timestamp-synchronised views, front-facing view choice, raymaps; `tile_views` / `untile_views` |
| `fastvideo/train/models/wan/multiview.py` | `WanMultiViewModel`: online VAE encode, packing, clean-history noising, Euler sampler |
| `fastvideo/train/methods/fine_tuning/multiview_flow.py` | `MultiViewFlowMethod`: masked loss; validation PSNR/SSIM vs VAE ceiling and copy-frame-0 baseline |
| `scripts/rh20t/make_multiview_checkpoint.py` | builds the 22-input-channel diffusers checkpoint |
| `examples/train/configs/fine_tuning/wan/rh20t_multiview_{tokens,tiled}.yaml` | run configs |
| `scripts/rh20t/{train_multiview,setup_env,download_lowdim}.sbatch` | launch / env / data jobs |
| `fastvideo/dataset/rh20t_depth.py` | depth.mp4 decoding, depth cleaning, unprojection, z-buffered point splatting |
| `scripts/rh20t/{download_depth,prepare_unseen_checkpoints,train_multiview_h100}.sbatch` | depth download, unseen-view init checkpoints, multi-node H100 launcher |
| `scripts/rh20t/{check_depth_alignment,widen_checkpoint}.py` | depth/calibration alignment check; add zero input channels to a checkpoint |

## Data

- **Color video**: the authors' official 320x180 release, extracted by akiruga at
  `/n/netscratch/ydu_lab/Lab/akiruga/rh20t_lowres/data` (all 7 cfgs, 24,874 episode
  folders incl. human demos; color + per-frame timestamps + `metadata.json`).
- **Calibration and robot low-dim signals**: re-downloaded from
  `tonyfanggalaxies/RH20T` to `/n/lab_storage/ydu_lab/Lab/haonan/datasets/rh20t/{calib,lowdim}`
  by `scripts/rh20t/download_lowdim.sbatch` (`transformed/*.npy` + `metadata.json`
  only, ~6 MB per episode).
- The full-resolution copy at `/n/netscratch/ydu_lab/Lab/akiruga/rh20t/data` was found
  emptied on 2026-10-09 (directory tree intact, every file gone), most likely the
  netscratch 90-day purge acting on the 2021-22 mtimes that tar extraction preserved.
  Keep extracted RH20T trees on lab_storage, or touch them after extraction.

Frames are irregular in time (~7-11 Hz, per-camera drops), so each sample lays a
uniform `fps` grid over the cameras and takes each view's nearest frame, rejecting
views with any frame further than `max_sync_err_ms`. In-hand cameras are excluded.

### Point-cloud data availability

Local inspection on 2026-10-17 found 189 `intrinsics.npy` files, 189
`extrinsics.npy` files, and 1,610 calibration depth snapshots (`imgs/*_d.npy`)
under the lab_storage calibration root. Two inspected snapshots were 720x1280
`uint16` arrays. These are calibration observations, not task-episode depth
sequences; their depth units and RGB alignment still need verification.

The old full-resolution episode tree currently contains no files. A sampled
camera from each of the seven low-resolution configurations contained only
`color/color.mp4` and `color/timestamps.npy`. No ready-made point clouds or
task-depth paths were found under the lab_storage RH20T root.

The [official RH20T release](https://huggingface.co/datasets/tonyfanggalaxies/RH20T)
describes corrected 320x180 depth images separately as `RH20T_depth`. However,
on 2026-10-21 its Hugging Face API file listing contained only calibration,
color, and low-dimensional data splits. Both the `RH20T_depth` directory API
and the `RH20T_depth/RH20T_cfg3.tar.gz` download returned HTTP 404. The README
also lists Baidu links, but their download availability has not been verified.
Do not assume the corrected depth is downloadable from Hugging Face merely
because the dataset card describes it.

**Episode depth source (2026-10-21).** The community mirror
[`HaojunChen/RH20T`](https://huggingface.co/datasets/HaojunChen/RH20T) has the original
(uncorrected) per-episode depth as `depth/RH20T_cfgN_depth.tar.gz.part-NNN` (~690 GB over
37 parts, robot and human episodes, every camera): `RH20T_cfgN/<episode>/cam_<serial>/depth.mp4`.
Each frame is 640x720 grey with the low byte of a uint16 depth map on top and the high
byte below (rh20t_api `extract.convert_depth`), aligned to the colour camera. Depth frame
`i` pairs with colour frame `i`: on a sampled cfg3 episode the depth frame counts equalled
the `color/timestamps.npy` lengths for all 8 cameras. Units are `raw / 1000` metres for both
camera types (L515 `f*` serials store quarter-millimetres, which the API multiplies by 4 and
then divides by 4000). The video coding is lossy, so the high byte can be off by one (256 mm
spikes); `clean_depth` drops pixels more than 5 cm from their 3x3 median.
`scripts/rh20t/download_depth.sbatch` keeps robot episodes and static cameras only, under
`/n/lab_storage/ydu_lab/Lab/haonan/datasets/rh20t/depth`.

**Alignment check** (`scripts/rh20t/check_depth_alignment.py`, cfg3
`task_0001_user_0016_scene_0001`, 5 synchronised static cameras). For each camera, the other
cameras' depth at the same instant was fused and rendered into it:

| camera | median depth (m) | coverage | median abs. depth error, render vs. own sensor (cm) | same, rendered with the next camera's pose (cm) |
|--------|-----|-----|-----|-----|
| 036422060909 | 0.82 | 0.74 | 0.7 | 24.5 |
| 104122062295 | 0.64 | 0.75 | 1.0 | 9.4 |
| 104122062823 | 0.61 | 0.73 | 0.5 | 29.7 |
| 104122063550 | 0.91 | 0.83 | 12.4 | 40.2 |
| f0172289 | 0.69 | 0.70 | 1.4 | 21.3 |

Depth scale, intrinsics scaling, extrinsics and frame pairing are consistent to about 1 cm
on four of the five cameras. 104122063550 is worse (12 cm) and should be looked at before
trusting it as a context view. Only one episode was checked.

## Point-cloud conditioning: unseen views

The first geometry-conditioned experiment. Of the 4 views, `n_context_views` (2, chosen
at random) are observed at frame 0. The other views have no observation and are
generated from frame 0 on. With `point_cloud: true`, the context views' frame-0 depth
is fused into one coloured point cloud (in view 0's frame, metres) and z-buffer splatted
into every view at 160x320 (`render_context_cloud`). The VAE latent of that render plus a
coverage mask go into latent frame 0 as 17 extra input channels, zero-initialised. Only
frame 0 is used, so no future observation reaches the conditioning. The future 3D state
is not predicted explicitly; this tests whether observed geometry helps the model
generate consistent unseen views over time.

| config | conditioning | init |
|--------|--------------|------|
| `rh20t_unseen_pc.yaml` | RGB history of the context views + raymaps + point-cloud render | `datasets/models/rh20t_mv_tokens_10k_pc17` |
| `rh20t_unseen_rgb.yaml` | RGB history of the context views + raymaps (ablation) | `datasets/models/rh20t_mv_tokens_10k` |

Both are warm-started from the RGB tokens baseline at step 10,000
(`scripts/rh20t/prepare_unseen_checkpoints.sbatch`: DCP export, then
`widen_checkpoint.py` for the extra inputs). Both use `depth_root`, so they train and
evaluate on the same episodes and cameras. Validation adds `val/*_ctx` (context views,
frames 1..), `val/*_tgt` (target views, every frame) and, for the point-cloud run,
`val/*_render_tgt` (the render held fixed over time, a geometry-only reference). These
are still image metrics; cross-view geometric consistency is not measured yet.

### Matched evaluation

Training-time validation scores each run on a few clips that depend on its GPU count, so
runs are not comparable through it. `scripts/rh20t/eval_unseen_matched.py` scores a DCP
checkpoint on val indices 0..N-1 (254 = one clip per val episode). The clip, cameras,
context views and sampling noise depend only on the index, so all checkpoints get the
same inputs. It records per-view, per-frame PSNR/SSIM of the prediction, the copy-frame-0
baseline and the render. Unseen-view PSNR is also split into pixels the context cloud
covers and pixels it does not. `scripts/rh20t/summarize_unseen_eval.py` checks that the
inputs match and reports paired 95% bootstrap intervals against the first arm.

```bash
sbatch scripts/rh20t/eval_unseen_matched.sbatch <checkpoint-N> outputs/eval_unseen_matched/full/<name>
python scripts/rh20t/summarize_unseen_eval.py rgb=outputs/eval_unseen_matched/full/<rgb> \
    pc=outputs/eval_unseen_matched/full/<pc>
```

Results from 2026-10-29 on all 254 val clips (`outputs/eval_unseen_matched/full/summary*.{md,json}`).
Samples count training clips seen from the RGB tokens baseline onward:

| checkpoint | samples | unseen PSNR / SSIM | observed PSNR |
|------------|---------|--------------------|---------------|
| RGB, step 10,000 | 320k | 16.72 / 0.550 | 22.41 |
| point cloud, step 10,000 | 160k | 17.30 / 0.572 | 22.52 |
| RGB, step 32,000 (`outputs/rh20t_unseen_rgb_cont_20261028`) | 1.02M | 18.37 / 0.644 | 22.80 |
| point cloud, step 43,000 (`outputs/rh20t_pc_week_20261024`) | 1.06M | 19.08 / 0.678 | 22.67 |
| copy frame 0 (observed views) | – | – | 26.12 |

At matched samples, the point cloud adds +0.71 dB PSNR on unseen views (95% CI
+0.61 to +0.82) and +0.034 SSIM. It wins on 83% of clips. The gain is about the same on
pixels the cloud covers and pixels it does not, and it does not depend on render quality.
The render therefore seems to act as a global layout cue rather than being copied into
the image. On observed views neither model beats copying frame 0. The gap is 6.3 dB at
frame 1 and 1.7 dB at frame 12, so it comes from the first generated latent rather than
from drift.

## Setup

```bash
sbatch scripts/rh20t/setup_env.sbatch           # uv venv at venvs/mvwm, editable FastVideo (cu126)
source /n/lab_storage/ydu_lab/Lab/haonan/venvs/mvwm/bin/activate
python scripts/rh20t/make_multiview_checkpoint.py \
    --out /n/lab_storage/ydu_lab/Lab/haonan/datasets/models/wan_multiview_ray6/Wan2.1-T2V-1.3B-Diffusers
pytest fastvideo/tests/dataset/test_rh20t_depth.py fastvideo/tests/dataset/test_rh20t_multiview.py \
    fastvideo/tests/train/test_rh20t_multiview_model.py -q
```

## Train

### Week-long point-cloud continuation

The 2026-10-24 continuation uses the completed point-cloud checkpoint at step
10,000, four H100s, per-GPU batch size 4, and gradient accumulation 2 (effective
batch 32). Its isolated run directory is `outputs/rh20t_pc_week_20261024/`.
The `control/` directory holds the launch configuration, source hashes, segment
controls, and job manifest; `source/` freezes the Python package for this run.
The initial checkpoint is copied separately under `initial/` and is not pruned.
Do not launch another writer into this directory from a parallel agent session.

Four sequential allocations budget 48 + 48 + 48 + 24 hours (168 hours, or 672
H100 GPU-hours). This is allocated runtime, including startup, validation, and
checkpoint I/O; queue delays extend the calendar duration. Jobs do not requeue
automatically. After a normal completion or time limit, the next segment selects
the newest structurally complete checkpoint, checking DCP storage bounds and all
four RNG snapshots. Unexpected job failures or absent checkpoint progress stop
the continuation chain. An interrupted save is skipped in favor of an earlier
complete checkpoint, so some work since the last save may be repeated.

Validation remains every 1,000 steps on the same four-rank subset as the original
point-cloud run; checkpointing is every 500 steps, retaining three recent saves.
The step limit is set high so allocation limits bound this experiment. This
continuation does not replace the planned matched RGB-versus-point-cloud
evaluation, and scheduled jobs do not imply continuous agent monitoring.

### H100 launch configuration

For H100 runs, `scripts/rh20t/train_multiview_h100.sbatch` selects
`kempner_h100` and defaults to two nodes with four GPUs each. It derives the
distributed world size from the allocation. The existing configs stop at
10,000 steps, so continuing a completed run requires a larger
`--training.loop.max_train_steps`. For an isolated continuation, set an absolute
`--training.checkpoint.resume_from_checkpoint` and a new
`--training.checkpoint.output_dir`. These runs still use RGB and camera-ray
conditioning, not point clouds.

```bash
sbatch scripts/rh20t/train_multiview.sbatch examples/train/configs/fine_tuning/wan/rh20t_multiview_tiled.yaml
sbatch scripts/rh20t/train_multiview.sbatch examples/train/configs/fine_tuning/wan/rh20t_multiview_tokens.yaml \
    --training.loop.max_train_steps 200      # dotted overrides are forwarded
# unseen views (needs the depth download and prepare_unseen_checkpoints.sbatch)
sbatch -J rh20t_unseen_pc scripts/rh20t/train_multiview_h100.sbatch \
    examples/train/configs/fine_tuning/wan/rh20t_unseen_pc.yaml
sbatch -J rh20t_unseen_rgb scripts/rh20t/train_multiview_h100.sbatch \
    examples/train/configs/fine_tuning/wan/rh20t_unseen_rgb.yaml
```

The first run walks all episodes once (~20 s; 12,746 usable robot episodes, 12,492
train / 254 val) and caches the record list under `datasets/rh20t/cache/`; delete it if
the data changes. `training.distributed.sp_size` must be 1 (each rank draws its own batch).

Smoke runs (4x A100-40GB, 30 steps, 2026-10-09): tokens bs 2/GPU ~1.5 s/step, tiled bs 8/GPU
~2 s/step, both with full activation checkpointing (tiled bs 8 OOMs on 40 GB without it).
VAE round-trip ceiling on held-out clips: 32.4 dB PSNR (tokens) vs 26.2 dB (tiled); the
copy-frame-0 baseline is 25.6 dB. Resume from `checkpoint-N` (model, optimizer, RNG,
sampler position) was verified.

Full runs (2026-10-16, 2 nodes x 4 H100, 10,000 steps, global batch = 8 GPUs x per-GPU
batch). Validation covers 32 held-out clips (2 batches x 2 clips on each of 8 ranks), so
differences under about 1 dB are within noise:

| design | batch / GPU | wall time | PSNR pred / VAE / copy0 (dB) | SSIM pred / VAE / copy0 |
|--------|-----|-----|-----|-----|
| tokens | 4 | 5.4 h | 22.6 / 32.3 / 25.4 | 0.82 / 0.94 / 0.91 |
| tiled | 16 | 6.0 h | 21.3 / 26.1 / 25.4 | 0.71 / 0.80 / 0.91 |

Neither design beats copying frame 0 on these image metrics yet. W&B: project
`hca/rh20t_multiview`, runs `mv_tokens_8xh100` and `mv_tiled_8xh100`.

## Cluster notes

- `fastvideo-kernel` (sparse-attention kernels) is not installed: its x86 wheel needs glibc
  >= 2.34 (cluster: 2.28) and the sdist needs nvcc. Dense attention does not use it.
- The torch cu126 wheel has no Blackwell (sm_120) kernels, so the launcher constrains
  `kempner_requeue` to `h100|h200|a100`. `holygpu8a19102` failed twice with
  "CUDA-capable device(s) is/are busy"; pass `--exclude=holygpu8a19102` if it recurs.
- The empty-prompt UMT5 embedding is precomputed into `<checkpoint>/null_prompt_embed.pt` by
  the converter; loading the fp32 text encoder on every rank OOMs A100-40GB.

## Differences from the reference implementation

- Validation samples with a plain Euler flow-ODE solver (30 steps) instead of UniPC and
  reports PSNR/SSIM at full per-view resolution only (no LPIPS, no tile-resolution
  metrics), so numbers are not directly comparable with akiruga's runs.
- Episodes that fail to decode for any reason are skipped and resampled.

## Not yet done

- Action conditioning: the low-dim signals (`tcp_base`, `gripper`, `force_torque`) are
  being downloaded but are not read by the dataset or the model yet.
- Human demos (`*_human`) are skipped; they have color but no robot data.
