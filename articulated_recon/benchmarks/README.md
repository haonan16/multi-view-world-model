# RGB scene and motion experiments

Target: recorded RGB or an RGB-conditioned forecast → estimated scene/motion →
frozen robot commands → a separately constructed evaluation simulator.

The code in this directory implements input preparation, validation, comparison
configuration and evaluation submission. It does not constitute a completed
RGB-only reconstruction or a successful independent simulator rollout.

- [Protocol and staged comparison matrix](README_protocol.md)
- [External baseline installation and launch adapters](README_external_baselines.md)
- `rgb_frontend.py`: isolate one recorded RGB camera, then predict metric depth
  and camera intrinsics with an existing local UniDepthV2 checkpoint.
- `keyframe_compression.py`: compress saved fitted trajectories and measure the
  interpolation error. This is separate from generating only five keyframes.
- `run_unidepth.sbatch`: bounded GPU job using existing dependencies/checkpoints;
  choose account and partition on submission. No automatic download.

## Source RGB preparation

```bash
python articulated_recon/benchmarks/rgb_frontend.py prepare \
  --source /path/to/source.mp4 --out /path/to/isolated-sequence \
  --task-mode reconstruction

python articulated_recon/benchmarks/rgb_frontend.py prepare \
  --source /path/to/source.mp4 --out /path/to/isolated-first-frame \
  --task-mode forecast
```

Video input retains native resolution. Forecast preparation decodes one frame.
Prepared arrays contain only `rgb` and `timestamps`. For a multiview NPZ choose
exactly one `--camera-index`; depth, calibration and geometry are not exported.
The caller must establish that the input is recorded RGB: a generated video
cannot become an observation merely by passing it to this tool.

```bash
python articulated_recon/benchmarks/rgb_frontend.py infer-unidepth \
  --bundle /path/to/isolated-sequence --out /path/to/new-predictions \
  --checkpoint /path/to/existing/unidepth-v2-vitl14/snapshot \
  --revision CHECKPOINT_COMMIT --device cuda
```

Frame zero estimates camera intrinsics. Later frames reuse that estimate under
the explicit fixed-camera assumption; each inference gets a fresh camera
instance because UniDepth mutates camera objects internally. Identity source
extrinsics define coordinates, not registration to a real robot. The output
contains camera-Z depth, raw predicted error and an inverse-error **rank**.
That rank is not a calibrated probability. Table estimation, segmentation,
object fitting and robot registration are still downstream stages.

## Retrospective keyframe diagnostic, 2026-11-19

We tested all 15 preselected historical tracks: six `joint-sequence` variants,
eight `joint-silhouette` variants and the 25-frame `stable_support` block fit.
Seven older block tracks had only six valid poses out of nine; the diagnostic
reported them as unsupported instead of promoting visualization-filled poses.
The eight supported tracks are one block fit and seven variants of the same
bottle sequence, not eight independent scenes.

At a five-state budget, worst block position deviation from its full fitted
trajectory was 74.3 mm for uniform selection, 122.7 mm for estimated contact-event
selection and 56.8 mm for greedy motion selection. Bottle worst-position ranges
were 20.6–53.9 mm for uniform selection and 17.5–28.5 mm for motion selection.
The block's mean deviation increased slightly (18.7 to 20.2 mm), and some
rotation deviations increased. Contact-only selection equaled uniform for
the bottle because those saved files supplied no contact boundaries.

A follow-up budget sweep on the same block fit gave worst position deviations:

| Retained poses (of 25) | Uniform | Contact events | Motion-aware |
| --- | ---: | ---: | ---: |
| 5 | 74.3 mm | 122.7 mm | 56.8 mm |
| 9 | 41.5 mm | 41.5 mm | 13.5 mm |
| 13 | 27.0 mm | 41.5 mm | 7.4 mm |

All policies reproduce the original fit exactly when all 25 poses are retained.
This is one existing fit, not a held-out selection rule or a robot success test.

The motion selector consumes the whole fitted trajectory and greedily preserves
bounding-box point motion. These are compression errors against saved predictions,
not ground-truth errors, measured runtime savings, or execution outcomes. The
historical fits used supplied calibration/support. This result motivates testing
motion bends plus contact timing; it does not establish that five generated
keyframes are sufficient or that the world model improves reconstruction.

```bash
python articulated_recon/benchmarks/keyframe_compression.py \
  --trajectory /path/to/fit/trajectory.npz \
  --trajectory /path/to/another-fit/trajectory.npz \
  --budget 5 --out /path/to/new-report.json
```

The adjacent historical `scene.json` supplies the existing mesh and declared
support plane. Both trajectory variants and unsupported inputs remain in the
report. Original geometry, poses and plans are never modified.
