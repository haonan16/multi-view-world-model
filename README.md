# 4 camera views → 64-camera light stage with Lyra 2.0

This branch contains only what is needed to reproduce the light-stage pipeline **up to the Lyra flights**. It takes the 4
calibrated views of one RH20T timestep, either ground-truth frames or our multiview model's predicted frames, and
generates 64 new views on a dome around the table with NVIDIA's [Lyra 2.0](https://github.com/nv-tlabs/lyra). Everything
is pretrained; nothing is trained.

What the pipeline does, in one paragraph: it lifts the 4 views to 3D with metric depth (VGGT-Omega), places 64 cameras on
a dome around the workspace, and flies Lyra 2.0 through them. There are four short flights, one starting at each real
camera. All 4 real views (colour, metric depth and calibrated pose) are loaded into Lyra's 3D memory on every flight, so
what the real cameras saw is warped into each new camera and only unseen content is generated.

**Result on scenes 145 and 000 (t = 12, GT and predicted frames):**
- 64 sharp, mutually consistent views per case, with the robot arm in every view.
- About 8 min per case on one H100 with the 4-step DMD sampler.
- Generated cameras stay within about 2–5 cm of their requested poses. A single long flight drifted 20–36 cm, which is
  why the flights are short.

The flights can be viewed with the flight viewer used in the project
(`/n/lab_storage/ydu_lab/Lab/akiruga/lightstage_flight_viewer/index.html` on FASRC). Point-cloud fusion of the flights
is **not** in this branch: none of our fusion attempts were good enough to keep. Lyra's own reconstructor
(`vipe_da3_gs_recon.py` in the Lyra repo), run on one flight video at a time, gave the cleanest clouds so far.

## Layout

```
lightstage/
  setup_node.sh                  one-time node setup: env, code copy, Lyra clone + patch, checkpoints
  patch_lyra_multiview_seeds.py  1-hunk patch to Lyra: seed views are passed separately from the trajectory
  inputs/vggt_dynamic_clouds.py  metric depth for the 4 views (VGGT-Omega + Sim(3) to the calibrated cameras)
  rig/prep_lightstage.py         per-case rig.json: input cameras and the table plane   -> copied to /tmp/akiruga_lightstage
  rig/table_up.py                RANSAC table plane on the VGGT-Omega cloud
  lyra/lyra_lightstage2.py       64-camera dome + short seeded Lyra flights (the pipeline)   -> copied to /tmp/akiruga_lyra
  lyra/lyra_lightstage.py        helpers used by it (input loading, look-at, scene captions)
  lyra/run_lyra.sh               environment wrapper that runs a script from the Lyra-2 directory
  lyra/dl.sh                     downloads the Lyra 2.0 checkpoints (~97 GB)
```

The scripts hardcode `/tmp/akiruga_lyra`, `/tmp/akiruga_lightstage` and a few FASRC paths (listed below). `setup_node.sh`
recreates that layout on a fresh node.

## Requirements

- One or two 80 GB GPUs. Lyra uses about 68 GB per process. Tested on H100.
- The packed conda env `/n/netscratch/ydu_lab/Lab/akiruga/envs/test2.tar.zst`: Python 3.12, torch 2.9 cu129,
  transformer_engine, flash-attn, megatron-core, MoGe.
- VGGT-Omega code at `/n/lab_storage/ydu_lab/Lab/akiruga/vggt-omega`, with checkpoint `facebook/VGGT-Omega`
  `vggt_omega_1b_512.pt` in the HF cache `/n/netscratch/ydu_lab/Lab/akiruga/hub`. Only needed for the input step.
- About 100 GB of local `/tmp` for the Lyra checkpoints.

Always set `PYTHONNOUSERSITE=1`, because `~/.local` packages leak in otherwise.

## Inputs

Per held-out scene, in `ROOT=/n/netscratch/ydu_lab/Lab/akiruga/mv_infer/v1_heldout15/<scene>/`:
- `gt.npy`, `pred_tokens.npy`, `pred_tiled.npy`: uint8 `[V=4, T=13, H=160, W=320, 3]`.
- `cams.npz`: `intrinsics` (K at 160×320) and `c2w` (OpenCV camera-to-world, metres, view 0 = identity).

These come from `scripts/rh20t/infer_multiview.py` on the `multiview-wan` branches. The predictions need the multiview
model code, which is not part of this branch; the GT frames and cameras are RH20T data.

Metric depth for those views (step 1 below) goes to `ROOT/clouds_native/<scene>/<source>/{dc.png, color.jpg, meta.json}`.
That's a 16-bit depth atlas at 368×720 plus a colour atlas, and per view and frame the K, the camera-to-world `M` in the
calibrated frame, and the depth range.

## Steps

```bash
# 0. node setup (once per node), from the repo root
bash lightstage/setup_node.sh
PY=/tmp/akiruga_envs/test2/bin/python; export PYTHONNOUSERSITE=1

# 1. metric depth of the 4 views (VGGT-Omega, aligned to the calibrated cameras)
ROOT=/n/netscratch/ydu_lab/Lab/akiruga/mv_infer/v1_heldout15
$PY lightstage/inputs/vggt_dynamic_clouds.py --root $ROOT --out $ROOT/clouds_native --native

# 2. per-case rig: input cameras + table plane  -> /tmp/akiruga_lightstage/cases/<scene>_<src>_t12/rig.json
cd /tmp/akiruga_lightstage
CASES=145:gt,145:pred_tokens,145:pred_tiled,000:gt,000:pred_tokens,000:pred_tiled $PY prep_lightstage.py

# 3. Lyra flights (one process per GPU; the model loads once per process)
cd /tmp/akiruga_lyra
SCRIPT=lyra_lightstage2.py ./run_lyra.sh 0 --cases 145_gt_t12 145_pred_tokens_t12 145_pred_tiled_t12 --use_dmd &
SCRIPT=lyra_lightstage2.py ./run_lyra.sh 1 --cases 000_gt_t12 000_pred_tokens_t12 000_pred_tiled_t12 --use_dmd &
wait
```

Output per case in `/tmp/akiruga_lyra/out2/<case>/`:
- `rig2.json`: the 64 dome cameras (OpenCV c2w, metres, calibrated frame), the target point, up vector and radius.
- `flight_k/video.mp4`: Lyra's output for flight k, 416×832 at 16 fps, 81–241 frames.
- `flight_k/traj.npz`: the requested camera for every frame (`w2c`, `intrinsics`, and `base`, the re-basing transform
  that makes real view k the identity).
- `flight_k/stage_frames.json`: which frame of the flight lands on which of the 64 dome cameras.
- `frames/ls_000..063.png`: the 64 light-stage images.

`/tmp` is node-local and is lost when the job ends, so copy `out2/` somewhere permanent.

## How it works

**Rig** (`make_rig` in `lyra_lightstage2.py`):
- 64 cameras: 3 rings × 20 at 20 / 40 / 60° elevation, plus 4 at 75°.
- Radius 1.15 m around a target 0.20 m above the table, all looking at the target. "Up" is the table normal from step 2.
- Earlier versions aimed just above the table at the real cameras' radius (about 0.84 m). That put the arm out of frame,
  or a camera inside the robot.

**Flights** (`assign`, `flight`):
- One flight per real view, covering the dome cameras nearest to it in azimuth.
- Frame 0 is the real camera. A lead-in (position lerp + rotation slerp) reaches the first dome camera, then the path goes
  ring by ring, serpentine in azimuth, along the dome surface.
- Consecutive frames are at most 3° apart in position or rotation.
- Length is padded to 1 + 80k frames, because Lyra generates in 80-frame chunks.

**Seeding** (`run_flight`):
- All 4 real views go into Lyra's multiview spatial cache (`--multiview_ids 0 1 2 3`), with the flight's own view first.
- Each seed is its VGGT-Omega image crop, its metric depth and its calibrated pose. The world is re-based so the flight's
  real view is the identity.
- Depth is already metric, so `--pose_scale 1.0 --no-use_moge_scale`.
- Other settings: `--num_retrieval_views 4`, resolution 416×832, `--use_dmd` (4-step DMD LoRA), default seed.
- Per-scene text prompts are in `lyra_lightstage.CAPTIONS`; new scenes need one.
- Vanilla Lyra can only seed from frames of the trajectory itself. `patch_lyra_multiview_seeds.py` makes
  `run_lyra2_sample()` take the seeds from `mv_video` / `mv_depth` / `mv_w2c` / `mv_K` in the data batch instead.
  It's applied by `setup_node.sh` to Lyra commit `9fffc9a`.

## Notes

- The full 35-step sampler (drop `--use_dmd`) is sharper, but on scene 145 it invented a much larger robot arm of the
  wrong design, and it takes about 9 min per 80 frames. DMD was used for all results.
- On a shared node, kill any GPU reservation **by PID**. `pkill -f <pattern>` also kills your own shell when the pattern
  appears in its command line.
