# Articulated 4D reconstruction of manipulation scenes

Given a manipulation video, recover the parts that move as an explicit, simulator-ready 3D model per frame:
- the robot as its articulated URDF (joint angles plus base pose);
- the manipulated object;
- the target it interacts with;
- the joint of an articulated object (axis, travel, limits).

The fit runs in 3D against multiview depth and cameras. A pixels-only render-and-compare version is kept as an ablation. The output (`scene.json`) loads into SAPIEN, so it can be used for replay, physics checks or inverse dynamics.

Test scenes:
- ManiSkill PegInsertionSide ("standard").
- ManiSkill drawer opening ("drawer").
- One real RH20T Franka episode (cfg 5, 5 RealSense views).

Each scene is run on ground-truth inputs and on the multiview world model's outputs (predicted depth and full rollouts).

**Real RH20T, v2 (25 episodes):** see [README_real_v2.md](README_real_v2.md).
- **Robot:** 2.8 mm mean surface error from multiview depth alone.
- **Objects:** what moves comes from per-episode annotation. SAM 3 gives masks and SAM 3D gives initial assets, then multiview depth plus silhouettes give shape and pose.
- **Score:** median silhouette F@2px 0.85, in both fitted and held-out views.

## Pipeline

Times are for the peg scene, sim GT: 4 views, 11 frames, 448×256, one H100.

| # | step | what it does | file | time |
|---|---|---|---|---|
| 0 | input | multiview RGB video, per-frame depth, calibrated cameras (K, w2c) | `artirecon/data.py` | – |
| 1 | lift | back-project every depth map into metric world points | `data.backproject`, `fit_robot.prepare` | < 0.1 s |
| 2 | table | RANSAC plane; foreground = points above it | `fit_robot.prepare` | 0.1 s |
| 3 | robot, frame 0 | slide the URDF base column over the table, then 32 joint hypotheses with Adam; keep the best | `fit_robot.py`, `robot.py` | 43 s |
| 4 | robot, tracking | per-frame multi-start from the previous frame (forward pass), a backward pass, joint refinement, and an explicit search over the base-yaw / joint-1 gauge | `fit_robot.py` | ~7 s / frame |
| 5 | residual | points not explained by the robot are clustered at frame 0; a point is **absent** at frame t if some camera now sees through it | `objects.py` | ~1 s |
| 6 | objects | fully absent → free rigid box (6-DoF per frame). Partly absent → articulated (static parent + prismatic child, axis searched). Static and touched by a moving part → target. Rigid with the hand → robot attachment. Then physics priors (below). | `objects.py`, `scene_fit.py` | ~25 s |
| 7 | export | `scene.json` (parts, joints with axis/limits, keyframe poses), overlays, metrics, SAPIEN build | `export.py`, `evaluate.py`, `sim_replay.py` | – |

Total is about 3 min per 11-frame clip. A fast setting (5k points, 30 steps, 1 warm start; `fast_test.py`) runs at 0.9 s/frame, with robot error going from 23.6 to 27.5 mm on the real clip.

**Objective** (`geom.py`): surface samples of the candidate state are projected into every camera and z-tested against that camera's depth:
- on the surface → attract (robust loss);
- in space the camera sees as empty → penalty;
- occluded → no evidence.

A robust chamfer from observed points to the model is added. There is no rasteriser in the loop, and everything is batched and differentiable (own torch FK from ManiSkill's `panda_v2.urdf`, plus pytorch3d knn).

**Physics priors**:
- Non-penetration: if the peg ends inside a solid target box, the box must have a through-hole, so it is carved (4 slabs, as ManiSkill builds it).
- An open-top container is inferred from surfaces seen inside it.
- Hand attachment.
- Grasp closure.
- GT-free plausibility checks: robot–object penetration, object–table penetration, finger–object gap.

**Predicted-depth extras**:
- Per view/frame depth-scale self-calibration, using the robot as a metric calibration object (`*_cal`).
- A 3D→pixel hybrid that refines joints with soft-silhouette IoU against Grounding-DINO + SAM2 masks (`hybrid.py`).
- For real data, a 6-DoF per-camera extrinsic refinement against the URDF posed at the logged joints (`real_recalib.py`).

## Results

Robot error is the mean surface distance between the fitted URDF and the GT robot. Objects are chamfer distance to the GT mesh. All values are in mm. The full ablation table is in [`RESULTS.md`](RESULTS.md).

| scene / input | robot | objects | articulation |
|---|---|---|---|
| peg, sim GT, 4 views + depth + cams | **1.2** (joints 0.2°) | peg 3.7, target with inferred hole 6.5 | – |
| drawer, sim GT, 4 views + depth + cams | **2.1** (0.65°) | cabinet 12, drawer 20 (box vs open tray) | axis **0.0°**, travel **1.6 mm** |
| real RH20T, 5 views, sensor depth, cams re-calibrated on robot | **13.0** (fingers 2.0) | 3 cm block not recovered | – |
| peg, real RGB + model-predicted depth, + pixel hybrid | 18.8 (27 without pixels) | unreliable | – |
| peg, model A rollout (1 view → 4 views + depth), self-calibrated | 39.8 | peg 75 | – |

Comparisons:

| | 4 views + depth + cams | 1 view + depth + cam | 1 view pixels only |
|---|---|---|---|
| peg sim, robot / peg | 1.2 / 3.7 | 2.8 / 15.4 (occluded gripper) | 23.3 / 131 |
| drawer sim, axis ° / travel mm | 0.0 / 1.6 | 47 / 45 | 127 / 490 |
| real, robot | 23.5 → 15.0 (depth cal) → 13.0 (cam recal) | 15.6 | 100 |

Depth source for the peg scene with 4 views (robot error):

| GT depth | model B depth + hybrid | model A rollout | model B rollout |
|---|---|---|---|
| 1.3 | 18.8 | 39.8 | 110 |

Takeaways:
1. **Metric multiview depth.** With metric multiview depth and cameras, the dynamic parts come out at mm level and articulation is exact.
2. **One view.** One view with depth is enough for the robot (thanks to the kinematic prior), but it loses occluded parts. Pixels alone fail at metric pose.
3. **World-model depth.** The machinery works on world-model depth (robot within a few cm), but object geometry is limited by that depth's 2–3 cm error. Better metric depth from the model is what would close the gap.
4. **Rollouts.** These are forecasts. After frame 0, GT errors measure plan-vs-reality. Agreement with the model's own depth is the meaningful number there.
5. **Real data.** Unmodelled hardware (cables, mounts) and background motion are filtered by two rules:
   - a cluster touching the robot at frame 0 is robot hardware;
   - an object must be touched by the gripper to count as manipulated.

   The 3 cm block is still missed. The calibration is 8 days older than the recording, and the cameras disagree on table height by up to about 2 cm.

## Data and conditions

| source | what | loader |
|---|---|---|
| `gt512` | ManiSkill package observations, 512², 5 keyframes | `load_gt512` |
| `gt` | the world model's GT frames of the same episode, 448×256, 11 frames at 1 Hz | `load_dense("gt")` |
| `est_B` | real RGB of all 4 views, depth predicted by world model B | `load_dense("est")` |
| `roll_A` / `roll_B` | world-model rollout from view-0 frame-0 only: generated other views, future frames, depth | `load_dense("roll")` |
| `real` | RH20T episode, 5 RealSense views, sensor depth, logged joints as GT | `load_real` (prepared by `real/prep_rh20t_episode.py`) |

| input condition | meaning |
|---|---|
| `mv` | all views + depth + cameras |
| `mono_depth` | view 0 + depth + camera |
| `mono_rgb` | view 0 pixels only (unknown camera, 60° FOV guess, scale from the URDF) |
| `*_cal` | with robot-based depth self-calibration |
| `*_hybrid` | then refined with pixel silhouettes |
| `*_recal` | real cameras re-calibrated on the robot |

The paths default to the FASRC locations and can be overridden with environment variables:

| variable | default | contents |
|---|---|---|
| `AR_PKG` | `.../shared-maniskill/maniskill-reconstruction-scenes-20261101` | ManiSkill scene package (GT obs, meshes, states) |
| `AR_GEO` | `/n/lab_storage/ydu_lab/Lab/akiruga/geo_runs` | world-model eval outputs (`eval_s7000/A`, `eval_s9000/B`) |
| `AR_MSR` | `/n/lab_storage/ydu_lab/Lab/akiruga/maniskill_render` | dense ManiSkill renders (GT frames) |
| `AR_REAL` | `./real` | prepared real episodes (`obs.npz`, `meta.json`) |
| `AR_PANDA_URDF` | ManiSkill's `panda_v2.urdf` in the `maniskill` env | robot URDF + meshes |

## Usage

Environments:
- `test2` (torch + pytorch3d; needs an sm_90 GPU) for fitting.
- `maniskill` (SAPIEN 3) for `sim_replay.py`.

Always set `PYTHONNOUSERSITE=1`.

```bash
PY=/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python

# fit -> evaluate -> export, many jobs in one process; outputs results/<scene>/<tag>/
$PY driver.py standard:gt512:mv drawer:gt512:mv standard:est:mv_cal:B real:gt:mv_cal   # <scene>:<source>:<condition>[:run A|B]
sbatch run_driver.sbatch standard:gt:mv                       # same on gpu_requeue / kempner_requeue
$PY driver.py --objects <jobs>                                # re-run only the object stage on saved robot fits
$PY run_scene.py gt512 standard mono_depth                    # one job

# pixels-only ablation and 3D+pixel hybrid
$PY run_masks.py gt512 standard                               # Grounding-DINO + SAM2 masks, view 0
$PY run_rgb.py gt512 standard                                 # pixels-only fit
$PY run_masks_allviews.py est standard B                      # -> results/masks/standard_est_allviews.npz
$PY hybrid.py results/standard/est_B__mv results/masks/standard_est_allviews.npz [w_pix]

# real data
$PY real/prep_rh20t_episode.py --color_dir <rh20t color> --depth_dir <rh20t depth> --cfg 5 --out real/<ep> --T 10
$PY real_recalib.py <ep>                                      # -> real/<ep>_recal
REAL_EPISODE=<ep>_recal $PY driver.py real:gt:mv_cal         # fit on the re-calibrated rig
$PY fast_test.py                                              # speed/accuracy of the fast setting

# simulator replay + physics checks (maniskill env)
/n/lab_storage/ydu_lab/akiruga/.conda/envs/maniskill/bin/python sim_replay.py results/standard/gt512__mv \
    --cameras $AR_PKG/standard/observations_gt.npz

# reports
$PY build_site.py                                             # site/index.html: table, 3D viewer, overlays
$PY deck/deck_media.py && $PY deck/deck_clips.py && python deck/build_deck.py   # deck/articulated_recon.pptx
```

Each `results/<scene>/<tag>/` holds:
- `fit.pkl`, `scene.json`, `metrics.json`, `log.txt`;
- `viewer.js`, `overlay.mp4`, `overlay_grid.jpg`;
- `sim/` (renders + physics report) where the replay was run.

## Files

| file | role |
|---|---|
| `artirecon/data.py` | sources → common `Obs` (rgb, depth, K, c2w, steps), back-projection |
| `artirecon/robot.py` | URDF parser, torch FK, surface sampling, mesh export |
| `artirecon/geom.py` | multiview z-test depth objective + robust chamfer |
| `artirecon/fit_robot.py` | table, base search, multi-start tracking, refinement, depth self-calibration |
| `artirecon/objects.py` | residual clustering, presence test, free / articulated / static / attachment fits, hole inference |
| `artirecon/scene_fit.py` | robot + objects → full scene state |
| `artirecon/masks.py`, `fit_rgb.py` | text-prompted masks; pixels-only soft-silhouette fitter |
| `artirecon/evaluate.py`, `gt.py` | GT loading, metrics (surface distance, chamfer, axis/travel, IoU, depth agreement), plausibility |
| `artirecon/export.py`, `render.py` | `scene.json`, viewer data, overlays |
| `driver.py`, `run_scene.py`, `eval_scene.py`, `test_objects.py` | job runners |
| `hybrid.py`, `run_masks*.py`, `run_rgb.py` | pixel ablation and hybrid |
| `real/prep_rh20t_episode.py`, `real_recalib.py`, `fast_test.py` | real-data prep, camera recalibration, speed test |
| `sim_replay.py` | SAPIEN 3 build, kinematic replay renders, hold / joint-drive checks |
| `build_site.py`, `site_template.html` | comparison site |
| `deck/` | slide-deck media and pptx builder |

## Limitations

- Objects are boxes. The GT drawer is an open tray with a handle, so its chamfer is dominated by representation, not pose. Its extents are within about 1 cm, and its axis and travel are exact.
- Hole clearance (3 mm) is assumed. Hidden geometry is inferred from motion and contact, not observed.
- Joint limits are known only over the observed range.
- On real scenes, small manipulated objects are not yet recovered (calibration drift). This is the main open item.
