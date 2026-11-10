# Real RH20T, v2: robot + annotated objects (SAM 3, SAM 3D, multiview depth)

What changed from v1 on real data:
- v1 found objects from point-cloud residuals and decided automatically what moved.
- v2 takes "what is moved, and where" from a per-episode annotation, written by looking at the frames.
- SAM 3 says where that object is in every view and frame.
- SAM 3D gives an initial mesh for non-box objects.
- Multiview depth plus silhouettes give metric shape and 6-DoF pose.

The robot is still fitted from multiview depth alone. The logged joints are used only to recalibrate the cameras and for evaluation.

## Pipeline (per episode, `real_v2.py`)

| step | what | file |
|---|---|---|
| 0. prep | 5–6 fixed RealSense views, sensor depth, RH20T calibration; 20 frames over the manipulation span ±2 s | `real/prep_rh20t_episode.py --T 20 --pad_s 2` |
| 1. cameras | per-camera 6-DoF ICP of each view's depth against the URDF posed at the logged joints (the calibration target): 1.5–4° and 10–55 mm corrections, median robot–depth residual 8–20 → 3.5–5 mm | `real_v2.recal` |
| 2. robot | table self-calibration, then URDF fit from multiview depth (joints not given): frame-0 search, per-frame tracking, backward pass, joint refinement, depth-scale self-calibration | `artirecon/fit_robot.py` |
| 3. annotation | one JSON per episode: object name, SAM 3 text prompt, role (moved / static target), shape class (box / hull / sphere), optional push mode, one click on the object in one view at frame 0 | `real/v2/<ep>/ann.json`, sheets from `ann_sheet2.py` |
| 4. masks | SAM 3 (`facebook/sam3`, transformers) text-prompted instance masks in every view and frame | `seg_sam3.py --many` |
| 5. assets | SAM 3D Objects mesh from the clicked mask (hint view) for hull objects; decimated to 20k faces | `sam3d/sam3d_batch.py`, `sam3d/decimate.py` |
| 6. objects | track, shape, pose (below) | `artirecon/objects_v2.py` |
| 7. metrics | robot vs logged joints; object mask IoU (fitted views and a held-out view), depth agreement, support contact, table penetration, grasp interval vs logged gripper | `real_v2.py`, `summarize_v2.py` |

### Objects (`objects_v2.py`)
1. **Instances in 3D.** Every SAM 3 mask of the prompt is lifted with its view's depth. Points on the fitted robot surface and the table are removed, which gives the mask's 3D centre and size.
2. **Tracking.** It starts from the clicked instance (the smallest mask under the click) and runs a grasp state machine on the fitted robot:
   - Each frame, every view's instance nearest the predicted position is taken. Instances with a mismatched 3D size (a container around the object, or a neighbour) are rejected.
   - If nothing is near, a wider re-detection gate is tried, which must agree across at least 2 views.
   - **Rest → held:** a closing event (the gripper was open within 3 frames and is now at a width above 6 mm), with the fingertip point within 4 cm of the object's observed points. The fingertip point is 15 cm from the hand: the real fingers are longer than the URDF's.
   - **Held:** the object is predicted rigidly with the hand.
   - **Release:** the gripper opens.
   - **Push objects** are predicted to follow the hand in the table plane instead.
3. **Shape.** It comes from the rest frames, all views fused, after a density filter for flying pixels.
   - Support height comes from the depth ring around the masks, or from the object's own lowest points when it rests on something such as a scale, rack or shelf.
   - Box: gravity-aligned, footprint from the upper part of the object (mask edges bleed depth onto the support).
   - Hull: a convex footprint extruded.
   - Sphere.
   - SAM 3D mesh: orientation from SAM 3D's predicted camera pose mapped to world, scale and position from the fused points.
   - Then a **silhouette shape refinement**: per-axis scale plus pose against the SAM 3 masks of up to 4 rest frames.
4. **Pose per frame (6-DoF).** The terms:
   - silhouettes: model points not occluded by the observed depth must fall in the instance mask, and the mask must be covered by the projected model (2D chamfer both ways);
   - 3D chamfer to the fused mask points;
   - multiview depth z-test;
   - table non-penetration (hard);
   - resting on the support when not held;
   - rigid-in-hand prior while held.

## Results (25 RH20T cfg5 episodes, 23 tasks)

The full per-object table is in [`results_real_v2/SUMMARY.md`](results_real_v2/SUMMARY.md), with the held-out-view version in `results_real_v2/SUMMARY_ho0.md`. There is a page with videos and crops at `results/real_v2/site/index.html`.

**Robot.** It is fitted from multiview depth only, and compared against the logged joints. Each robot fit takes about 3–5 min per episode.

| | value |
|---|---|
| mean surface error | 2.8 mm (median 2.6, range 2.0–4.0 per episode) |
| gripper | 6.3 mm |
| joints | 1.1° |
| v1 on the same first episode | 13 mm (10 frames, no per-frame recalibration) |

The few frames above 9 mm all fall where the logged gripper width jumps open or shut between frames. The logged state leads the physical fingers there, so those are ground-truth timing artefacts.

**Objects.** There are 25 moved objects and 11 static targets. The metrics:
- **Silhouette F@2px** is the share of each silhouette within 2 px of the other: rendered model vs SAM 3 mask, at 640×360. It is size-independent, which matters because many objects are only 15–40 px across.
- **Held-out** means view 0 was excluded from tracking, shape and pose fitting, then used to score.

| | fitted views | held-out view |
|---|---|---|
| moved: silhouette F@2px (mean / median) | 0.87 / 0.89 | 0.83 / 0.85 |
| moved: objects with F@2px ≥ 0.8 | 23 / 25 | 17 / 25 |
| moved: mask IoU (mean / median) | 0.66 / 0.68 | 0.46 |
| static targets: mask IoU | 0.70 | 0.64 |
| held state detected | 21 of 25 moved objects (2 of the 25 are pushes, never held) | |
| grasp interval vs logged gripper (IoU over frames) | 0.56 | |

Best cases (IoU / F@2px):
- plate put upright into a rack: 0.88 / 0.97;
- glass lid onto a pan: 0.80 / 0.92;
- green cups picked up: 0.76–0.78 / 0.93;
- cube pushed: 0.74 / 0.93;
- roll taken off a shelf: 0.73 / 0.90;
- cube moved between trays: 0.70 / 0.90.

Hardest cases:
- **Thin objects:** a few pixels wide at 360p, and their depth is mostly background. Knife 0.33 / 0.59, pool cue 0.47 / 0.74, marker 0.49 / 0.81.
- **Missed grasps:** the roll (held almost fully open) and one plate. The object is still tracked by its masks.
- **Support contact for plates around racks:** the support height estimated from the surrounding depth is unreliable there, so the contact numbers are large, but the silhouettes fit.

What helped, measured on the episodes where each change was made:

| change | effect |
|---|---|
| per-frame camera refinement on the robot + 20 frames | robot 13 → 3.2 mm |
| silhouettes as the main shape/pose signal (depth bleeds at object edges) | cube IoU 0.41 → 0.60, box extents 41–53 mm → 25–36 mm |
| grasp state machine on the fitted gripper (closing event, fingertip reach, release on opening, closing on nothing, or the object staying while the hand leaves) | grasp interval IoU 0.30 → 0.56 |
| smallest SAM 3 mask under the click, 3D-size gating of candidates | handset 0.06 → 0.61; cube inside a tray 0.17 → 0.44 |
| open containers (trays, boxes, racks) instead of solid boxes | cube in tray 0.44 → 0.70 |
| SAM 3D initial asset, competing with a 2.5D hull (best rest-frame fit wins) | phone base 0.75 → 0.78 (depth agreement 0.26 → 0.45); handset correctly falls back to the hull |
| support from the object's own lowest points when it rests on something | marker on a pen holder 0.27 → 0.43 |

Annotation took one look per episode at two views (first and last frame) and one click per object. The annotations are in `real/v2_annotations/`.

## Commands
```
# data (no archives stored): colour + robot state from HF Malak-Mansour/RH20T, depth from introvoyz043/RH20T
real/stream25.sh color; real/stream25.sh depth; real/prep25.sh
python ann_sheet2.py <ep>                         # frames 0/last, 2 views, pixel grid -> write real/v2/<ep>/ann.json
python seg_sam3.py --many <list of real/v2/<ep>>  # SAM 3 masks
sam3d/run_sam3d.sh ... | python sam3d/sam3d_batch.py <ep list>   # SAM 3D assets (sam3d env)
python real_v2.py <ep> --stage robot              # cameras + robot
python real_v2.py <ep> --stage objects            # objects + metrics + overlays
python real_v2.py <ep> --stage objects --holdout 0 --tag _ho0     # held-out view evaluation
python summarize_v2.py real/v2/final25.txt [--tag _ho0]
```
GPU notes:
- test2's pytorch3d is built only for sm_90. On other GPUs (here an A100 MIG slice), `artirecon/p3d_compat.py` replaces `knn_points` with a Triton nearest-neighbour kernel (K=1), and rendering uses a GPU point-splat z-buffer (`artirecon/splat.py`).
