# RGB scene and motion experiments

Main target: one observed RGB image plus a task → world-model-generated
multiview task sequence → agent-built shared 3D scene and motion → frozen robot
commands → a separately constructed evaluation simulator. Recorded full-video
reconstruction is a diagnostic for this pipeline, not the main deployment input.

The code in this directory implements RGB inference, source-scene fitting,
comparison configuration, frozen planning and independent execution. The block
pipeline has executed in an independent proxy scene, but six frozen RGB-only
plans failed to grasp the object. The world-model-to-successful-execution goal
has not been demonstrated by these experiments.

- [Protocol and staged comparison matrix](README_protocol.md)
- [Strict single-image/task forecast adapter and predicted-depth integration](README_strict_forecast.md)
- [Shared block geometry from generated forecast views](README_forecast_geometry.md)
- [Native SAM3 forecast tracking](README_forecast_tracker.md)
- [Completed block tracking and fresh-agent comparison](README_block_forecast_results.md)
- [Matched forecast agent packets](README_representation_forecast.md)
- [Sealed forecast-agent evaluation](README_forecast_agent_comparison.md)
- [Generated-evidence contact phases](README_forecast_contacts.md)
- [Four-view and 3D forecast render](README_forecast_render.md)
- [First-image-only fixed deployment](README_fixed_deployment.md)
- [Offline replay of frozen native execution](README_registered_replay.md)
- [External baseline installation and launch adapters](README_external_baselines.md)
- [Source-only object initialization](README_source_geometry.md)
- [Source-only refinement experiments](README_source_refinement.md)
- [Fixed-registration TCP planning](README_registered_plan.md)
- [Frozen execution in an independent reference scene](README_execution.md)
- [Generated-camera consistency diagnostic](README_camera_audit.md)
- [Fresh RGB-only MANO fitting](README_rgb_hands.md)
- [Native URDF export to the shared actor format](README_urdf_export.md)
- [SAM3D shape-prior comparison](README_shape_prior.md)
- [Matched UniDepth / DA3 Metric experiment](README_depth_comparison.md)
- [Posthoc sensor-depth and confidence audit](README_confidence_audit.md)
- [Independent HO-Cap hand, object and camera evaluation](README_hocap_evaluation.md)
- [Real RGB representation study and training boundary](README_representation_study.md)
- [Frozen real-video training / held-out split](README_real_video_split.md)
- [Native Pi4 geometry and motion frontend](README_pi4.md)
- [Independent Pi4 depth and camera-drift evaluation](README_pi4_evaluation.md)
- `rgb_frontend.py`: isolate one recorded RGB camera, then predict metric depth
  and camera intrinsics with an existing local UniDepthV2 checkpoint.
- `keyframe_compression.py`: compress saved fitted trajectories and measure the
  interpolation error. This is separate from generating only five keyframes.
- `run_unidepth.sbatch`: bounded GPU job using existing dependencies/checkpoints;
  choose account and partition on submission. No automatic download.
- `sam3_rgb_masks.py` and `sam3d_rgb_asset.py`: fresh segmentation candidates and
  image-conditioned assets using existing author checkpoints. Candidate row IDs
  are not persistent tracks; generated meshes are not validated geometry.
- `world_model_views.py`: source-video-conditioned helper views with estimated
  intrinsics and explicitly designed virtual cameras. Generated-camera accuracy
  is not assumed from the conditioning requests.
- `fixed_deployment.py`: one declared virtual robot placement shared across
  variants. This is not a measured camera-to-robot calibration. The independent
  evaluator applies the identical transform to its reference without snapping.
- `independent_scene.py`: attaches evaluator-only geometry/cameras after the
  command submission has been sealed. The present reference is an RGB-D
  reconstruction proxy, not exact ground-truth CAD.

## Fresh RGB-only runs, 2026-11-19

Native public source RGB was isolated before inference: 80 RH20T block frames
at 640×360 and 10 Hz, and 120 HO-Cap bottle frames at 640×480 and 30 Hz.
The isolated arrays contain only RGB and timestamps. No measured depth or
published camera matrices enter the following predictions.

UniDepth predicted metric depth and intrinsics on both sequences. Its measured
internal totals were 54.4 s (block) and 78.5 s (bottle), including model loading
but excluding Python import/startup. SAM3 then produced candidates on every
frame: 217.4 s and 325.4 s of inference, respectively. The bottle had one object
and table candidate on every frame. The block had multiple object candidates
on every frame and no table candidate on 51/80 frames, requiring explicit
association and pooled table evidence.

Fresh image-only SAM3D assets were generated for both objects. Model inference
was 43.5 s for the block and 17.1 s for the bottle; cold initialization took
about 521 s, and the complete two-asset job took 11 min 5 s. These timings must
not be conflated with warm optimization time. Initial wrapper projection IoU
was 0.670 and 0.316, respectively; this is a predicted-mask sanity check, not
ground-truth 3D accuracy. SAM3D scale/pose still require model-derived fitting.

The separate cuboid/translation baseline exposed these failures:

| Depth evidence | Supported poses | Estimated dimensions | Final bottom below estimated table |
| --- | ---: | --- | ---: |
| All valid points | 80/80 | 44.1 × 42.8 × 74.9 mm | 37.9 mm |
| Global top-half confidence rank | 63/80 | 44.5 × 44.6 × 76.1 mm | 39.3 mm |

The confidence filter drops frames 40–56, during manipulation/occlusion.
These table discrepancies are internal inconsistencies, not measured errors
against GT. Neither result is execution-ready. A separately declared contact
policy left geometry exactly unchanged but produced split grasp intervals:
33–38, 40–58 and 60–63 for all points; 32–39 for rank filtering. Missing grasp
evidence is not filled merely to permit a plan.

The RH20T block recording appears once in the world model's **training**
manifest and not in its test manifest (split hash bucket 9051, test threshold
200/10000). It remains a development/matched-ablation case, not held-out
generalization evidence. HO-Cap is outside that RH20T fine-tuning dataset;
pretraining exposure is not established.

Follow-up source-only fitting found temporal drift in the predicted **static
table**, not just errors on the moving block. Frame-zero-anchored multiplicative
depth correction reduced held-out background-plane residual from 17.20 to
3.65 mm. It also removed the planner's apparent pregrasp-motion violation.
An independent RGB proximity/co-motion estimator found one continuous grasp
hypothesis at frames 29–67, while retaining conflicting 3D-distance evidence.

Joint shape/translation fitting with soft support improved source silhouettes
but left 6–9 mm penetration near release. A separate optimizer parameterized
every frame with nonnegative table clearance, keeping the same depth,
silhouette, temporal and support losses. Both variants then compiled through
unchanged geometric planner guards. Mean support residual fell to 0.84/1.36 mm,
while depth surface residual increased from about 2.4 to 3.0 mm and the inferred
box became flatter. These are source-model consistency metrics, not GT shape
accuracy or execution success. No endpoint was moved after planning.

Using only the fresh SAM3D canonical mesh's aspect ratios (not its camera pose
or metric scale) restored approximately cubic 42 mm dimensions. This worsened
predicted-depth residual while improving support agreement. The fitted object
remains a cuboid envelope with fixed orientation, not a tracked SAM3D mesh.

Four plans—free/aspect-constrained shape crossed with within-object/soft
confidence ranks—were then sealed and each executed exactly once. All failed
to grasp or lift the independent reference block. Initial reconstructed-centre
error was approximately 87–92 mm; the gripper closed in empty space. TCP
tracking at the end of closure was about 0.042 mm, showing that the command
target itself was wrong. No plan, geometry or controller was updated from these
outcomes. The reference is a separately reconstructed RGB-D proxy, not exact
CAD ground truth, and its unchanged support has a 1.435-degree tilt plus an
initial 9.284 mm object/plane gap. No individual method was aligned or snapped
to this reference. The deployment transform was one shared virtual setup,
not a measured physical robot calibration.

The post-run audit found roughly 84–91 mm underestimated source-camera depth,
despite centre projections agreeing within about 1.5 pixels. The common rigid
deployment preserves the placement discrepancy. Against the proxy's oriented
bounding dimensions, sorted-dimension error remained approximately 7–8 mm
for every variant and estimated volume remained 59–62% of the proxy volume.
The shape prior therefore did not demonstrate a material size or placement
improvement. These evaluation measurements were withheld from the concurrent
depth-comparison and hand-fitting work.

Two matched all-point UniDepth controls subsequently executed once each and
also missed grasp/lift: six new executions in total, zero successful grasps.
Both DA3 Metric variants were rejected before execution for 10.140/10.093 mm
pregrasp drift against the fixed 10 mm guard. This marginal rejection is not
physical failure. Posthoc scoring against the separate proxy found approximately
33 mm DA3 blocks, 16.3 mm sorted-dimension error and 268/217 mm initial/final
centre error; the all-point UniDepth control had about 7.65 mm size error and
87/86 mm centre error. These are single-clip proxy comparisons.

Fresh 25-frame world-model helpers completed with source-latent clamping checked
at every conditioned forward pass. However, matched static backgrounds moved
only 0.27/0.30/0.78 pixels, where the requested cameras predict approximately
21–24 pixels. DA3's unposed estimates also gave almost zero source/helper
baseline for the first two generated views. Five generated keyframes did not
fix this: the first two helpers remained source-like, and the third's matched
motion disagreed with its requested camera. Requested poses therefore cannot
be treated as verified extrinsics for fusion. These diagnostics use predicted
depth/K and correlated background matches, not GT camera measurements.

A 17-frame prefix diagnostic also tested text/history guidance 2/1, 2/0 and
0/0 with the same camera requests and seed. All three remained source-like in
the static-background audit. Guidance 0 is still conditioned on the supplied
text; it is not a null-text condition. This does not establish the root cause
of weak camera adherence or justify treating the requests as actual poses.

First-source-frame conditioning, matching the author's single-view training
mask, also failed to establish camera adherence: first two helpers moved matched
background points about 0.44/0.46 pixels versus 15.34/12.36 requested at time
zero. The third had only nine retained matches and is inconclusive. Full-clip
camera-design ancestry remains, so this is a conditioning diagnostic, not a
strict forecasting result.

The 13-frame HO-Cap bottle generation completed with its own verified task-text
embedding. Matched backgrounds moved 0.37/0.36/0.60 pixels versus approximately
60/44/59 requested. Human/object appearance changes do not supply the missing
viewpoint change. These helpers are therefore not fused under their requested
poses as if they were calibrated observations.

A subsequent block camera-control experiment isolated intrinsics K from camera
poses C in a two-by-two comparison. The same first source image, 17-frame horizon,
checkpoint, seed and sampler produced visibly different viewpoints with the
published rig poses under either K. Published K alone with the virtual poses
still produced near-copies: helpers 1/2 shifted 0.459/0.474 pixels versus
16.03/12.36 requested. On this case, our virtual pose requests are therefore an
actionable contributor to the weak camera response. This is a calibrated-camera
diagnostic, not a strict RGB-only improvement. The wide-view outputs have too few
reliable feature matches to verify camera accuracy; they have not been fused into
a reconstruction or used for planning. See [the complete camera comparison](README_camera_audit.md).

Fresh RGB-only MANO initialization and the actual RobotFitter port have also
run on both clips (100-iteration diagnostic budget). The bottle and block port
stages took 297.0 and 172.7 seconds, respectively. Mean reprojection error to
predicted RGB landmarks fell from 7.73 to 2.74 pixels and 9.68 to 2.48 pixels.
This is not a hand-pose GT metric: bottle finger contact/handedness remain
uncertain, and the relevant block hand lacks accepted evidence after frame 67.
The first side-selection diagnostic tested both licensed hand priors but bound
only the selected prior in its manifest; the current adapter binds both and
adds an explicit soft detector-handedness-prior alternative. Earlier outputs
are preserved rather than relabeled.

The right-hand variant reduced source landmark error to 1.744 pixels. Its
fresh SAM3D bottle mesh was then tracked with the same 13 fitting frames and
shape/pose budgets. RGB identity tracking at the native 120 frames plus explicit
predicted-hand occlusion recovered 13/13 associations versus 4/13 for the
original 3D association gate. Common source-mask IoU rose from 0.306 to 0.779;
on the four already-supported frames it changed only from 0.774 to 0.795.
This primarily restores the missing lift segment. The combined change does not
isolate depth bias from sparse-frame motion or occlusion. The actual SAM3D
nozzle topology is retained; contact remains unknown and no bottle plan is
declared ready. See [the detailed bottle comparison](README_rgb_hands.md).

Independent HO-Cap annotations confirm that the hand's articulation improved:
mean corresponding-vertex error after subtracting each hand's own wrist fell
from 27.96 to 9.40 mm. This diagnostic does not rotate or rescale either hand.
Absolute vertex error worsened from 62.93 to 75.99 mm, with approximately
80 mm wrist-position error remaining. Better articulation therefore did not
solve global placement. The dataset MANO surface was reproduced from its
original parameters and verified against camera-frame joint annotations;
landmark-definition differences are recorded separately.

The exact HO-Cap bottle CAD/pose comparison likewise distinguishes tracking from
metric accuracy. On the same four supported keyframes, v1 to v2 centroid error
changed from 96.04 to 102.31 mm and unaligned symmetric surface-sample distance
from 66.37 to 69.56 mm. Recovering the remaining nine keyframes does not establish
better placement on the common frames. Sorted bounding-dimension error improved
from 15.33 to 12.20 mm, but minimum thickness remained 42.05 mm versus the CAD's
66.53 mm (v1: 32.62 mm). This isolates an extent improvement, not complete shape
accuracy. Held-out camera overlays use unchanged
frozen geometry and publisher calibration only in the evaluator. The historical
observation bundle differs from the original cameras by one common rigid world
frame; its relative camera transforms and intrinsic matrices were verified.

An evaluator-only HO-Cap audit compared these frozen metric depth predictions
with the corresponding measured depth, without scale correction. UniDepth's
global top-half confidence ranks retained only 12.7% of predicted hand pixels
and 27.6% of bottle pixels. Mean absolute depth error increased from 95.8 to
124.0 mm for the hand and from 198.5 to 232.7 mm for the bottle. Mask-interior
erosion reduced the errors but did not reverse this selection failure. These
ranks describe relative, scale-invariant prediction error; they do not certify
absolute metric scale. This is one clip, not a general calibration benchmark.

On the same bottle clip and all valid pixels, DA3 Metric improved bottle depth
error from 198.5 to 88.8 mm and table error from 108.0 to 38.2 mm, but worsened
hand error from 95.8 to 106.4 mm. Together with the opposite block comparison,
this does not support choosing one depth source globally. All sensor comparisons
are posthoc evaluation only and were withheld from active fitting workers.

## Legacy independent-scene control, 2026-11-19

After removing render-only initialization failures, two previously frozen
calibrated plans each executed once against the same recorded RGB-D reference
proxy. Commands, controller and physics were unchanged. The earlier plan had
28.9 mm XY endpoint error and failed the predeclared 25 mm destination gate;
the existing stable-support variant had 20.0 mm XY error and passed all seven
predeclared grasp/lift/release/stability/destination/support gates. Orientation
errors were 35.6° and 3.9°. The videos render logged robot/object states offline,
without another physics execution.

This is one clip and two legacy calibrated variants. It is **not** a new
RGB-only success, exact real-world ground truth, or evidence of generalization.

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

[UniDepthV2's own documentation](https://github.com/lpiccinelli-eth/UniDepth/blob/main/assets/docs/V2_README.md)
describes this output as estimated scale-invariant log error, ranked relative to
one input. It should not be interpreted as certainty about absolute metric
scale. The separate sensor-depth audit scores frozen predictions in their
original metric units and does not rescale them to the reference.

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
