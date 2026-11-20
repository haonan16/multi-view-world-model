# Block forecast: tracking and fresh-agent comparison, 2026-11-20

Four fresh agents completed a matched source/helper × first-image/sequence
initialization comparison. **The extra sequence initialization did not reduce
agent time or recover the pickup trajectory.** This is one attempt per cell on
a development clip, not a generalization or execution result.

## Method and input boundary

The input was one observed RGB image and the task “Pick up the small block on
the left and move it to the right.” The saved world model generated 17 times
over 5.33 seconds and four views. Only view 0 at time 0 is observed. Future
source-view frames are generated too. This checkpoint/run exported RGB; depth
and cameras came from DA3, initial metric scale/intrinsics from UniDepth, and
masks from SAM3. It did not export jointly generated depth, masks or normals.

Every agent received the same first-image 3D seed. Within each evidence branch,
the two agents received byte-identical RGB/depth/mask evidence and the same
tools. The initialized condition additionally received a numerical fit's shared
shape, sequence poses and future scale estimates. Consequently this tests the
value of **extra sequence initialization**, not 3D versus no 3D. The source-only
condition still sees the generated source-camera forecast; it is not a
no-world-model baseline.

Each fresh Codex agent had the same model, prompt, 600-second budget and
40-tool-call limit. Inputs were immutable and isolated from the repository,
other agents, network tools and private evaluation. There was no retry. All
four used their own RGB-corner fitting code rather than the supplied joint
optimizer; all four inspected RGB and rendered overlays. Generated numerical
depth scales remain hypotheses, not measured calibration.

## Measured agent results

| Evidence | Initialization | Agent time | Extra initialization time | Supported times | Source mask IoU | Helper mask IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Source forecast | First-image seed | 354.5 s | — | 11/17 | 0.845 | 0.373 |
| Source forecast | Plus sequence fit | 429.6 s | 7.8 s | 11/17 | 0.830 | 0.403 |
| Source + helpers | First-image seed | 283.7 s | — | 11/17 | 0.598 | 0.512 |
| Source + helpers | Plus sequence fit | 330.5 s | 18.9 s | 11/17 | 0.618 | 0.559 |

All scores use the common supported intersection, frames 0–10, against the
same fixed source/helper references. Source scores have 11 evidence rows;
helper scores have 21 of 33 possible rows because unknown masks are unscored.
IoU is agreement with predicted segmentation, **not true geometry accuracy**.
Source/helper depth-camera estimates disagree, so neither table column alone
establishes a better reconstruction. No hypothesis significance is claimed.

The agents retained a stationary block through 3.33 seconds and marked the
remaining six frames unsupported. They changed the initial geometry but did
not establish a supported grasp, lift or destination. Both initialized agents
took longer; all four largely rebuilt geometry from the first-image seed and
visible RGB corners. Total pipeline latency is longer than these agent-only
times. No simulator execution was attempted for these four results.
Both initialized agents inspected the sequence fit and restarted from the
common first-image seed. This measures availability of an initialization;
enforced warm-start refinement has not been tested here.

## Tracking and world-model diagnostics

- The first native tracker attempt seeded a background object. High tracker
  presence did not detect that mistake. This version is retained as a failure
  diagnostic and excluded from the agent inputs.
- The corrected initialization associates the source object using existing
  mask-only criteria and transports its seed to helpers through robust static
  image matches. Two helpers passed; the third failed the spatial-support gate
  and remained unknown. Native tracking inference took 10.94 seconds for
  51 images, excluding job startup/model loading.
- Late tracking still fails: at 5.33 seconds, 97.3% of the source “object” mask
  overlaps the predicted hand. The agents rejected this as block-motion
  evidence. Unsupported poses are hidden in the comparison video.
- Nine numerical fits tested three DA3 branches (source, unposed helpers,
  requested-camera helpers) crossed with baseline, soft support and an initial
  RGB-rest policy. Every fit hit the 70-evaluation limit. The rest policy found
  no anchors and matched baseline; soft support reduced negative clearance
  without establishing correct shape or contact.
- The helper baseline inflated the initial block to approximately
  81 × 69 × 42 mm, versus 50 × 48 × 32 mm for source-only fitting. Neither is
  a measured object size. Agents subsequently replaced these fits.
- At a requested 20° camera spread, two helpers moved matched static features
  only 0.34/0.45 pixels versus predicted requested movements of 19.55/16.75
  pixels. A matched 40° run changed appearance more but warped the background
  and lacked enough reliable matches to validate the camera geometry. These
  are consistency diagnostics under predicted initial depth/K, not GT camera
  measurements. Increased viewpoint appearance change did not validate fusion.

## Independent evaluation and scope

An initial-only geometry evaluator was attempted after all four output hashes
were sealed. It refused the available archived reference because none of its
320 RGB images exactly matched the input. Both sources were 640×360 uint8, so
this was not a resolution mismatch. The camera/frame provenance remains
unresolved; no approximate alignment or geometry scoring was substituted.
Independent shape, contact and execution accuracy therefore remain unmeasured.

The earlier recorded-video proxy executions are separate experiments. They
must not be attached to these forecast reconstructions as execution evidence.
No new model training was performed in this comparison.

## Evidence and reproduction

Outputs are under `outputs/block-forecast-20261120/` (ignored by Git):

- `agents-v2/sealed-suite.json`: all terminal receipts and final scene hashes.
- `agents-v2/comparison-v2/comparison.json`: corrected common-supported scores;
  the original all-frame diagnostic comparison is preserved separately.
- `agents-v2/{source,helper}/{A_raw,B_object}/output/REPORT.md`: agent reports.
- `media/agents-v2/four-way-comparison-labeled/four-agent-comparison.mp4`:
  all four shapes in a common coordinate frame and predicted table.
- `generated/generated-four-views.mp4`: observed initial/source forecast and
  generated helper views; source RGB is natively 3 fps, held at 30 fps.
- `media/tracker-v2-comparison/framewise-vs-video-tracker.mp4`: segmentation
  comparison, including remaining late failures.
- `media/camera20-vs-camera40-v2/camera-spread-comparison.mp4`: camera control.
- `evaluation/agent-initial-proxy-v1/identity-diagnostic.json`: reference refusal.

See [agent packet preparation](README_representation_forecast.md),
[sealed comparison](README_forecast_agent_comparison.md),
[tracking](README_forecast_tracker.md) and
[independent initial-only evaluation](README_forecast_initial_proxy.md).

## Next controlled comparisons

The target full method is image/task → generated helpers → confidence-aware
object/part shapes and SE(3) tracks → agent refinement → frozen plan. Current
results reach only partial block reconstruction. The next useful additions are:

1. Identity/occlusion-aware initialization that preserves the observed shape,
   refuses unsupported motion and downweights inconsistent helper cameras.
2. Matched ablations removing sequence initialization, generated helpers,
   confidence weighting, and shared rigid motion one factor at a time. A true
   no-world-model baseline must not receive generated future source RGB.
3. Separate world-model controls for one-frame prediction, full-clip view
   generation and five-keyframe generation; these have different information.
4. Authenticate independent geometry/evaluation data, then freeze commands
   before execution. Repeated agents and held-out clips are needed before
   claiming runtime, geometry or generalization improvements.
