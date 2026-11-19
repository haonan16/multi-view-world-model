# RGB-only reconstruction comparison protocol

This document defines the staged experiment plan and input validator. See
[the measured run status](README.md) for completed conditions and failures;
listing a condition here does not mean it has run. Model checkpoints, exact RGB
clips, budgets and the evaluation boundary must be bound for each frozen run.

## First batch

The initial cases are the RH20T block and HO-Cap bottle demonstrations. Each uses
one recorded RGB camera, with the same crop, timestamps and source bytes across
matched conditions. The default config lists eight primary experiments:

| Per scene | Reconstruction inputs | Temporal sampling |
| --- | --- | --- |
| Source baseline | Source RGB, model-estimated depth/cameras/masks/support | Dense |
| Helper-view baseline | Same source plus world-model-generated helper views | Dense |
| Five uniform frames | Same source and generated helper views | Five uniformly spaced frames |
| Five event frames | Same source and generated helper views | Five frames selected using estimated contact events |

Four optional controls add source-only dense reconstruction with supplied
calibration, and with sensor depth, on both cases. Their condition labels are:

| Experiment label | Input-manifest condition | Permitted inputs |
| --- | --- | --- |
| `strict_rgb` | `rgb_only` | RGB and model-derived products, with declared global model/URDF/MANO priors |
| `calibrated_rgb` | `calibrated_rgb` | RGB and model-derived products, plus supplied calibration |
| `rgbd_oracle` | `rgbd_oracle` | Explicit oracle control allowing measured depth/GT sources |

The initial sparse rows sample **reconstruction inputs**. Reusing a dense
world-model generation and taking five frames does not demonstrate reduced
generation cost. Native sparse generation is a later experiment. Similarly,
compressing an already fitted 3D trajectory measures fidelity to that fit, not
independent reconstruction accuracy.

## Separate later comparisons

- **First-frame forecasts:** only the first observed frame is supplied. The
  source-only baseline must obtain future source-camera motion from the world
  model; a monocular depth estimate of frame zero is not a forecast. The helper
  condition adds generated cameras to that same forecast. Future recorded RGB,
  sensor depth and trajectories cannot enter the method.
- **External methods:** SceneRig receives one frame, while OVOW receives the
  full clip. Their native procedures and budgets differ, so these rows are
  marked **unmatched**. They provide reconstruction comparisons and do not
  include our human-to-Panda action conversion. Use `external_baselines.py` for
  separate release/preflight/launch controls.
- **Peg insertion and RoboCasa:** precision and environment-generalization
  targets, currently listed as not scheduled. No GPU allocation is implied.

## Input provenance and execution gates

`artirecon.input_protocol` follows every consumed artifact's parent IDs. A model
output cannot erase sensor-depth or GT ancestry. Generated RGB must be marked
`generated=true`; this comparison additionally labels its `view_role` as
`helper` or `source_forecast`.

Metric scale and **independent target-robot registration** must be resolved from
permitted inputs before action execution. A human demonstration contains no
automatic robot-base transform. An assumed identity registration or the held-out
evaluation scene's transform is not a solution. Until these are established, the
input check reports blockers and leaves execution readiness false.

Before running a matched group, fix source hashes, source camera/crop/time span,
model revisions, seeds, optimization steps, maximum agent calls, hardware and
timeouts. Keep those choices independent of held-out geometry and execution
outcomes. The final plan and commands must be sealed before one rollout in the
independent fixed evaluation scene. No simulator feedback goes to the planner,
and no successful outcome is selected from retries.

Contact-event selection uses estimated source evidence in reconstruction mode,
and model-predicted evidence in forecasting mode. Five frames cannot always
cover all events: `scene_sequence.select_keyframes` reports omitted boundaries.
Unknown confidence remains unknown; it is not assigned an invented probability.

## Shared reconstruction output

`artirecon.scene_sequence` defines one sequence with SI units, an explicit
coordinate frame and metric-scale status, timestamps, shared actor/object shape,
full SE(3) trajectories, observation flags, a support plane, and optional contact
intervals with provenance. MANO stores one beta vector and its wrist/PCA track;
URDF stores a model reference and named joint trajectories. A moving mesh stores
one geometry and one rigid pose per frame. Invalid rotations, missing state
frames, changing per-frame shapes and malformed contact intervals are rejected.

`run_robotfitter_mano.py` exports `hand-results/<case>/scene-sequence.json`
alongside its existing outputs. This is currently **the fitted hand plus its
input support plane**, not the downstream object reconstruction or an executed
plan. No contacts are inferred during export. A hand frame is marked observed
only when `point_counts > 0 AND visible`. The licensed MANO model is referenced
locally and is not copied into the scene.

Manifest-verified runs require actual observation timestamps; legacy files may
use the explicitly labeled frame-index/FPS fallback. Legacy exports retain
unverified provenance and unresolved metric scale. A validated input report is
preserved, including unresolved robot registration. Exporting a scene does not
resolve any execution blocker or certify geometry accuracy.

## Commands

From `articulated_recon`:

```bash
# Eight primary plans plus four controls; writes exclusively, never overwrites.
python benchmarks/experiment_matrix.py plan --out /path/to/experiment-plan.json
python benchmarks/experiment_matrix.py validate /path/to/experiment-plan.json

# Only when preparing later stages; these flags still launch no jobs.
python benchmarks/experiment_matrix.py plan \
  --include-forecast --include-external --out /path/to/later-plan.json

# Check one materialized input bundle against its intended comparison.
python benchmarks/experiment_matrix.py inputs /path/to/experiment-plan.json \
  --run-id block__strict_rgb__reconstruction__source_only__dense \
  --manifest /path/to/input-manifest.json --verify-files
```

The last command validates declared provenance and file identity. It does not
prove the truth of producer metadata, image consistency, estimated geometry or
physical success, and it does not seal or execute a plan.

## Reporting

`freeze_submission.py` provides an audit interface between the registered
planner and independent evaluator. `freeze` binds the validated shared scene,
input manifest, command file, producer metadata, planner/executor files and
controller configuration. It accepts no evaluation-scene argument. The
`attach-evaluation` step records fixed evaluator artifacts after sealing;
`claim` creates one exclusive execution receipt before simulator construction.
Use `--help` for the required file arguments. This tool launches no simulator.

The producer metadata must already name the scene, command and planner hashes,
the protocol digest, robot-registration artifact, coordinate frame and command
format, and declare `simulator_feedback_used=false`. Hash checks enforce declared
file identity, not process isolation or truthfulness of historical data access.
The legacy action compiler still recenters motion and its executor builds the
reconstructed scene. Use the separate `registered_plan.py`, `independent_scene.py`
and `execute_registered.py` route for the new fixed-registration experiment;
do not relabel legacy rollouts as independent evaluation.

The shared measurement template initializes geometry, contact/clearance,
trajectory, execution, stage runtime and agent-effort values to **null**, meaning
unmeasured. Zero is a measurement and must never stand in for a missing run.
Report stage timing separately (world model, depth, masks/assets, actor/object
fit, planning and execution), including cold loading costs. Count agent calls,
tokens, cost and refinements explicitly, including failed attempts.

Report external methods by input scope and native budget. Report all fixed
variants, failed reconstructions and rejected plans. Model confidence ranks are
not calibrated uncertainty unless a separate held-out calibration establishes it.
