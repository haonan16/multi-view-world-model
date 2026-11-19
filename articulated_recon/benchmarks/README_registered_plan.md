# Registered TCP planning

`registered_plan.py` turns a shared reconstructed object track into a TCP and
parallel-gripper command sequence. It **does not solve Panda IK, inverse
dynamics, collision-free motion or physical execution**. No simulator or
evaluator scene is accepted as an input.

## Required inputs

- A validated shared scene with estimated metric scale, one shared object
  geometry, observed rigid poses, and one explicit continuous grasp interval.
  A grasp ending before the clip ends additionally requires a table-support
  interval at the first released frame. Unknown or disjoint grasp state is
  rejected; the adapter never fills a first-to-last grasp through missing events.
- A materialized input manifest with resolved metric scale and registration.
  Model estimates remain estimated. Supplied calibration retains the
  `calibrated_rgb` condition; this tool does not relabel it RGB-only.
- One fixed deployment registration, materialized and hash-bound as a consumed
  `robot_registration` artifact. It has schema `deployment-registration-v1`,
  `source_frame`, `target_frame`, `T_robot_scene`, `deployment_type` (`virtual`
  or `measured`), `procedure`, `fixed_across_methods=true`, and
  `derived_from_evaluation_geometry=false`.
- A complete fixed planner configuration. Save it before comparing variants.
  The configuration includes gripper width, speed limits, approach/retreat
  distances, optional deployment workspace bounds, and geometric grasp priors.

A virtual placement derived from source RGB/model estimates must be declared
as virtual and estimated. It is not a measured camera-to-robot calibration.
The historical `shared-table-registration-v1` evaluator frame is not accepted:
its translation and support correction were derived from a reference object's
pose and mesh, so copying its matrix into a calibration file does not remove
that provenance.

## Method

The adapter applies the supplied rigid registration to every object pose. It
chooses a parallel-jaw grasp from the reconstructed bounding envelope, expresses
the TCP relative to the object, and propagates that fixed relative transform
along the estimated object motion. The top-down mode uses the input support
normal; side mode uses an explicitly configured object axis. This is a robot
grasp prior, not direct copying of human finger joint angles.

Actual observation timestamps set transport intervals. Translation is linearly
interpolated and orientation uses SLERP. Segments may be slowed to the fixed
speed limits, with both original and command durations recorded. Approach,
closure and retreat are explicit synthetic robot phases, not predicted video
motion. The initial robot-to-pregrasp path remains unchecked.

Object shape and position are never repaired during planning. Estimated
below-table geometry, an oversized grasp, excessive pregrasp movement, missing
observations or out-of-workspace targets cause rejection. There is no
recentring, support lifting, substitute target or evaluator-based correction.
Passing these geometric checks does not establish grasp stability or collision
clearance.

## Commands

From `articulated_recon`:

```bash
python benchmarks/registered_plan.py config \
  --object-id block --output /path/planner-config.json
python benchmarks/registered_plan.py compile \
  --scene-path /path/scene-sequence.json \
  --input-manifest /path/input-manifest.json \
  --registration-id deployment-registration \
  --config-path /path/planner-config.json \
  --output /path/new-plan-directory
```

The output directory must not already exist. `plan.npz` contains
`T_robot_tcp`, `gripper_width_m`, `gripper_command` (+1 open / -1 force-close),
`phase`, `duration_s`, `command_time_s`, `source_time_s`, `source_time_valid`,
the unchanged registered object track and its true timestamps. Synthetic phases
have `source_time_valid=false`. The closed target width is a controller command,
not a claim that the object's width is zero.

`plan-metadata.json` records source, manifest, planner, configuration,
registration and plan hashes; it can be passed to `freeze_submission.py` with
the actual planner file and a controller targeting the same robot frame.
Sealing binds these inputs before evaluator attachment. It does not validate
IK, dynamics, command compatibility or execution success.

The API and its tests use synthetic array fixtures. Running those tests is not
a new result on the block or bottle demonstrations.
