# Frozen registered-plan execution

`execute_registered.py` connects the new shared-scene planner to native Panda
physics. It is separate from the legacy recorded-reconstruction executor. A
completed code/probe check is not a successful reconstructed manipulation trial.

The native CPU probe (cluster job 62170953, 2026-11-19) constructed a generated
cube and a tilted plane, preserved their initial configuration, and executed 40
controller steps. Final TCP error was 0.070 mm. It exercised collision building,
Panda control and contact/state logging; no recorded scene or scientific task
plan was used. The tested executor SHA-256 was
`6155cb3189d0ca3a550900db1ad942acd3d643d63188b1daf8305e1f174291a0`.

The planner receives estimated scene geometry and a declared deployment
registration. It emits TCP poses and open/close commands without loading a
simulator or the evaluator. The executor applies that fixed sequence through
ManiSkill's absolute `pd_ee_pose` IK/PD controller. This controller uses robot
proprioception; object poses, contacts and success measurements only enter
logging and scoring. This is **not learned inverse dynamics**.

## Order of operations

Run these commands from `articulated_recon` with the appropriate Python runtime.
Paths below illustrate an already validated, registered method run. Finalize
the producer/executor code and both configs **before** freezing the submission.
Use the same planner/controller settings and deployment artifact for every
compared method.

```bash
python benchmarks/registered_plan.py config --object-id block --output planner.json
python benchmarks/execute_registered.py config --target-frame panda_base --output controller.json
python benchmarks/registered_plan.py compile \
  --scene-path method/scene.json --input-manifest method/registered-input-manifest.json \
  --registration-id deployment__fixed_deployment --config-path planner.json --output method/plan
python benchmarks/freeze_submission.py freeze \
  --scene method/scene.json --input-manifest method/registered-input-manifest.json \
  --plan method/plan/plan.npz --plan-metadata method/plan/plan-metadata.json \
  --planner benchmarks/registered_plan.py --executor benchmarks/execute_registered.py \
  --controller-config controller.json --output method/submission.json
```

The evaluator stage runs only after this seal exists. Its source calibration and
recorded RGB-D reference are **evaluation-only** inputs. The deployment artifact
must be the exact one already consumed by the planner.

```bash
python benchmarks/independent_scene.py \
  --submission method/submission.json --raw-reference evaluator/recorded-reference.json \
  --source-cameras evaluator/published-source-camera.npz --source-view 0 \
  --source-camera-id 036422060215 --deployment common/deployment.json \
  --output method/evaluation
python benchmarks/execute_registered.py run \
  --submission method/submission.json \
  --evaluation-receipt method/evaluation/evaluation-attachment.json \
  --output method/execution
```

The first use of a deployment manifest may have registration ID
`fixed_deployment`; `fixed_deployment.attach` adds the `deployment__` prefix.
Read the actual manifest instead of silently replacing missing registration.
Forecasting cannot reuse a deployment derived from a full observed video.

## What stays fixed

- The seal binds scene, plan, producer, executor, controller config and active
  input files. The config additionally binds validation-helper hashes, runtime
  package versions and the Panda URDF hash.
- An exclusive execution receipt is written before environment construction.
  A second output directory cannot replay the same seal. Infrastructure errors
  retain the receipt and report exactly how many steps were submitted/recorded;
  there is no automatic retry.
- The independent scene preserves its transformed object placement and support
  plane, including tilt, offsets or existing proxy penetration. The object is
  never snapped onto the method's table, recentered on a predicted grasp, reset
  during execution, or attached with an artificial grasp constraint.
- The plane's SAPIEN local +X normal is rotated to its declared normal. Rendering
  is disabled; native collision geometry and fixed material parameters remain.
- The first pregrasp command interpolates from the fixed initial robot pose.
  Later targets are already densely sampled by `registered_plan`. Each command
  receives at least one 20 Hz control tick, with nearest-tick duration rounding.
  `timing.json` reports the resulting duration change and fixed final settling
  interval. Source clocks retain explicit validity, including negative valid
  timestamps. Phase retiming is not represented as original-video timing.

Outputs include `actions.npy`, `records.json` (robot joints, TCP, object poses,
contacts and source-clock mapping), timing and score summaries. These saved
states can produce an offline replay video without rerunning physics.

## Limits

The current deployment is a **virtual benchmark placement**, estimated from one
RGB-only initializer and shared unchanged across methods. It is not measured
real-robot registration. The independent geometry is a recorded RGB-D
reconstruction proxy with assumed density/friction, not exact CAD/GT physics.
The current object uses a convex collision hull, suitable for a simple block;
it does not preserve cavities or articulated mechanics. Geometric plan guards
do not guarantee reachability, collision avoidance or grasp success.

Runtime pinning and interface validation are not OS-level process isolation, a
full container hash, or proof that a producer truthfully declared every input.
For stronger isolation, run planning with only its allowlisted files mounted,
then provision evaluator assets to a separate process after freezing.
