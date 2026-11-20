# Real RGB representation study

This is an experiment plan and implementation status, not a claim of completed
GPU inference, fresh agent results, training, or successful execution.

## Question and comparison

Which organization of the same model-derived evidence lets an agent recover a
shared scene and motion more accurately, with less refinement? The useful result
would identify a compact prediction target for the world model. Predicting RGB,
depth, or rigid motion alone is not claimed as a new method.

The first comparison is deliberately matched:

| Condition | Evidence available to a fresh agent |
| --- | --- |
| A: raw | Source RGB, predicted geometry/cameras, predicted object masks, the same generated canonical mesh and point correspondences |
| B: initialized | All A evidence, plus an organized shared mesh, estimated object poses and explicit unsupported states |

No condition may gain reference depth, CAD, supplied camera calibration, or
extra source frames through its initialization. Use the same model, tools,
agent budget and source-only feedback. Report preprocessing time separately
and include it in total time; preprocessing is not free agent effort.

Compare five versus denser agent-visible keyframes as a second factor. If the
frontend inferred geometry from a denser sequence, this is **agent input
compression**, not a five-frame frontend or a faster world model. Record both
the frontend context and exposed frame IDs. Full-video reconstruction and
single-image future prediction remain separate experiments.

## What is being built

1. Released Pi4 frontend: RGB to estimated cameras, metric geometry and dense
   correspondences, preserving native predictions and scale conventions.
2. RGB masks establish task-object identity. The Pi4 moving-region output alone
   does not identify the object being manipulated.
3. A source-only first-frame fit places one SAM3D mesh. Both conditions receive
   the same canonical asset. Completion of unseen surfaces remains a model
   hypothesis.
4. `rigid_track_initializer.py` fits robust SE(3) between corresponding predicted
   object points. It rejects incompatible points without changing the mesh
   dimensions for each frame. Missing/degenerate evidence is explicitly
   unobserved; a carried-forward pose is merely an initial value.
5. A fresh agent refines the scene using only source render/depth/mask checks.
   The output is sealed before independent scoring.

The rigid initializer takes an already authenticated first-frame fit. It does
not yet automate that upstream mesh fit, infer contact, resolve absolute scale,
or perform planning. Low residuals are model self-consistency, not accuracy.
Systematic depth bias can survive a perfect rigid fit. No confidence head is
treated as calibrated metric certainty without evidence.

## Real data and training

The frozen split is in `real_video_split_20261119.json`: flat-box and reused
pump-bottle clips for development/training; the spatula clip held out. Source
camera and frame ranges are fixed before fitting. `README_real_video_split.md`
documents the selection limitations and preparation commands.

All three source bundles are now materialized from a pinned public HO-Cap
mirror: 360 original RGB frames, 27.5 MB of JPEGs, with decoded-array and byte
verification receipts. Identity against the older cluster archive remains
unverified. This is completed data preparation, not completed training.

Main method inputs are RGB. Sensor depth, annotations and supplied cameras are
kept in a separate evaluator location. Model-derived shared shapes, poses,
observability and explicitly known/unknown contact states can become training
targets. Do not convert absent annotations into negative contact labels or
identity-rotation targets.

Training is a subsequent measured experiment, not accomplished by writing a
dataset manifest or running a pretrained model. First establish which target
helps the agent; then train a bounded predictor/distillation pilot using the
real training clips and model-derived targets, and compare on untouched clips.
Two training clips and one held-out clip support a pilot only. Subjects overlap;
neither dataset-wide generalization nor unseen-pretraining exposure is proven.
No new world-model training has been completed in this study yet.

## Evaluation and deployment boundary

Report coverage, shared shape dimensions/surface error, absolute object/hand
placement, motion error, held-out-view renders, source agreement, runtime and
agent tool/refinement effort. Keep failures and missing outputs in the report.
Never align predictions to reference geometry and then report the aligned
result as absolute reconstruction accuracy.

The eventual execution comparison uses one explicit fixed deployment
calibration, a fresh reconstruction per method, a frozen plan, and the same
independent simulator scene. Reference geometry may initialize the evaluator;
it must not initialize the planner. No simulation outcomes may be used to
refine the evaluated plan. New-embodiment transfer and a sub-ten-minute complete
pipeline remain objectives, not established results.

## Current local checks

The rigid initializer's tests cover rejection of wrong correspondences,
recovery after occlusion without chained drift, unobservable rotation,
explicit visibility gating, preserved metric bias, shared mesh scale and
placement, and rejection of apparent nonrigid inflation. These are code checks,
not results on the real clips. The matched-agent harness also passed an actual
offline sandbox preflight in its intended workspace placement, including denied
sibling/reference reads and immutable input files. Native GPU inference and
fresh real-scene agent runs remain pending; no new trained weights or execution
success are claimed.
