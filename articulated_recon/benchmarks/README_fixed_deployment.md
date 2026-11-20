# Fixed virtual deployment from the first image

`fixed_deployment.py` chooses a virtual Panda placement from an RGB-derived
initial scene. It is a coordinate convention for a benchmark, not measured
camera-to-robot calibration. The same rigid transform must be used for the
planner and the independent evaluator; neither object nor table is separately
recentered after evaluation geometry is opened.

For strict single-image forecasting, create a **new** deployment from the fresh
one-frame scene before planning and execution:

```bash
python benchmarks/fixed_deployment.py create \
  --first-image-forecast \
  --scene-path results/initial-scene/scene-sequence.json \
  --input-manifest results/initial-scene/input-manifest.json \
  --source-camera-id 036422060215 \
  --output control/first-image-deployment
```

The flag invokes the strict forecaster's input checks: one observed RGB image,
one timestamp, hash-bound active ancestry, and no later depth, camera, scene or
generated-future arrays. The scene must be in `source_camera` coordinates,
contain one supported initial object and an estimated support plane, and declare
its metric scale. A full-video reconstruction manifest is rejected even if its
scene is subsequently reduced to frame zero.

The transform rotates the estimated table normal to Panda +Z and places the
initial object's estimated table foot at `(0.45, 0, 0)` metres. It does not
rescale the object or repair geometry. Table/scale errors therefore remain in
the separately transformed evaluator scene. The deployment receipt records the
initial-image, scene and input-manifest hashes and the observed-time cutoff.

After reconstruction of the **generated** future, attach this unchanged
deployment to the forecast method manifest:

```bash
python benchmarks/fixed_deployment.py attach \
  --method-manifest results/forecast-scene/input-manifest.json \
  --deployment-manifest control/first-image-deployment/input-manifest.json \
  --output results/forecast-scene/registered-inputs.json
```

Attachment requires the same protocol, condition and observed RGB source bytes.
Generated future data remain model predictions. Full-video deployments cannot
be attached to a forecast. The registered planner then consumes
`deployment__fixed_deployment`; the evaluator receives the exact same
`control/first-image-deployment/deployment.json` after the plan is sealed.

The reconstruction-mode CLI without the flag retains its existing behavior and
its full-video provenance warning. Synthetic tests exercise the forecast
boundary and unchanged cross-method transform. They are not a real execution
result.
