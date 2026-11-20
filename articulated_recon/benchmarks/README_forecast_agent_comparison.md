# Frozen forecast-agent comparison

Compare the four matched block runs: raw evidence or structured initializer,
each with source-forecast evidence or generated helper views. The primary
question is whether initialization reduces agent effort while improving the
reconstructed shared shape and motion. One attempt per condition is descriptive.

Run this only after all four harness runs have terminal `status.json` receipts:

```sh
python compare_forecast_agents.py \
  --suite /absolute/path/to/suite \
  --reference /absolute/path/to/source-raw-packet \
  --helper-reference /absolute/path/to/helper-raw-packet \
  --initial-scene-bundle /absolute/path/to/initial-scene \
  --output /absolute/path/to/new-comparison
```

The script checks each final scene against its terminal hash before reading
comparison evidence. It never changes scenes, plans, or input packets. Missing
or invalid final scenes remain explicit failed results.
The primary summaries use the intersection of supported frame indices across
valid scenes. Coverage and all-frame diagnostics remain separate; unsupported
numeric placeholders are not counted as recovered trajectories. Predicted-table
penetration uses an explicit 1e-6 m roundoff tolerance and retains raw clearances.
If the initial scene manifest points outside its folder, add
`--initial-scene-path /absolute/path/to/scene-sequence.json`; the file must still
be authenticated by `--initial-scene-bundle`'s manifest.

It reports:

- **Common source agreement:** visible-mask IoU and predicted-depth residual
  on identical source pixels, depth, and cameras for every scene. Initial
  observed RGB and generated future RGB are summarized separately. Occluded
  or unknown masks remain unmeasured; overlap counts accompany depth residuals.
- **Common helper agreement:** the same metrics per generated helper camera.
  This measures how well one shared scene explains the predictions; it cannot
  establish that the predicted cameras or their generated views are correct.
- **Shape/motion plausibility:** shared mesh dimensions/topology, translation
  and rotation changes, supported-frame coverage, and clearance against the
  same first-image predicted table. A candidate's edited table is not used.
- **Measured agent effort:** runtime, reported tokens/tool calls, and sampled
  valid scene revisions from the harness. Branch labels and initializer runtime
  remain visible for failed runs too. Time to the first valid scene comes from
  the terminal measurement or earliest externally sampled checkpoint, with its
  scope labeled. Missing accounting stays null.

Candidate-declared per-frame depth scales do not alter the common reference.
The agent's own `scene_tools.py` may separately report agreement conditional on
its estimated scale; that is a different diagnostic. Source and helper camera
estimates may themselves be inconsistent, so both common-reference reports
must remain visible rather than selecting the more favorable one.

These are prediction-agreement and plausibility measurements, **not geometry
accuracy or execution success**. No future recorded motion is a target because
the generated forecast is hypothetical. A separate first-frame measured RGB-D
check is possible after all scenes are sealed and an authenticated initial-only
reference is prepared. The earlier reconstructed block proxy is not ground
truth. This script does not load that private reference or invoke simulation.
