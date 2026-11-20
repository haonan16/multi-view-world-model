# Matched fresh agents on a strict first-image forecast

`representation_forecast.py` prepares two representations of the same forecast
evidence. It launches no inference. The planned four cells are source_unposed
raw/initialized and helper_unposed raw/initialized, each using every forecast
timestamp. The initializer choice is the saved baseline fit, selected before
comparison; rest/plane numerical variants are separate ablations.

Both representations receive exact native DA3 predictions, tracked masks, the
first-image scene, the same saved fitting configuration, generated RGB contact
sheets and a self-contained numerical fitting tool. Common observations convert
DA3 depths and camera translations using **one first-image UniDepth/DA3 ratio**.
Future fitted scales are not present in the raw evidence. The initialized packet
additionally receives the fitted shared cuboid, poses and scale vector. Its
completed initializer time is recorded separately from agent time. Either agent
may invoke the identical fitting tool and may refine its result.

Only source view0 at time0 is observed. Every other image is generated. Missing
or rejected tracker views remain unknown; neither high tracker presence nor a
transported seed proves object identity. All native DA3 intrinsics are retained.
The source-camera gauge across timestamps assumes a fixed source camera. Scale,
hidden shape, contact and generated camera consistency remain uncertain.

## Inputs and launch

Required directories are the strict condition, generated observations, the
selected DA3 branch, corrected tracked masks (source-only export for the source
condition), and the completed baseline fit. The condition's referenced initial
geometry and scene manifests/files must be available. The baseline fit supplies
`scene-sequence.json`, `fit-config.json`, `input-manifest.json`, `execution.json`.
Do not use the known-wrong first tracker version as the measured input.

```sh
PYTHON representation_forecast.py config --python PYTHON --output agent-config.json
PYTHON representation_forecast.py pair \
  --condition CONDITION --helper GENERATED --predictions DA3/source_unposed \
  --masks TRACKED/source-only --initializer FITS/source_unposed-baseline \
  --config agent-config.json --output SOURCE_PAIR \
  --remote-prefix REMOTE_ROOT --local-prefix LOCAL_MIRROR
PYTHON representation_forecast.py pair \
  --condition CONDITION --helper GENERATED --predictions DA3/helper_unposed \
  --masks TRACKED --initializer FITS/helper_unposed-baseline \
  --config agent-config.json --output HELPER_PAIR \
  --remote-prefix REMOTE_ROOT --local-prefix LOCAL_MIRROR
PYTHON representation_agent_harness.py prepare \
  --packet SOURCE_PAIR/A_raw --packet SOURCE_PAIR/B_object \
  --packet HELPER_PAIR/A_raw --packet HELPER_PAIR/B_object --output SUITE
PYTHON representation_agent_harness.py run --suite SUITE --run-id RUN_ID
PYTHON representation_agent_harness.py run --suite SUITE --run-id RUN_ID --execute
```

Use absolute paths and place packet/suite outputs under the workspace, **not
`/tmp`**. Omit relocation prefixes for local inputs. Relocation writes only
controller-side clones; original manifests remain byte-identical. The receipt
records original/derived manifest and preparation hashes while preserving every
artifact hash. Existing strict first-image validation is rerun on the relocated
inputs, including recreation of the conditioning arrays from the initial image.
Saved bytes and hashes stay exact. Recreated virtual camera float64 values allow
only eight local floating-point spacing units for platform arithmetic roundoff;
RGB, K, timestamps and source indices still require exact equality. The relocation
receipt records the observed discrepancy. `PAIR/controller/initial_scene_comparison`
is the hash-verified, local first-image scene bundle for the sealed comparison's
`--initial-scene-bundle` argument.

The forecast config explicitly selects `gpt-6.2-sol`, high reasoning, 600 seconds,
40 tool calls and a 1,000,000 reported-token ceiling. The ceiling includes cached
input tokens and is not a spending target. Input, cached input, uncached input
and output tokens are reported separately; model requests/cost remain unknown.
The four runs must share the exact config and prompt. There is one fresh attempt
per cell and no automatic retry. Offline preflight performs no agent inference;
only `--execute` invokes the fresh agent.

Packets contain per-view contact sheets covering all times, avoiding 68 separate
image tool calls. The prompt requires actual image and output-overlay inspection,
scene refinement and an honest report. Numerical fitting alone is not an agent
result. Both packets offer `scene_tools.py seed`, `optimize`, `validate`, `check`.

## Checks, export and interpretation

`scene_tools.py check` computes primary silhouette/depth agreement against the
common frozen reference. Each frame/view also has a separately named
`candidate_scale_diagnostics` section using the candidate's scale hypothesis.
Candidate scale cannot improve the primary metric by redefining its reference.
All values are source-prediction consistency, never GT geometry/contact/forecast
accuracy. For comparisons across source/helper branches, use one frozen source
reference for all four scenes and report helper agreement separately.

```sh
PYTHON representation_forecast.py export --packet PACKET \
  --scene PACKET/output/scene-sequence.json --output AGENT_BUNDLE
PYTHON render_forecast_reconstruction.py --helper PAIR/controller/helper \
  --bundle AGENT_BUNDLE --output AGENT_VIDEO --fps 30
```

The export is compatible with the existing forecast renderer and its strict
manifest checks. Its `geometry.npz` applies candidate scales for visualization,
explicitly labeled in `execution.json`; it is not the primary comparison
reference. Use `PAIR/controller/helper` as the relocated renderer helper bundle.
Future scene objects must have `observed=false`; `prediction_supported[0]=false`
and generated model provenance are required. Unsupported intervals stay unknown.

Synthetic tests exercise A/B equality, scale separation, source-only lineage,
multiview overlays, remote relocation, renderer ingestion and an isolated Python
subprocess running the packet's copied optimizer. These establish plumbing, not
real tracker quality or a successful agent outcome.

The actual installed-CLI smoke test passed on 2026-11-20 for both synthetic
forecast representations, including blocked sibling/repository reads, immutable
inputs and denied network. The copied optimizer also completed inside that same
restricted profile with isolated Python (`-I`). Receipt:
`outputs/representation-forecast-smoke-20261120/preflight-receipt.json`.
No fresh agent/model inference was launched by this test. The outer execution
sandbox was escalated only to permit nested macOS seatbelt creation; the tested
inner source-only sandbox remained enforced.
