# Fresh source-only representation agents

`representation_packet.py` and `representation_agent_harness.py` prepare and
instrument actual fresh agents. They do not label an optimizer baseline as an
agent. No inference launches unless the runner receives `--execute`.

Use the same frozen model/config and exact prompt for every comparison. The
intended main conditions are:

| Condition | Shared source evidence | Additional representation |
| --- | --- | --- |
| A_raw | selected RGB, predicted depth/native world points, predicted cameras, masks/unknown masks, canonical model asset and selected point correspondences | none |
| B_object | byte-identical evidence | one canonical mesh, rigid SE(3) track, observation flags and explicit unknown contact/hidden geometry |

The main B initializer comes from the source-only model mesh and rigid track
adapter. The optional `--weak-hull-initializer` is a separate weak ablation: a
visible-surface hull and centroid translations with unestimated identity
rotations. It is not the main proposed representation and does not recover
occluded extent. Nothing selects case-specific primitive dimensions by hand.

Five versus thirteen inputs is an **agent input budget** factor. A model run
that already consumed a denser clip remains a denser frontend run. The packet
retains the explicit `frontend_context_frame_ids`; reducing an agent packet
does not demonstrate faster frontend inference.

## Source contracts

The authenticated `geometry.npz` uses RGB `[T,V,H,W,3]`, camera-Z `depth_m`
`[T,V,H,W]`, increasing `timestamps[T]`, and static `[V,...]` or dynamic
`[T,V,...]` predicted `K`/`c2w`. The current harness accepts one observed source
view. Native `points_world[T,V,H,W,3]` take precedence over unprojection through
an approximate pinhole K. Pi4 `local_points_m`, confidence and pinhole residuals
are retained. An unavailable K with valid native world points remains an
unknown projection; its render metrics are null.

`source_frame_indices[T]` identifies original clip indices, including strided
frontends. Packets distinguish `geometry_row_ids` from `native_frame_ids`.
The default frontend context is those source indices when the producer supplies
them; otherwise it remains null unless explicitly supplied. Use
`--frontend-context-frame-ids indices.json` for an explicit original-index list.

Masks require `object_masks[T,V,H,W]` plus `object_mask_known[T,V]` and optional
`occlusion_mask[T,V,H,W]`. Alternatively, provide the authenticated native SAM3
mask file, its manifest and the object prompt. Duplicate candidates are
unknown. Mask grid/timing/source roots must match; no silent resize or guessed
association occurs. Pi4 dynamic probability alone is not object identity.

Optional selected correspondences use an authenticated `tracks.npz` with:

- `tracks3d[T,N,3]`, exact geometry `timestamps[T]` and scalar `coordinate_frame_name`;
- `query_frame[N]`, `query_frame_indexing` (`original_source` or `geometry_rows`), and optional query pixels/flat IDs;
- optional `tracks2d[T,N,2]`, boolean visibility, raw track visibility/confidence.

Only queries originating in selected source frames are retained. Each packet
gets the same rows/columns and query metadata. Track files can be declared as
`depth` or `object_geometry` in the RGB-only lineage. Asset, track and mask
source RGB roots must match the geometry source root.

A structured initializer is an inline `scene_sequence` JSON plus a receipt
binding `source_observations_sha256`, `native_frame_ids`,
`frontend_context_frame_ids`, `input_scope="selected_frames_only"`,
`initializer_sha256`, and `producer_sha256`. It also binds
`shared_asset_sha256`, `masks_sha256`, and `tracks_sha256` when consumed. Raw
evidence named by that receipt must be present in both packets. The receipt may
include `elapsed_seconds`; unmeasured preprocessing times remain null.
The initializer must use exactly selected timestamps and the packet gauge.
This authenticates declared provenance/file identity, not the truth of producer
claims or the accuracy of the recovered scene.

## Preparing and running

Use a Python runtime with NumPy, SciPy and Pillow. Preserve its venv path rather
than resolving the executable symlink to the base interpreter. Freeze the
intended model explicitly, for example:

```bash
python benchmarks/representation_packet.py config \
  --model gpt-6-astra --python /absolute/path/to/venv/bin/python --output config.json

# Repeat with B_object, adding --initializer and --initializer-receipt.
# Both conditions receive the same --asset, --masks and --tracks arguments.
python benchmarks/representation_packet.py build \
  --observations geometry.npz --manifest input-manifest.json \
  --condition A_raw --budget 5 --config config.json --output packets/A5

python benchmarks/representation_agent_harness.py prepare \
  --packet packets/A5 --packet packets/B5 \
  --packet packets/A13 --packet packets/B13 --output suite

# Offline isolation preflight only. Read suite.json for its fixed shuffled IDs.
python benchmarks/representation_agent_harness.py run \
  --suite suite --run-id 000-A_raw-5frames

# Explicitly start one fresh measured agent after preflight passes.
python benchmarks/representation_agent_harness.py run \
  --suite suite --run-id 000-A_raw-5frames --execute
```

Freeze each packet before agents start. The runner requires exact immutable
input inventories, rejects extra files/symlinks and old outputs, and writes one
exclusive execution receipt per packet. A failed or timed-out run cannot be
silently retried through a new suite. Fresh replicas must have newly prepared
packet directories and declared run IDs.

The actual CLI sandbox preflight tests Python imports, denied sibling/repository
reads, immutable input writes, allowed output writes and a denied network
connection. It must pass before any inference. Keep measured packets outside
broadly readable system temporary roots. An actual disposable `/private/tmp`
probe failed sibling-read isolation on this host; the same synthetic A/B pair
under ignored `benchmarks/outputs/representation-harness-smoke-20261119` passed
both cells. The receipt is that directory's `preflight-receipt.json`. These were
offline infrastructure tests, not recorded-scene reconstruction results.
Creating a nested macOS seatbelt from the outer execution sandbox failed with
`sandbox_apply: Operation not permitted`; the successful probe ran outside only
that outer sandbox while retaining the inner restricted Codex profile.
Canary checks are evidence about the tested boundaries, not proof against every
possible host/tool access path.

## Agent outputs and source tools

Each fresh agent receives the exact `representation_agent_prompt.txt`, source
PNGs, immutable arrays/schema/tool and a selected output directory. It writes
`output/scene-sequence.json`, `output/REPORT.md` and self-reported accounting.
The tool accepts inline shared mesh geometry (including one constant stretch)
or simple shapes and validates full proper SE(3) tracks. A source check writes
per-frame RGB overlays and camera-Z render arrays. Hidden/occluded pixels are
excluded; absent masks and unavailable projection produce null scores. No
actor-mesh, simulator, contact-truth or execution check is performed.

The external runner logs CLI JSONL, measured elapsed time, tool counts, token
usage when exposed, and sampled distinct valid scene hashes. A CLI completed
turn is not an individual API request: `model_requests` remains null because
the CLI events do not expose that count. Monetary cost also remains null.
Token ceilings operate on completed-turn usage and can be exceeded by an
in-flight request; such a run is explicitly marked over budget. Wall/tool
limits stop the process group, retaining failed/partial attempts. Checkpoint
counts are sampled valid scene revisions, not an exact count of every write.
Rendered-file existence does not prove the agent visually inspected it; the
event log and report must support that claim.

Model-derived training targets are modular arrays for canonical extent,
translation, rotation-6D, observation gates and contact validity. Missing
shape/rotation provenance defaults to unknown. Rotation gates can be explicit
model-supported estimates; they are not GT labels. Hidden SAM3D completion and
contact remain unverified. Match these pseudo-targets only within a declared
train/test split; the packet does not access training or evaluation results.

Focused tests: `python -m pytest -q mano/tests/test_representation_packet.py`.
