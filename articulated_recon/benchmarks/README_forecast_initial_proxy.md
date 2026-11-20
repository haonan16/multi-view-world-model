# Independent first-image geometry check

`evaluate_forecast_initial_proxy.py` compares four sealed agent reconstructions
with the earlier recorded RGB-D reconstruction proxy. This proxy is **not exact
GT CAD**. Its camera and reconstruction can contain errors. The evaluator checks
only initial shape and placement; it does not score forecast motion, contacts,
planning or execution.

All four actual scene/status hashes must match terminal harness receipts before
either reference file is opened. `package-seals` makes a small portable copy of
those scene/status files so the remote exporter performs the same check. The
root seal schema is `all_agents_sealed: true`, `suite_sha256`, and four `runs`
with `id`, `scene_path`, `scene_sha256`, `status_path`, and `status_sha256`.
Optional `first_image_rgb_sha256` additionally binds the observed RGB bundle.

The reference camera archive's native first RGB frame must equal the isolated
observed image exactly: same uint8 dimensions and every pixel. Missing RGB,
different crops, resizing, JPEG changes or mismatched frames cause refusal.
There is no approximate image match or automatic resampling fallback. Camera
IDs are checked if the archive contains them; otherwise the caller-declared
source-view mapping is disclosed.

The exporter retains only the reference's initial mesh/pose and maps it through
`inverse(source_c2w)`. It does not optimize scale, registration, pose, table height
or alignment. Any historical linear scale in the proxy's initial pose is baked
into the same mesh without moving its surface. Metres follow the archived
reconstruction producer's convention.

After the seal is written and the evaluator is authorized:

```bash
python benchmarks/evaluate_forecast_initial_proxy.py package-seals \
  --sealed-suite /path/agents/sealed-suite.json --output /path/sealed-proxy-inputs
# Copy only this small sealed bundle and the evaluator code to the cluster.
python benchmarks/evaluate_forecast_initial_proxy.py export \
  --sealed-suite /path/sealed-proxy-inputs/sealed-suite.json \
  --observed-manifest /path/inputs/first-image/input-manifest.json \
  --raw-reference /path/recorded-reference.json \
  --source-cameras /path/recorded-obs.npz --source-view 0 \
  --source-camera-id 036422060215 --output /path/new-initial-proxy
# Copy the tiny proxy and export receipt back to the evaluator directory only.
python benchmarks/evaluate_forecast_initial_proxy.py score \
  --sealed-suite /path/agents/sealed-suite.json \
  --proxy-bundle /path/new-initial-proxy --output /path/initial-proxy-scores.json
```

Metrics use surface-area-weighted centroids and 2,048 deterministic area-uniform
surface samples per object, each queried against the other mesh's exact
triangles. Extents are measured in the proxy object's fixed axes, so orientation
error contributes; no fitted oriented-bounding-box alignment is applied.
Outputs preserve each scene hash and label the reference as a reconstruction
proxy. Tests cover seal-before-reference ordering, tampering, exact RGB identity,
camera-only coordinate conversion, and preservation of translation/scale error.

Preparing this evaluator and passing synthetic tests does not establish any
real geometry score. Do not expose its references or results to fitting agents.
