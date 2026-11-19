# Native URDF shared-scene export

The MANO runner already emitted `scene-sequence.json`. Amani's native robot
runners previously emitted `fit.pkl` and the older `scene.json`/viewer formats.
`run_scene.py` and `driver.py` now also call the array-only
`artirecon.urdf_scene_exports.export_native_urdf` hook. Native files are unchanged.

The shared export currently contains the URDF actor and optional fitted support
plane. Native object meshes, articulated object results, and contacts remain in
the existing native outputs; this hook does not convert or infer those fields.
Joint names follow the actual `robot.active` order, not a hardcoded Panda list.
Static or time-varying base poses are preserved, as are supplied timestamps and
coordinate-frame metadata. Missing observed flags remain explicitly unobserved.

Legacy fit results often contain control-step indices but no second timestamps.
The exporter does **not** guess a frequency. It writes
`shared-scene-export.json` with `status: not_exported` until actual timestamps or
an explicit control frequency are available. For example, place this metadata
in the run's `scene-export-metadata.json` before running, using the actual clock
and frame of that recording:

```json
{
  "timestamps": [0.0, 0.13, 0.41],
  "coordinate_frame": {
    "name": "declared_source_frame",
    "handedness": "right",
    "up_axis": "unknown",
    "pose_convention": "T_world_local"
  }
}
```

Alternatively set `AR_SCENE_EXPORT_METADATA=/absolute/path/metadata.json`. A
`control_hz` declaration can replace explicit timestamps when native `steps`
exist; the export preserves `steps/control_hz` without resetting the epoch.
Native second timestamps take precedence, and conflicting supplied timestamps
are rejected. The numeric example above is illustrative, not a dataset clock.

Sidecar metadata cannot upgrade a legacy run's protocol or metric scale. Without
an already validated native input report, the export remains **unverified** with
**unresolved metric scale**. An old RGB-only fit that was placed using evaluation
cameras does not become a strict RGB-only result by changing export formats.
The exporter does not establish target-robot registration or executable actions.

No GPU rerun is needed to test this hook: its tests use synthetic arrays. The
native fitting code and original reconstruction results are not modified.

## Native-result smoke test

A CPU smoke test exported Amani's existing `standard/gt512__mv/fit.pkl`: five
frames with nine Panda joints. Every joint/base value was preserved exactly.
The native path is
`/n/lab_storage/ydu_lab/Lab/akiruga/articulated_recon/results/standard/gt512__mv/fit.pkl`,
with SHA-256 `d024ead3c4b8a4d8ab1ca4f36e3bcbe40af9e244bfac5b042d691fe1a20aea22`.
The same episode's source metadata declares a 20 Hz control clock; its native
steps `[5,53,102,151,200]` therefore retain timestamps
`[0.25,2.65,5.10,7.55,10.00]` seconds. The source packet's action indices and the
source render's step/timestamp grid were crosschecked before conversion.

The current Panda v2 URDF path declared in Amani's deployed robot module was
parsed to recover joint order, which matched the native export. The historical
runtime URDF hash was not recorded and remains unavailable. Original observation
flags were also absent, so the new actor explicitly retains `observed=false` in
all five frames instead of inventing evidence. The export remains unverified with
unresolved metric scale; its original fit file hash was unchanged.

This tests faithful serialization, not reconstruction accuracy. The new artifacts
are under `results/native-urdf-export-smoke-v1/` in the cluster experiment root.
Compact local copies are in ignored `outputs/localdiagnostics/native-urdf-export-smoke-v1/`.
