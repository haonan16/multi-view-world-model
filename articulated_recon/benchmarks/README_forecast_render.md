# Forecast reconstruction video

`render_forecast_reconstruction.py` renders actual generated RGB and a completed
`forecast_geometry.py` bundle. The left panel shows all four native views with
the fitted block wireframe. The right panel is a separate 3D display orbit of
the shared box, estimated pose track, RGB-colored predicted depth points and
the initial predicted table plane. It does not load simulator/evaluator data or
claim a MANO hand mesh. The finite table patch is a display choice, not estimated
table dimensions.

```bash
python benchmarks/render_forecast_reconstruction.py \
  --helper /path/to/generated-rgb \
  --bundle /path/to/completed-forecast-fit \
  --output /path/to/new-render
```

Defaults are 30 fps output, a 0.5-second initial hold and a 0.5-second final hold.
The 17-frame, 3 fps forecast has a 5.333-second prediction horizon; the default
video contains 190 frames over 6.333 seconds including holds. The displayed
native frame/time and holds are labeled explicitly. No additional world-model
prediction or interpolated RGB/depth image is manufactured.

Native RGB and its projected wireframe stay together while held. The 3D panel
uses SLERP rotations and linear translations only between supported poses,
labeled “interpolated 3D pose.” Its point cloud remains the native held cloud.
Unsupported intervals do not receive a smooth invented pose. Use
`--no-interpolate-3d` to hold all 3D poses as well. The moving display camera is
an illustrative orbit, not a predicted/observed camera trajectory.

For the one-camera `source_unposed` fit, all four raw generated images are still
shown, but helper-view wireframes are omitted and labeled “no fitted camera.”
The renderer does not silently project through requested cameras that were
absent from that fitted condition. Four-camera helper fits use their actual
fitted camera estimates for all wireframes. Predicted hand-mask pixels hide
overlaid block edges where that mask is available.

Outputs are `forecast-reconstruction.mp4`, three PNG previews and
`render-receipt.json`. The receipt records input/scene/video hashes, native and
display timing, exact held-frame mapping, point counts, box dimensions, orbit
and interpolation semantics. The renderer verifies completion/provenance and
pixel/timestamp identity before rendering, then decodes the saved MP4 to check
dimensions, frame count and codec delivery. Existing output directories are
never overwritten.

The runtime needs NumPy, SciPy, Pillow, OpenCV and an existing `ffmpeg` binary
or `imageio-ffmpeg`. No network download or model job is launched. A real encoded
synthetic-fixture video is covered by
`mano/tests/test_render_forecast_reconstruction.py`; those tests establish
software behavior, not recorded-case reconstruction accuracy.
