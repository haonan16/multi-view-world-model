# Independent Pi4 depth evaluation

`evaluate_pi4_depth.py` evaluates completed, frozen RGB-only Pi4 predictions.
It does not run Pi4, fit geometry, select a model, optimize scale or pose, or
provide reference data to an agent. There are **no real Pi4 scores yet** from
this evaluator; its tests use synthetic arrays with known errors.

The two explicit steps are:

```bash
python articulated_recon/benchmarks/evaluate_pi4_depth.py seal \
  --predictions /path/to/completed/pi4-case \
  --output /path/to/evaluator/pi4-case-seal.json

python articulated_recon/benchmarks/evaluate_pi4_depth.py evaluate \
  --predictions /path/to/completed/pi4-case \
  --seal /path/to/evaluator/pi4-case-seal.json \
  --reference /path/to/evaluator/measured-depth-stride5/hocap_box/reference-manifest.json \
  --output /path/to/evaluator/pi4-case-depth-report.json
```

The first command requires `inference.json` with completed RGB-only status and
verified geometry/source lineage. It records hashes for the inference receipt,
geometry, input manifest and original isolated source RGB. It never accepts a
reference argument. The second command verifies that seal **before opening any
reference**, and checks the consumed files again before writing a report.
Both commands refuse to overwrite their report. A seal is an explicit workflow
boundary, not operating-system isolation or proof that a separate process has
never seen evaluation data.

The downloaded subset of a Modal run is sufficient if its locally materialized
manifest authenticates `geometry.npz` and the original RGB. Native raw outputs
may remain remote; the evaluator never substitutes a rendered preview for depth.

## Pixel and frame correspondence

The reference contract is `hocap-depth-reference-v1` from the fixed HO-Cap
selection. It binds the source RGB bundle, nominal timestamps, original frame
IDs, uint16 PNG hashes and official camera-Z/units convention. The evaluator
checks `source_input_manifest_sha256` against the Pi4 inference receipt's
`source_manifest_sha256`: the curator's other field named
`source_manifest_sha256` refers to a separate source-provenance document.

`geometry.source_frame_indices` indexes the original RGB bundle. Reference
`original_frame_ids` adds the declared clip offset; no nearest-time matching or
frame shifting is allowed. Every predicted RGB frame must equal the exact
LANCZOS/multiple-of-16 resize of its selected original source frame. Predicted
`depth_m` must equal the Z component of native `local_points_m`.

Original aligned camera-Z depth is converted with `uint16 * 0.001` metres.
It is sampled onto the model grid at nearest pixel centres, using the same
image extent; there is no crop, bilinear depth averaging, scale alignment or
hole filling. Zero depth remains invalid. The current reference set contains
source indices `0,5,...,115`; predictions at other indices are explicitly
reported as reference-unavailable. A listed reference with a missing or changed
file is an integrity error, rather than a missing measurement silently skipped.

## Metrics and limitations

Per frame and pooled across valid paired pixels, the report provides absolute
error (mean/median/p90), signed bias, RMSE and absolute relative error. It also
reports mean frame MAE and separate model/sensor/paired coverage counts.
Missing predictions or sensor holes are never counted as correct pixels.

The confidence comparison selects the upper half of **native model scores**
using per-frame midrank percentiles, before looking at sensor validity or
category labels. Ties are retained, so actual retention may exceed half and is
reported. It is not a calibrated probability of metric accuracy. Categories do
not rerank their own pixels. Optional hand/object/background statistics require
explicit verified publisher segmentation IDs; undeclared IDs remain unknown.
The current 75 MB reference collection has no segmentation labels, so category
results remain unavailable instead of treating the entire image as background.

Camera drift is a separate diagnostic: predicted metric c2w transforms are
compared to the first predicted camera, reporting translation in metres and
rotation in degrees. No dataset camera matrix is used. Interpreting this motion
as drift relies on the documented fixed-camera capture convention, which the
report repeats. It does not establish absolute camera accuracy, contact quality,
or successful robot execution.

Sensor depth has holes and boundary noise; it is not perfect geometry. Frames
and pixels from the same clip are correlated. Held-out reports must stay out of
active fitting/model selection; no result is fed back into frozen predictions.

The synthetic test suite checks preserved metric bias, native frame/timestamp
alignment, nearest resampling and holes, confidence selection independent of
reference validity, missing references, segmentation unknowns, relative camera
motion, incomplete-run rejection, and prediction/reference hash tampering.
