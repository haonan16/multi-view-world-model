# Frozen real-video representation split

The selection in `real_video_split_20261119.json` was frozen at
2026-11-20 02:33:29 UTC, before any new representation fits or heldout scores.
Its SHA-256 is
`f38c0ec93d6edaae77a040e95724091fd4abeedbd20102a9f1069315f75df1a5`.

| Case | Role | HO-Cap sequence | Original frames | Source camera |
|---|---|---|---|---|
| Flat box | Train | `subject_9/20231027_123725` | 53–172 | `117222250549` |
| Pump bottle | Train, reused development | `subject_3/20231024_162756` | 34–153 | `117222250549` |
| Spatula | Heldout | `subject_9/20231027_125315` | 68–187 | `117222250549` |

Each source contains 120 original 640×480 RGB frames at nominal 30 fps, without
cropping. The common camera was chosen because all three previously cached
packages contain it. Box and spatula are additional clips relative to the old
bottle development case. RH20T block is excluded because it is known to occur
in world-model training. Absence of HO-Cap from all pretrained models' training
data has not been established.

The historical November 11 shortlist and windows were annotation-assisted.
Box and spatula also share a subject. This is a small curated feasibility split;
it cannot establish population or subject generalization. Training on the bottle
does not make it a fresh test example.

## Method and evaluator boundary

`real_video_data.py plan` reads the existing archive's integrity index and binds
an exact member allowlist. Missing source frames fail planning; the tool never
silently substitutes another clip or camera. `extract` checks the sealed plan
hash and every selected archive member's size and SHA-256. It copies only source
JPEGs into `method/<case>/rgb/`. Sensor depth, label arrays, geometry metadata,
calibration, and sampled other-camera RGB go into `evaluator/<case>/`.

`pack` takes only the isolated `method/` directory. It verifies each source hash
and image size, then writes the existing frontend interface:

- `rgb.npz`: exactly `rgb[T,H,W,3]` uint8 and `timestamps[T]` float64;
- `input-manifest.json`: observed `source_rgb` root, unresolved scale and robot registration;
- `first-frame.png`: decoded first source image;
- `source-manifest.json` and `extraction-audit.json`: original frame IDs, hashes, source camera, and timestamp provenance.

Timestamps are **nominal original frame ID / 30 seconds**, not measured sensor
clock readings. The original JPEGs remain available to verify the decoded RGB.
No camera matrices, depth, annotations, or other recorded views enter the method
bundle. Interface separation is audited; it is not an operating-system access
control barrier between processes owned by the same user.

The small training set may use model-derived pseudo targets only. Training on
sensor depth, dataset MANO/object labels, or dataset calibration would violate
this condition. Keep the whole spatula sequence out of parameter fitting,
hyperparameter selection, and model-selection feedback. Seal every compared
variant before the evaluator opens references; do not send heldout metrics back
to active fitting or training agents.

Evaluator preparation uses source-camera depth and labels for all 120 frames,
plus 13 uniformly spaced frames from two other recorded cameras. Missing optional
references are recorded explicitly. Preparation does not produce scores; absent
measurements stay null. Object CAD may need a separately verified evaluator-only
asset extraction after reading sequence metadata.

## Reproduction

Use storage under the user's lab directory. The verified cached archive is
`/n/lab_storage/ydu_lab/Lab/surajrdy/local-experiment-archive-20261112/workspace.tar.zst`.
No assets belong in Git, and no large dataset download is needed. Discover the
nearby verified JSON or JSONL index before setting `INDEX` below. The commands
refuse to overwrite existing plans or bundles.

```bash
python real_video_data.py plan \
  --selection real_video_split_20261119.json \
  --archive-index "$INDEX" --output "$DATA/extraction-plan.json"

# Copy the printed plan_sha256 into PLAN_SHA before extraction.
python real_video_data.py extract \
  --plan "$DATA/extraction-plan.json" --plan-sha256 "$PLAN_SHA" \
  --archive "$ARCHIVE" --output "$DATA/bundles"

python real_video_data.py pack --method-root "$DATA/bundles/method"
```

Extraction streams the existing archive and writes only selected members. Use a
small CPU batch job if archive decompression would burden a login node. Packing
requires NumPy and Pillow; planning and extraction use the Python standard
library plus the existing `zstd` binary. These commands launch no models.

Six focused tests cover RGB/evaluator separation, compatibility with the existing
frontend loader, original timestamps, missing source frames, split overlap,
development-as-heldout rejection, modified plans or bytes, and overwrite guards.
They passed in the existing offline dependency environment.

## Materialized source-only fallback

The authenticated cluster connection was unavailable during preparation. A
bounded public-mirror fallback retrieved exactly the frozen 360 JPEGs:
27,465,200 source bytes in total. It read ZIP central-directory indexes first,
then requested only selected compressed member byte ranges. It did not download
the full 19.3 GB subject-3 or 12.6 GB subject-9 archives, labels, depth, calibration,
or model weights. The mirror revision is pinned to
`d9a562638bc4eda48451ce1211385371bbd17be0`.

Each downloaded member passed the pinned ZIP index's CRC32 and uncompressed-size
checks; SHA-256 hashes were then recorded for every JPEG and packed RGB bundle.
Whole-ZIP LFS hashes are recorded as publisher-declared, **not independently
verified**. Byte identity against the older cluster archive remains **unverified**
until its integrity index is accessible. This fallback did not substitute a
different sequence, camera, frame window, or split.

The prepared bundles and receipts are under
`outputs/real-video-20261119/mirror-rgb/`, outside Git. Its `method/<case>/`
directories are ready for the existing `rgb_frontend.load_source_bundle` API.
`download-plan.json`, `download-receipt.json`, and `source-validation.json`
preserve the extraction and validation evidence. The `code/` directory preserves
the exact fallback downloader and preview producer.

| Case | Packed `rgb.npz` SHA-256 |
|---|---|
| `hocap_box` | `5949798d7260ab910132393dc6fcc345f8b6bfb67998635045b8d39d0c2da2fc` |
| `hocap_bottle` | `88b0485310418d7c8f76fc924b9c31ac45ee01d739fb60efb539bd98d0b9557c` |
| `hocap_spatula` | `4699492e5e1581f02b6f7ae6387ccfb6ee46b72aa73319e19e4f6315818b4862` |

`source-preview.mp4` shows all three source clips side by side at their nominal
30 fps, preserving each 640×480 image. Only train/heldout labels are added; no
model or annotation overlays are used. The tables contain distractor objects.
In the spatula case the manipulated utensil is pink; a teal spatula remains on
the table. `method/target-prompts.json` fixes the source-RGB-derived prompt
`pink spatula being lifted` before fitting. New representation scores remain
unmeasured.

## Predicted-geometry previews

`render_pi4_preview.py` reads only a completed Pi4 run's `geometry.npz` and
`inference.json`. It renders source RGB at the model's resolution, predicted
local-Z depth, and a virtual orbit around the native predicted world points.
Colors come from all sampled source pixels; no task masks, evaluator references,
generated RGB, mesh completion, or hidden-surface synthesis are used.

The point panel uses a constant set of at most 60,000 flattened image indices,
one sequence-wide clipping box, fixed framing, and one-pixel nearest-depth
rendering. The first predicted camera defines the display orientation. The
depth panel uses one sequence-wide 2nd–98th percentile color range. These are
display choices, not geometry corrections or per-frame alignments.

```bash
python render_pi4_preview.py --run-dir "$PI4_RUN" --output "$PREVIEW"
```

The default video is 30 fps. For the fixed stride-five inputs it repeats each
6 Hz prediction five times, including its virtual camera view; geometry and RGB
are not interpolated. `render.json` preserves input hashes, original timestamp
and source-index maps, repeated-frame mapping, display clipping, virtual cameras,
sampling indices, and the fact that metric scale is model-estimated. Six tests
check synthetic projection, rigid-coordinate invariance, nearest-point
occlusion, shared clipping, frame timing, and actual MP4 encoding. They are
renderer tests, not reconstruction-quality results.

## Evaluator-only measured-depth references

`outputs/real-video-20261119/mirror-rgb/evaluator/measured-depth-stride5/`
contains 72 original depth PNGs and three original sequence metadata files,
7,421,896 bytes in total. Each case uses source indices `0,5,...,115`; original
IDs are those indices plus its frozen window start. `references.json` binds the
three `hocap-depth-reference-v1` manifests. They record source RGB bundle and
JPEG identities, exact original IDs/timestamps, PNG hashes, units, convention
evidence, and zero-hole preservation. Their verification status is complete;
evaluation status remains unmeasured and scores are null.

The pinned official [sequence loader](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/loaders/sequence_loader.py)
requests a scale of 1000, and the official
[depth reader](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/utils/io.py)
divides original integer depth by that scale. The references therefore declare
`scale_to_m=0.001`. The loader deprojects with color-camera inverse intrinsics
and the unnormalized `[u,v,1]` ray, establishing the consumed color-grid camera-Z
convention. It reuses one published RealSense extrinsic per sequence; the
fixed-camera flag records that capture convention, not a new measurement of
microscopic camera motion or calibration accuracy.

Label NPZ files were skipped: their ZIP central directory alone needs 77,630,789
bytes, exceeding the 75 MB combined reference budget. The reader stopped before
that transfer. Reference manifests set label files and label ID maps to null;
object/hand/background scores requiring those masks remain unavailable. The
download and finalization receipts confirm that no predictions were opened and
no metrics were computed during reference preparation.
