# Native SAM3 video tracking for generated forecasts

`forecast_tracker.py` runs the cached Hugging Face `Sam3TrackerVideoModel`, the
same learned video-memory API used in the previous MANO pipeline. It replaces
independent block detections with a temporally tracked object. It does not
forward-fill old masks or use geometry to fabricate a missing detection.

Each of the four views is tracked independently. View 0 starts from a unique
SAM3 block mask on the original-resolution observed first image, cropped and
nearest-neighbor resized using the authenticated world-model conditioning
transform. Each helper view starts from its unique generated frame-zero SAM3
mask. Missing/ambiguous seeds leave that entire view unsupported. No later
recorded image, hand annotation, manual click or simulator state is used.

```bash
python benchmarks/forecast_tracker.py \
  --helper /path/results/generated \
  --condition /path/inputs/forecast-condition \
  --baseline-masks /path/predictions/forecast-framewise-masks \
  --initial-masks /path/predictions/initial-sam3 \
  --checkpoint /n/netscratch/ydu_lab/Lab/akiruga/hf_cache/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7 \
  --output /path/predictions/forecast-video-tracker \
  --preflight
```

Remove `--preflight` only inside the bounded GPU allocation. The verified cached
runtime is `/n/lab_storage/ydu_lab/akiruga/.conda/envs/test2/bin/python`, with
Transformers 5.8.0. Models/processors load from that local snapshot with
`local_files_only=True`; there are no downloads. Inference uses bfloat16,
`frame_idx=0`, `obj_ids=[1]`, a predicted mask input, and forward-only
`propagate_in_video_iterator`. Source/helper streams never share a tracker
session or memory.

An explicit optional source-seed variant adds
`--source-association-config /path/initial-scene/initializer-config.json`.
It recomputes `source_geometry_v2.associate_masks` on the single native source
image's SAM3 candidates. Its fixed heuristic combines detector score, relative
mask area and overlap with the predicted table's image bounding box; the
temporal continuity term is zero for this singleton input. It accepts a winner
only when the saved configuration's margin separates the candidates. Missing
or ambiguous cases stay unsupported. This is a task-selection hypothesis,
not ground truth. It never reads a saved selected row, initial fitted geometry,
depth or future image to decide the seed. Config and selector-code hashes,
all candidate scores, the actual margin and selected row are recorded. Helper
views keep the original unique-candidate rule. Without this flag, the original
source policy is unchanged.

`--helper-seed-transport` replaces the helper's independent candidate choice
with automatic first-frame source-to-helper mask transport. Mutual SIFT matches
(ratio 0.75) and a 3-pixel RANSAC homography use source background outside all
predicted object/hand masks with a four-pixel margin. The transported mask is
a **background-plane correspondence hypothesis**: object parallax, generated
distortion and hallucinations can still make it wrong. It uses neither requested
camera matrices nor later frames nor simulator information.

Fixed gates require at least 12 inliers, 35% inlier fraction, broadly distributed
source/target matches, median reprojection error ≤2px, 90th percentile ≤3.5px,
at least three inliers within 40px of the source block, and no more than 8px
extrapolation outside the source inlier hull. Projected area must remain within
0.25–4× its source area and contain at least 12 pixels. Failed correspondence
leaves the helper unsupported; it does not fall back to an unrelated detection.
All thresholds, homographies, error measures and seed centers are recorded.
Helpers still have independent video memories after seeding; cross-view object
identity is estimated, not verified. Source-only output ancestry stays unchanged.

The output retains native full-resolution mask logits and object-presence
logits in `tracker-logits.npz`. Default gates are mask logit > 0, sigmoid object
presence > 0.5, and at least 12 mask pixels. Occluded/absent/small predictions
are excluded from the candidate packet; raw scores remain available. Sigmoid
presence is **not calibrated mask accuracy**, and the seed frame is explicitly
conditioned on a predicted mask. Detector scores are never replaced by a
constant 1.

`masks.npz` retains the existing framewise hand/table candidates unchanged and
replaces only the block rows. `score_kind=0` denotes those original detector
scores; `score_kind=1` denotes tracker object-presence scores. Its schema remains
compatible with forecast geometry/contact fitting. `source-only/` contains
only the independently tracked view 0, its logits and its own manifest; helper
RGB is absent from its consumed ancestry. Distinct terminal/raw/source-mask
artifact IDs allow all-view geometry and source-only contacts to be combined
without confusing their different files.

The tracker can improve temporal identity when framewise text detection drops
out, but it can drift or follow a model-generated hallucination. Report mask
coverage and geometry/contact outcomes separately. Passing the CPU adapter
tests is not evidence of successful tracking on the actual generated clip.
