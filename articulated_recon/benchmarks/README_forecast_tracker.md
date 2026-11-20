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
