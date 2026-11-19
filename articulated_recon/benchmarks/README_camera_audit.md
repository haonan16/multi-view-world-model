# Requested-camera consistency diagnostic

`helper_camera_audit.py` asks whether generated helper views show the camera
motion requested from the model. It uses the original single RGB view, its
**predicted** depth/intrinsics and the designed virtual cameras. No measured
depth, supplied dataset cameras or independent evaluator assets are consumed.

SAM3 hand/block candidates are removed from source-feature detection, with a
four-pixel margin at 320×160. Mutual SIFT matches (ratio 0.75) are filtered by a
generic 3-pixel homography RANSAC. This filter does not use the requested camera.
Source depth then predicts where these features should appear under that camera.
The report compares requested-pose reprojection against an identity/copy baseline.

The raw decoded source view is a VAE-only control. View0 restored from the exact
source RGB is the observation; views1–3 are generated. Native depth and K receive
the same `[0,20,640,320]` crop and 320×160 resize used for generation. Input hashes,
time indices and source pixels are verified before measuring.

## First block result, 2026-11-19

Across all 25 sampled timestamps of `block-rgb-helper-v1`:

| Compared view | Background matches | Median measured shift | Median requested shift | Median requested reprojection error |
| --- | ---: | ---: | ---: | ---: |
| Raw decoded source control | 5,110 | 0.17 px | approximately 0 px | 0.17 px |
| Generated helper 1 | 4,369 | 0.27 px | 20.72 px | 20.63 px |
| Generated helper 2 | 4,278 | 0.30 px | 23.13 px | 23.17 px |
| Generated helper 3 | 995 | 0.78 px | 23.52 px | 23.54 px |

Identity projection was closer than the requested-pose projection for every
retained helper match. This supports **source-like camera placement in the
matched static background**, rather than the intended novel viewpoint. It does
not mean every generated pixel was copied or every object stayed unchanged.

These matches are correlated, mostly planar-background observations. Estimated
depth/K may be inaccurate; segmentation, changed textures and occlusion affect
which features can be matched. This is an internal consistency result, not a
ground-truth camera estimate or a physical-execution result.

The author sampler and checkpoint camera mode were checked: PRoPE conditioning
was enabled, c2w baseline normalization matched the training loader, and source
latents were clamped. Training full-view history also supplied target-view first
frames, whereas this experiment supplies only the full source sequence. That
difference and the 25-frame versus 17-frame training horizon are possible causes
to test, not causes established by this diagnostic.

```bash
python articulated_recon/benchmarks/helper_camera_audit.py \
  --generated /path/to/results/block-rgb-helper-v1 \
  --condition /path/to/conditions/block-rgb-helper-v1 \
  --geometry /path/to/results/block-native-reconstruction/unidepth-v1/geometry.npz \
  --masks /path/to/predictions/sam3-block-native/masks.npz \
  --output /path/to/new-camera-audit
```

`report.json` contains aggregate and per-frame measurements. The three PNGs show
green measured match locations; magenta lines terminate at the requested-camera
projections. Images use a 2× display scale; all reported errors use native pixels.

## Bounded guidance comparison

The follow-up experiment freezes three settings on the **same 17-frame prefix**,
seed and virtual cameras: text/history guidance `(2,1)`, `(2,0)` and `(0,0)`.
The first pair isolates history guidance; the third uses the author's validation
defaults. Source conditioning spans 0–5.3 seconds, rather than the original
0–7.9 seconds. Camera design remains inherited from the full source clip. This
comparison cannot establish reconstruction of the omitted final motion.

`world_model_views.py prepare-prefix` preserves the original K and c2w arrays
and takes the first existing timestamps without resampling the full clip.
`generate --text-guidance ... --hist-guidance ...` saves its sampling config
before inference and checks the appropriate number of source-clamped forwards.
The isolated cluster snapshot and `ablation-spec.json` pin inputs, code and the
three settings before the capped one-GPU job starts.

Setting text guidance to zero **still uses the cached task text**. For an input
with no appropriate cached description, `--null-text --text-guidance 0` explicitly
uses null context and reports `task_text_used=false`. It never silently substitutes
the block instruction for a bottle task.

A newly encoded instruction can instead use `--text-embeddings FILE
--text-embedding-receipt RECEIPT --prompt-key KEY`. This accepts the author's
task/null dictionary only after a completed receipt binds its hash, encoder and
tokenizer implementation, exact text, tensor shapes and dtype. The receipt and
embedding are rechecked after inference. Task text is recorded explicitly as
conditioning metadata, not mislabeled as ground-truth geometry or global model
weights. This metadata does not add a new modality to the input-protocol DAG.
