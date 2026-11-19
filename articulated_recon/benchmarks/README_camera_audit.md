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

## Published-rig first-frame control

A separate **calibrated RGB diagnostic** replaced both estimated intrinsics K
and designed virtual poses C with the original RH20T camera calibration. This
is not a strict RGB-only result or a validated forecasting/planning result.
The same first source image was observed; all later source images and all three
target views were generated. No target RGB, future RGB, measured depth or object
geometry conditioned the sampler.

The control used source `036422060215` and targets `037522062165`,
`104122061850`, `104122063678`. Original publisher files were recovered from the
verified archive, not from the historically RGB-D-repaired camera bundle:

| File | SHA-256 |
| --- | --- |
| `selected-mano-trials-20261111/intrinsics.npy` | `9975592f2cfbe451adfd87a5d10a358546847476cc9380ba071ed6fb73eae91c` |
| `selected-mano-trials-20261111/extrinsics.npy` | `ea9c8237f6395ed14f9444002fd3e152161ea3f5364b69bb567ef3506fb8f384` |

The original loader's convention was preserved: invert publisher world-to-camera
extrinsics, express all poses relative to camera 0, scale K from 1280×720 to
640×360, subtract the `[0,20,640,320]` crop origin, then resize K and RGB to
320×160. Proper rotations, source identity, nonzero baselines and finite arrays
were checked. Camera baselines were 0.992, 1.272 and 1.016 metres before the
unchanged author's mean-baseline normalization. The manifest explicitly records
`condition=calibrated_rgb` and `origin=known_calibration`; these matrices are
conditioning requests, not evidence that the output obeys them.

Exact paired sampler checks passed: 17 frames, 30 sampling steps, seed 20261119,
text/history guidance `(0,0)`, cached `task_0004` text, the same config and
step-10000 checkpoint, and exactly one observed source image. Both runs made
30 source-clamp checks with zero error. The prepared source RGB byte digest was
`fa6776bc8f33d02699e47871e1abe86b5705ec4927475cbfff217ae628e4671d`;
all future/target conditioning pixels were zero. Only K and C changed
numerically. The isolated driver snapshot changed input validation and
provenance labels; the author sampler and VAE were unchanged. The saved
`driver-provenance.diff`, `frozen-config.json`, `prep-audit.json`, and
`sampling-match-audit.json` record these checks.

**Observed result:** the published rig produces visibly different tabletop
viewpoints. These roughly follow the orientations and layout in the actual
recorded target views, unlike the two virtual-rig helpers that nearly copy the
source. Object placement and invented background details still differ. This
identifies camera conditioning as an actionable contributor; it does not prove
accurate geometry or correct generated motion. K and C changed together here,
so their individual effects are not yet isolated.

The fixed source-feature audit found only 6, 0 and 0 homography inliers in the
three wide-baseline generated views, below its unchanged minimum of 8. Their
parallax/epipolar accuracy is therefore **unmeasured**, not a pass. A separate
post-generation t0 contact sheet uses the four recorded RGB views for visual
evaluation only; repaired extrinsics and depth arrays are ignored. The
recorded targets correspond to the published rig; virtual requests are different
viewpoints and are not expected to pixel-align with those recorded targets. The
generation source came through the existing video encoding and differs from
the preserved RGB packet by 0.946/255 mean absolute intensity, while source
pixels are exactly identical between the two generation runs.

The published-rig job completed in 5m47s, with a reported model-stage duration of
35.47s and peak allocated GPU memory of 7.62 GB. The earlier virtual-rig run
reported 145.86s for that stage. Both used A100 3g.20GB slices on different nodes;
loading/cache/runtime variability was not isolated. **This is not a speedup
measurement.**

Compact media and receipts are under
`outputs/rgb-only-20261119/media/wm-published-camera-control/` relative to this
directory: `camera-conditioning-comparison.mp4`, `frame-000.png`,
`frame-016.png`, `observed-targets-t0.png`, and JSON audits. The durable cluster
run is `wm-calibrated-first-frame-20261119` under the user's lab-storage directory.
No model/data assets or generated media belong in the code commit.

### Predeclared K/C crossover

Two additional conditions were frozen together before a single capped GPU job:
estimated K + published C, and published K + virtual C. They use the same
source pixels, 17 frames, 30 steps, checkpoint, seed and `(0,0)` guidance above.
Exact array comparisons establish the intended two-by-two crossing. Both are
`calibrated_rgb` diagnostics; the estimated camera component retains its
original full-source camera-estimation ancestry, so neither is relabeled as a
strict one-frame forecast. Their conditioning tensors still contain no future
or target RGB pixels. No additional variants or automatic retries are part of
this batch. Results must be reported only after both generation receipts and
the frozen-config checks pass.

### Reproduce the calibrated controls from the package

`calibrated_camera_controls.py` packages preparation and the exact two-way K/C
crossing. `world_model_views.py generate --calibrated-camera-diagnostic` is the
explicit generation opt-in; the default RGB-only path rejects supplied-camera
lineage before importing GPU dependencies. Old frozen cluster driver files
remain unchanged; packaging does not imply an additional GPU reproduction.

Prepare a verified shared `calibrated_rgb` manifest containing these numeric
NPZ artifacts (no pickle is accepted by this adapter):

- One observed source artifact with `rgb[1,H,W,3]` uint8 and `timestamps[1]`.
  Set its `camera_id` or `source_camera_id` to the source serial. If absent, the
  preparation explicitly records the caller's assumption that rig camera 0 is
  the source; it does not claim independent identity verification.
- A `known_calibration` rig artifact with `K[4,3,3]`, camera-to-world
  `c2w[4,4,4]`, original `resolution_wh[2]`, and four ordered `camera_ids`.
  For the verified RH20T files above, use `K=intrinsics[serial][:,:3]` and
  `c2w=inverse(extrinsics[serial][0])`, with resolution `[1280,720]`.
  Record the original calibration files/hashes in its provenance. The adapter
  handles source resolution, crop/resize and source-relative transforms.

The timestamp JSON contains 17 increasing output labels beginning at the one
observed timestamp; it contains no future images. Newly prepared files mark
future source-frame indices `-1` instead of inventing observations. For the
measured control, the labels were
`[0,.3,.7,1,1.3,1.6,2,2.3,2.6,3,3.3,3.6,4,4.3,4.6,4.9,5.3]` seconds.

```bash
python articulated_recon/benchmarks/calibrated_camera_controls.py prepare \
  --input-manifest camera-inputs.json --source-id source_frame \
  --rig-id published_rig --timestamps-json output-times.json --output condition

python articulated_recon/benchmarks/world_model_views.py generate \
  --condition condition --repository /path/to/author/repo \
  --config /path/to/author/config.yaml --checkpoint /path/to/checkpoint.ckpt \
  --checkpoint-receipt /path/to/model-preflight.json --output new-result \
  --seed 20261119 --conditioning first_frame_diagnostic \
  --text-guidance 0 --hist-guidance 0 --calibrated-camera-diagnostic

python articulated_recon/benchmarks/calibrated_camera_controls.py cross \
  --estimated-condition /path/to/original-17-frame-condition \
  --published-condition condition --output new-crossed-conditions
```

Use the same explicit generator command for each crossed subdirectory. No
automatic submission, installation or retry is performed. `cross` verifies
identical source pixels and timestamps, preserves full estimated-camera
ancestry, swaps only K/C arrays, and obtains all future conditioning pixels
from the zero-filled published control. The supplied-camera status remains
explicit in manifests, sampling settings and generation receipts. Focused CPU
tests cover the crop/coordinate transform, wrong source identity, default-path
rejection, future-image rejection even after valid hash updates, and exact
crossed arrays without modifying their parents.
