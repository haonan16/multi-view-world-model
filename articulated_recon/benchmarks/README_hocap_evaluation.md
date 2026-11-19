# Frozen HO-Cap evaluation

These adapters evaluate completed source-only reconstructions. They do not fit,
align, rescale, plan, or access a simulator. Dataset annotations and published
calibration are evaluator-only; none of the resulting scores feed the frozen
reconstruction. File identity and declared provenance are enforced at the
interface, not through operating-system process isolation.

The November 19 bottle case is `subject_3/20231024_162756`, original frames
34–153, observed camera `117222250549`, 120 frames at 30 fps. The other two
recorded cameras were held out. Extraction used only previously cached files
from the verified workspace archive; no new dataset downloads were needed.
Licensed MANO assets, RGB, annotation labels and reconstructed GT surfaces stay
outside Git.

## Hand annotations

`hocap_annotation_audit.py` binds the completed fits and their observation
manifests before opening evaluator files. Source RGB and timestamps must equal
the selected recorded camera exactly. Every consumed annotation must belong to
the verified archive-extraction receipt. The adapter rechecks all input hashes
before saving a new report, and refuses to overwrite reports.

HO-Cap stores right/left parameter arrays with absent hands padded by -1.
This sequence has only a right-hand annotation. Its raw 45-PCA MANO parameters,
subject betas and world translation are passed through the raw MANO layer;
the wrist-recentering fitting adapter is deliberately not used. Output units are
metres. The original published source-camera transform converts the reference
mesh to the method's source-camera coordinates.

The archived observation bundle uses a different common rigid world frame.
All three relative camera transforms agree with the original publisher data
within 1.03e-7 matrix error, and all three K matrices match exactly. This is a
world-coordinate change, not a per-camera repair. It does not alter the held-out
projections. The separately saved `camera-provenance.json` binds this check.

Fingertip definitions differ. The method's SMPL-X convention uses
`[744,320,443,554,671]`; upstream manopth uses right-hand
`[745,317,444,556,673]`. GT-parameter-to-label comparison, without using method
predictions, establishes `[745,319,444,556,673]` for these cached labels.
The entire reconstructed annotation reproduces the 21 labels to below 0.001 mm.
All three joint-score conventions are reported, without selecting the best one.
Surface errors use common MANO topology; nearest-vertex distances are also
reported. Opposite-chirality correspondence is a diagnostic, not a correction.

The left-v1 fit predates the explicit `source_camera` frame name. Its identity
camera and fixed identity base were audited separately. Reproducing that
historical comparison requires `--allow-legacy-frame`; identity camera matrices
alone do not prove the base-frame convention. New fits must declare the frame.

## Object CAD

`hocap_object_audit.py` binds both mesh tracks to the same frozen RGB-only hand
and verifies their output manifests and all three required output files. It uses
the cached `G21_1` CAD in metres and each exact frame's camera-relative
`obj_poses[0]`. There is no pose, rotation, scale or shape registration.

Each mesh receives 10,000 area-uniform surface samples, with seed 100 for CAD
and 101 for predictions. Symmetric nearest-sample distance approximates surface
distance; it does not assume common topology. Centers are transformed
surface-area centroids, avoiding arbitrary mesh origins. Sorted oriented-box
dimensions provide a limited intrinsic-size diagnostic. Missing v1 poses are
excluded: common four and recovered nine keyframes are reported separately.

## Results from unchanged frozen outputs

| Hand metric | Left v1 | Right v2 |
|---|---:|---:|
| Absolute wrist error | 81.30 mm | 80.19 mm |
| Absolute joint error, original exported convention | 62.63 mm | 76.17 mm |
| Wrist-relative joint error, original exported convention | 22.17 mm | 9.64 mm |
| Absolute corresponding-vertex error | 62.93 mm | 75.99 mm |
| Wrist-relative corresponding-vertex error | 27.96 mm | 9.40 mm |

Wrist-relative metrics subtract each hand's wrist translation only. They do not
rotate or rescale either hand. The hand's articulation improved substantially;
its global metric placement did not. These references are fitted dataset MANO
annotations, not exact anatomy.

| Object metric | Mesh v1 | Mesh v2 |
|---|---:|---:|
| Observed fitted keyframes | 4/13 | 13/13 |
| Common-four centroid error | 96.04 mm | 102.31 mm |
| Common-four symmetric sampled surface distance | 66.37 mm | 69.56 mm |
| Sorted OBB dimension MAE | 15.33 mm | 12.20 mm |

CAD sorted dimensions are 66.53 × 103.87 × 208.01 mm. Predicted dimensions are
32.62 × 94.19 × 205.59 mm (v1) and 42.05 × 97.59 × 202.17 mm (v2).
The combined RGB association and hand-occlusion update recovered track coverage
and improved these dimensions, but did not improve placement on the matched
four frames. It is not an isolated association-only ablation. The all-13 v2
centroid error is 101.27 mm; it should not replace the matched-four comparison.

## Reproduce the evaluator

Run from `articulated_recon/benchmarks` in the existing environment with torch,
smplx, numpy, scipy, PyYAML and trimesh. The saved input/output paths are examples
from the cluster; choose new report filenames. These commands perform only
evaluation of frozen outputs, never reconstruction or physics replay.

```bash
RUN=/n/lab_storage/ydu_lab/Lab/surajrdy/rgb-scene-pipeline-20261119
AUDIT=/n/lab_storage/ydu_lab/Lab/surajrdy/bottle-annotation-audit-20261119
REF=/n/lab_storage/ydu_lab/Lab/surajrdy/cluster-workspace-20261113/mano-full-calibrated-20261111/inputs/bottle/obs.npz
MANO=/n/lab_storage/ydu_lab/Lab/surajrdy/cluster-workspace-20261113/mano-full-pipeline-20261111/assets/mano

python hocap_annotation_audit.py \
  --runs "$RUN/results/mano-rgb-native-v1" "$RUN/results/mano-rgb-side-prior-v2" \
  --case bottle --annotations "$AUDIT/raw-samples" \
  --verification "$AUDIT/raw-sample-verification.json" --reference "$REF" \
  --source-index 1 --camera-ids 115422250549 117222250549 043422252387 \
  --original-start 34 --side right --mano-models "$MANO" \
  --reference-tip-vertices 745 319 444 556 673 --allow-legacy-frame \
  --output "$AUDIT/reproduced-hand-report.json"

python hocap_object_audit.py \
  --tracks "$RUN/results/bottle-source-mesh-track-v1" "$RUN/results/bottle-source-mesh-track-v2" \
  --hand-run "$RUN/results/mano-rgb-side-prior-v2" --case bottle \
  --annotations "$AUDIT/raw-samples" --verification "$AUDIT/raw-sample-verification.json" \
  --reference "$REF" --source-index 1 --original-start 34 --object-id G21_1 \
  --cad "$AUDIT/raw-samples/cleaned_mesh_10000.obj" \
  --output "$AUDIT/reproduced-object-report.json"
```

## Held-out videos

`heldout_mano_audit.py --mesh-track` renders the actual frozen SAM3D mesh plus
MANO hand on source and two held-out recorded views, using only published
calibration for projection. It uses no sensor-depth occlusion. Object poses
between the 13 fitted keyframes are linearly interpolated in translation and
Slerp-interpolated in rotation for display only; frames are labeled accordingly.
Scale/shear, nonfinite and non-SE3 pose matrices are rejected before interpolation.

```bash
python heldout_mano_audit.py \
  --run-dir "$RUN/results/mano-rgb-side-prior-v2" --case bottle \
  --reference "$REF" --source-index 1 \
  --camera-ids 115422250549 117222250549 043422252387 \
  --mesh-track "$RUN/results/bottle-source-mesh-track-v2" \
  --output "$AUDIT/reproduced-heldout-video"
```

Original reports: `outputs/rgb-only-20261119/diagnostics/bottle-hand-annotations/`
(`report-final.json`, `object-CAD-report.json`, `camera-provenance.json`).
Video: `outputs/rgb-only-20261119/media/bottle-heldout-mesh/heldout-mano.mp4`.
The source view can appear plausible while held-out views expose metric
misplacement; the video is a projection diagnostic, not successful execution.
