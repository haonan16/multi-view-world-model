# Matched source-depth frontend comparison

The block comparison uses the same recorded RGB sequence, SAM3 candidates,
estimated UniDepth focal length, fixed deployment, and planning configuration.
DA3 Metric Large provides focal-normalized monocular depth; the frontend uses the
released focal/300 conversion on the processed grid with the **estimated** focal
length. Dataset calibration, sensor depth, and evaluation geometry are absent.
DA3 and UniDepth therefore share a learned camera anchor; they are not two
independently calibrated metric-depth systems.

DA3's standalone metric head supplies no confidence. Its stored all-ones array is
only a neutral compatibility weight. Both frontends use `point_policy='all'` for
this comparison. Rank-based filtering/weighting is rejected when confidence is
explicitly unavailable. Earlier UniDepth rank experiments remain separate.

## Frozen comparison stages

1. Verify complete inference receipt, geometry hash, materialized RGB-only
   lineage, and identical RGB/masks/cameras/timestamps. Pending inference is not
   read as a partial result.
2. `source_depth_audit.py` measures raw table drift relative to each method's own
   frame-zero predicted plane and raw foreground depth distributions on identical
   source masks. It does not align one method to the other or to a reference.
3. Apply the unchanged static-background calibration configuration. Rejected
   calibration stops that branch; bounds and thresholds are not relaxed.
4. Fit shared shape/motion with all valid object depth points, source silhouettes,
   temporal smoothness, and uncertain RGB support.
5. Compare global nonpenetration with free XYZ dimensions against the fresh
   SAM3D aspect-ratio prior with one fitted metric size. Preserve both variants.
6. Apply the same RGB phase estimator and attach the existing shared deployment.
   Compile TCP plans using the existing planner guards. No simulator feedback is
   consumed by these stages.

All geometry is estimated. A smaller background residual measures consistency,
not true depth accuracy. Foreground spread mixes actual shape, visibility, and
prediction error; smaller spread is not automatically better. Physically feasible
geometry and compiled TCP commands are not evidence of successful Panda execution.

The comparison runs from the isolated `code-variants/source-geometry-v2` tree and
writes `results/block-depth-all-comparison-v1/`, retaining raw audits, both
frontends, both shape variants, configs, provenance, and rejected attempts. The
DA3 inference and subsequent CPU experiments must actually finish before any
results are reported.
