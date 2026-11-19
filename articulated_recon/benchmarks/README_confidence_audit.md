# Independent depth-confidence audit

`depth_confidence_audit.py` scores frozen metric depth against separately held
sensor depth. It is an evaluator utility, not a reconstruction stage. It never
rescales depth, changes geometry, or returns points to a fitter.

Before scoring, establish the original camera/frame identifiers, sensor units,
and RGB/depth registration from the producer. `verify_exact_frames` rejects
timestamp approximation or changed RGB pixels; equal image sizes alone do not
establish registration. Record hashes of the prediction, source RGB, masks and
reference, and independently verify raw depth samples where available.

`audit_category` reports absolute/relative error, signed metric bias, valid
sensor coverage, and five equal-count confidence bins. Its primary
`within_frame_category_bins_low_to_high` groups pixels separately within each
frame/category, then reports both mean-frame and pixel-weighted errors. The
separately labeled pooled bins sort the original ranks over all frames.
`frozen_rank_top50` preserves the input rank>=0.5 selection; it does not recompute
a top-half selection inside an object mask. Empty categories have no score.

For UniDepthV2, the API's `confidence` is documented as estimated scale-invariant
log error. The frontend converts it to inverse ranks separately for each input
frame. These ranks are neither probabilities nor certified absolute-scale
uncertainties. Ranking errors against unscaled measured depth tests practical
metric-error selection; it does not reproduce the confidence training loss.

Use predicted masks consistently, report missing sensor pixels, and keep
boundary erosion as an explicitly separate diagnostic. Correlated pixels and
adjacent video frames are not independent experiments. A single sensor clip
does not establish dataset-general uncertainty calibration. Keep all oracle
measurements out of active reconstruction and planning workers.

Validation: `python -m pytest tests/test_depth_confidence_audit.py -q` from
`articulated_recon/mano`.
