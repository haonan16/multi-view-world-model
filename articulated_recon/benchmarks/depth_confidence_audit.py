"""Posthoc depth-error audit. Never rescales predictions or produces fitter inputs.

Callers must establish RGB/depth registration and units independently. These
checks bind exact recorded frames; matching array shapes alone is insufficient.
"""
from __future__ import annotations

import numpy as np


def verify_exact_frames(source_rgb, source_times, reference_rgb, reference_times):
    """Reject approximate timestamp matching, resampling, or different pixels."""
    if not np.array_equal(source_times, reference_times):
        raise ValueError("Recorded timestamps differ; nearest-frame matching is forbidden")
    if source_rgb.shape != reference_rgb.shape or not np.array_equal(source_rgb, reference_rgb):
        raise ValueError("Recorded RGB differs; depth grid correspondence is unverified")
    return {"frames": len(source_rgb), "rgb_exact": True, "timestamps_exact": True}


def error_metrics(prediction, reference, ranks=None):
    """Metric error in original units, never median/scale aligned."""
    if len(prediction) == 0:
        return {"pixels": 0}
    error = prediction.astype(np.float64) - reference
    absolute = np.abs(error)
    result = {
        "pixels": int(len(error)),
        "mae_mm": float(absolute.mean() * 1000),
        "median_absolute_mm": float(np.median(absolute) * 1000),
        "p90_absolute_mm": float(np.quantile(absolute, .9) * 1000),
        "bias_mm": float(error.mean() * 1000),
        "rmse_mm": float(np.sqrt(np.mean(error ** 2)) * 1000),
        "abs_rel": float(np.mean(absolute / reference)),
    }
    if ranks is not None:
        result.update(rank_min=float(np.min(ranks)), rank_max=float(np.max(ranks)),
                      rank_mean=float(np.mean(ranks)))
    return result


def metric_only_category(prediction_m, reference_m, category_mask, predicted_valid=None):
    """Depth without a confidence head: no invented scores, ranks, or bins."""
    p, r, m = map(np.asarray, (prediction_m, reference_m, category_mask))
    if p.ndim != 3 or r.shape != p.shape or m.shape != p.shape or m.dtype != np.bool_:
        raise ValueError("Expected matching T,H,W depths and boolean mask")
    if predicted_valid is None:
        predicted_valid = np.ones(p.shape, dtype=bool)
    if predicted_valid.shape != p.shape or predicted_valid.dtype != np.bool_:
        raise ValueError("Predicted validity shape/dtype mismatch")
    sensor_valid = np.isfinite(r) & (r > 0)
    model_valid = predicted_valid & np.isfinite(p) & (p > 0)
    selected = m & sensor_valid & model_valid
    per_frame = [dict(index=t, **error_metrics(p[t][selected[t]], r[t][selected[t]]))
                 for t in range(len(p))]
    values = [row["mae_mm"] for row in per_frame if row["pixels"]]
    count = int(m.sum())
    return {"category_pixels": count, "paired_pixels": int(selected.sum()),
            "paired_fraction_of_category": float(selected.sum() / count) if count else None,
            "sensor_valid_fraction_of_category": float((m & sensor_valid).sum() / count) if count else None,
            "all": error_metrics(p[selected], r[selected]),
            "mean_frame_mae_mm": float(np.mean(values)) if values else None,
            "per_frame": per_frame, "confidence_available": False}


def audit_category(prediction_m, reference_m, confidence_rank, category_mask,
                   predicted_valid=None, bins=5):
    """Equal-count rank bins within one category, plus the frozen rank>=.5 rule.

    Bins pool valid pixels across frames; temporal samples are not independent.
    Original confidence ranks remain unchanged. Per-frame means are also saved.
    Missing sensor depth is excluded and its coverage is reported separately.
    """
    arrays = [np.asarray(a) for a in (prediction_m, reference_m, confidence_rank, category_mask)]
    p, r, c, m = arrays
    if p.ndim != 3 or any(a.shape != p.shape for a in arrays):
        raise ValueError("Expected matching T,H,W arrays")
    if m.dtype != np.bool_:
        raise ValueError("Category masks must be boolean")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("Need at least two confidence bins")
    if predicted_valid is None:
        predicted_valid = np.ones(p.shape, dtype=bool)
    if predicted_valid.shape != p.shape or predicted_valid.dtype != np.bool_:
        raise ValueError("Predicted validity shape/dtype mismatch")
    sensor_valid = np.isfinite(r) & (r > 0)
    model_valid = predicted_valid & np.isfinite(p) & (p > 0)
    rank_valid = np.isfinite(c) & (c >= 0) & (c <= 1)
    paired = m & sensor_valid & model_valid & rank_valid
    pp, rr, cc = p[paired], r[paired], c[paired]
    order = np.argsort(cc, kind="stable")
    ordered = [error_metrics(pp[ix], rr[ix], cc[ix]) for ix in np.array_split(order, bins)]
    per_frame = []
    for t in range(len(p)):
        selected = paired[t]
        high = selected & (c[t] >= .5)
        low = selected & (c[t] < .5)
        fp, fr, fc = p[t][selected], r[t][selected], c[t][selected]
        frame_bins = [error_metrics(fp[ix], fr[ix], fc[ix]) for ix in
                      np.array_split(np.argsort(fc, kind="stable"), bins)]
        per_frame.append({
            "index": t, "category_pixels": int(m[t].sum()),
            "sensor_valid_pixels": int((m[t] & sensor_valid[t]).sum()),
            "all": error_metrics(p[t][selected], r[t][selected], c[t][selected]),
            "frozen_rank_top50": error_metrics(p[t][high], r[t][high], c[t][high]),
            "frozen_rank_bottom50": error_metrics(p[t][low], r[t][low], c[t][low]),
            "equal_count_bins_low_to_high": frame_bins,
        })
    def mean_frame_metric(key):
        values = [f[key]["mae_mm"] for f in per_frame if f[key]["pixels"]]
        return float(np.mean(values)) if values else None
    count = int(m.sum())
    frame_bin_summary = []
    for i in range(bins):
        rows = [f["equal_count_bins_low_to_high"][i] for f in per_frame
                if f["equal_count_bins_low_to_high"][i]["pixels"]]
        total = sum(row["pixels"] for row in rows)
        frame_bin_summary.append({
            "pixels": total, "frames": len(rows),
            "mean_frame_mae_mm": float(np.mean([r["mae_mm"] for r in rows])) if rows else None,
            "pixel_weighted_mae_mm": sum(r["pixels"] * r["mae_mm"] for r in rows) / total if total else None,
            "mean_frame_abs_rel": float(np.mean([r["abs_rel"] for r in rows])) if rows else None,
        })
    return {
        "category_pixels": count, "paired_pixels": int(paired.sum()),
        "sensor_valid_pixels": int((m & sensor_valid).sum()),
        "predicted_valid_pixels": int((m & model_valid).sum()),
        "paired_fraction_of_category": float(paired.sum() / count) if count else None,
        "sensor_valid_fraction_of_category": float((m & sensor_valid).sum() / count) if count else None,
        "all": error_metrics(pp, rr, cc), "equal_count_bins_low_to_high": ordered,
        "within_frame_category_bins_low_to_high": frame_bin_summary,
        "binning": "Stable sort original frozen rank within category, pooled across frames; equal counts; ties may span bins",
        "frozen_rank_top50": error_metrics(pp[cc >= .5], rr[cc >= .5], cc[cc >= .5]),
        "frozen_rank_bottom50": error_metrics(pp[cc < .5], rr[cc < .5], cc[cc < .5]),
        "mean_frame_mae_mm": {k: mean_frame_metric(k) for k in ["all", "frozen_rank_top50", "frozen_rank_bottom50"]},
        "per_frame": per_frame,
    }


def exclusive_categories(hand, object_mask, table):
    """Discard ambiguous hand/object pixels; exclude both from the table."""
    if hand.shape != object_mask.shape or hand.shape != table.shape:
        raise ValueError("Category mask shapes differ")
    if any(a.dtype != np.bool_ for a in (hand, object_mask, table)):
        raise ValueError("Category masks must be boolean")
    return {"hand": hand & ~object_mask, "object": object_mask & ~hand,
            "table": table & ~hand & ~object_mask}
