# SPDX-License-Identifier: Apache-2.0
"""Summarise matched unseen-view evaluations (scripts/rh20t/eval_unseen_matched.py).

    python scripts/rh20t/summarize_unseen_eval.py rgb_10000=<dir> pc_10000=<dir> pc_43000=<dir> \
        [--json out.json]

Each <dir> holds shard*.npz of one checkpoint. Arms must cover the same clips with identical
inputs (checked). Each clip is reduced to one number per metric (mean over its selected
images), so confidence intervals come from a paired bootstrap over clips against the first
(reference) arm. Selections: ``tgt`` = unseen views, every frame; ``ctx`` = observed views,
frames 1.. (frame 0 is the given history).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

METRICS = ("psnr", "ssim", "psnr_covered", "psnr_uncovered")
N_BOOT = 10000


def load_arm(path: Path) -> dict[str, np.ndarray]:
    shards = sorted(path.glob("shard*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard*.npz under {path}")
    parts = [dict(np.load(s)) for s in shards]
    keys = ("index", "episode", "context", "gt_mean", "pred", "copy0", "render")
    arm = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = np.argsort(arm["index"])
    arm = {k: v[order] for k, v in arm.items()}
    arm["step"] = int(parts[0]["step"])
    return arm


def selections(context: np.ndarray, n_frames: int) -> dict[str, np.ndarray]:
    """[N, V] context -> {name: [N, V, T] bool}."""
    tgt = np.repeat(~context[:, :, None], n_frames, axis=2)
    ctx = np.repeat(context[:, :, None], n_frames, axis=2)
    ctx[:, :, 0] = False
    return {"tgt": tgt, "ctx": ctx}


def per_clip(values: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """values [N, V, T], sel [N, V, T] -> [N] nan-mean over the selected images."""
    v = np.where(sel, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (e.g. nothing covered)
        return np.nanmean(v.reshape(len(v), -1), axis=1)


def paired_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    d = diff[~np.isnan(diff)]
    boot = d[rng.integers(len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def check_matched(arms: dict[str, dict[str, np.ndarray]]) -> None:
    ref_name, ref = next(iter(arms.items()))
    for name, arm in arms.items():
        for key in ("index", "episode", "context"):
            if not np.array_equal(arm[key], ref[key]):
                raise ValueError(f"{name} and {ref_name} differ in {key}: not a matched evaluation")
        if not np.allclose(arm["gt_mean"], ref["gt_mean"], atol=1e-4):
            raise ValueError(f"{name} and {ref_name} saw different ground-truth clips")


def summarize(arms: dict[str, dict[str, np.ndarray]]) -> dict:
    check_matched(arms)
    ref_name, ref = next(iter(arms.items()))
    sels = selections(ref["context"].astype(bool), ref["pred"].shape[2])
    rng = np.random.default_rng(0)
    rows: dict = {"n_clips": int(len(ref["index"])), "reference": ref_name, "arms": {}, "baselines": {}}
    for name, arm in arms.items():
        rows["arms"][name] = {"step": arm["step"]}
        for sname, sel in sels.items():
            for m, metric in enumerate(METRICS):
                clip = per_clip(arm["pred"][..., m], sel)
                entry = {"mean": float(np.nanmean(clip))}
                if name != ref_name:
                    entry["delta_vs_ref"] = paired_ci(clip - per_clip(ref["pred"][..., m], sel), rng)
                rows["arms"][name][f"{sname}/{metric}"] = entry
            rows["arms"][name][f"{sname}/psnr_by_frame"] = [
                float(np.nanmean(np.where(sel[:, :, t], arm["pred"][:, :, t, 0], np.nan)))
                if sel[:, :, t].any() else None for t in range(sel.shape[2])
            ]
    for base in ("copy0", "render"):
        sname = "ctx" if base == "copy0" else "tgt"
        sel = sels[sname]
        rows["baselines"][base] = {
            f"{sname}/{metric}": float(np.nanmean(per_clip(ref[base][..., m], sel)))
            for m, metric in enumerate(METRICS)
        }
        rows["baselines"][base][f"{sname}/psnr_by_frame"] = [
            float(np.nanmean(np.where(sel[:, :, t], ref[base][:, :, t, 0], np.nan))) if sel[:, :, t].any() else None
            for t in range(sel.shape[2])
        ]
    return rows


def print_markdown(rows: dict) -> None:
    print(f"{rows['n_clips']} matched val clips; deltas are paired vs {rows['reference']} (95% bootstrap CI)\n")
    cols = [f"{s}/{m}" for s in ("tgt", "ctx") for m in METRICS]
    print("| arm | step | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 2) + "|")
    for name, arm in rows["arms"].items():
        cells = []
        for c in cols:
            e = arm[c]
            cell = f"{e['mean']:.3f}"
            if "delta_vs_ref" in e:
                d, lo, hi = e["delta_vs_ref"]
                cell += f" ({d:+.3f} [{lo:+.3f}, {hi:+.3f}])"
            cells.append(cell)
        print(f"| {name} | {arm['step']} | " + " | ".join(cells) + " |")
    for base, vals in rows["baselines"].items():
        print(f"| baseline {base} | – | " + " | ".join(f"{vals[c]:.3f}" if c in vals else "–" for c in cols) + " |")
    print("\nPSNR by frame (tgt = unseen views, ctx = observed views; frame 0 of ctx is given)")
    series = {f"{n} {s}": a[f"{s}/psnr_by_frame"] for n, a in rows["arms"].items() for s in ("tgt", "ctx")}
    series.update({f"{b} {s}": v[f"{s}/psnr_by_frame"] for b, v in rows["baselines"].items()
                   for s in ("tgt", "ctx") if f"{s}/psnr_by_frame" in v})
    n_frames = len(next(iter(series.values())))
    print("| series | " + " | ".join(str(t) for t in range(n_frames)) + " |")
    print("|---" * (n_frames + 1) + "|")
    for name, vals in series.items():
        print(f"| {name} | " + " | ".join("–" if v is None else f"{v:.2f}" for v in vals) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="name=dir; the first is the reference arm")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    arms = {}
    for spec in args.arms:
        name, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"expected name=dir, got {spec!r}")
        arms[name] = load_arm(Path(path))
    rows = summarize(arms)
    print_markdown(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
