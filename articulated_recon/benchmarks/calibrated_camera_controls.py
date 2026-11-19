"""Explicit calibrated-camera controls; never a strict RGB-only/forecast path.

Inputs are numeric NPZ files with verified shared manifests. The source has one
RGB frame. The published rig has K, c2w, resolution_wh and camera_ids; c2w is
already camera-to-world. No pickle, depth, scene fit or target image is loaded.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, load_manifest, validate_manifest


DIAGNOSTICS = {"first_source_frame_published_camera_response", "first_source_frame_camera_cross"}


def checked_cameras(k, c):
    k, c = np.asarray(k, float), np.asarray(c, float)
    if k.shape != (4, 3, 3) or c.shape != (4, 4, 4) or not np.isfinite(k).all() or not np.isfinite(c).all():
        raise ValueError("Expected four finite K[3,3] and c2w[4,4] matrices")
    if np.any(k[:, [0, 1], [0, 1]] <= 0) or not np.allclose(k[:, 2], [0, 0, 1]):
        raise ValueError("Invalid pinhole intrinsics")
    rotation = c[:, :3, :3]
    if (not np.allclose(c[:, 3], [0, 0, 0, 1]) or
            not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5) or
            not np.allclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("Camera transforms must be proper rigid SE3")
    return k, c


def crop_rig(k, c, calibration_wh, image_hw):
    """Match the original loader's resolution scaling and central 2:1 crop."""
    k, c = checked_cameras(k, c)
    size = np.asarray(calibration_wh, float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Calibration resolution_wh must be finite positive [width,height]")
    h, w = image_hw
    if h < 2 or w < 2:
        raise ValueError("Source image is too small")
    ch, cw = min(h, w // 2), min(w, h * 2)
    x, y = (w - cw) // 2, (h - ch) // 2
    k = k.copy(); k[:, 0] *= w / size[0]; k[:, 1] *= h / size[1]
    k[:, 0, 2] -= x; k[:, 1, 2] -= y
    k[:, 0] *= 320 / cw; k[:, 1] *= 160 / ch
    c = np.linalg.inv(c[0])[None] @ c
    if np.min(np.linalg.norm(c[1:, :3, 3], axis=1)) <= 1e-3:
        raise ValueError("Target camera baselines must be nonzero")
    return k, c, [x, y, cw, ch]


def _record(manifest, report, artifact_id, base):
    row = next((a for a in manifest["artifacts"] if a["id"] == artifact_id), None)
    if row is None or artifact_id not in report["method_artifact_ids"] or not row.get("path"):
        raise ValueError("Input must be an active materialized artifact")
    return row, (base / row["path"]).resolve()


def _absolute_artifacts(manifest, base, prefix=""):
    result = copy.deepcopy(manifest["artifacts"])
    for row in result:
        row["id"] = prefix + row["id"]
        row["parent_ids"] = [prefix + x for x in row["parent_ids"]]
        if "path" in row:
            row["path"] = str((base / row["path"]).resolve())
    return result


def prepare(*, input_manifest, source_id, rig_id, timestamps, output):
    """Prepare one source frame plus a supplied four-camera rig, without inference."""
    import cv2
    manifest_path, output = Path(input_manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    report = load_manifest(manifest_path)
    if report["condition"] != "calibrated_rgb" or report["task_mode"] != "reconstruction":
        raise ValueError("Published-camera control requires calibrated_rgb reconstruction lineage")
    source, source_path = _record(manifest, report, source_id, manifest_path.parent)
    rig, rig_path = _record(manifest, report, rig_id, manifest_path.parent)
    if source["kind"] != "rgb" or source["origin"] != "observed_rgb" or source.get("generated") is not False:
        raise ValueError("Source must be one observed RGB frame")
    if rig["kind"] != "cameras" or rig["origin"] != "known_calibration":
        raise ValueError("Rig must explicitly declare known_calibration")
    with np.load(source_path, allow_pickle=False) as data:
        rgb, source_times = data["rgb"].copy(), data["timestamps"].copy()
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or source_times.shape != (1,):
        raise ValueError("Source NPZ must contain exactly one uint8 RGB image and timestamp")
    times = np.asarray(timestamps, float)
    if (times.shape != (17,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or times[0] != source_times[0] or source["time_range_s"] != [float(times[0]), float(times[0])]):
        raise ValueError("Declare 17 increasing output timestamps starting at the only observed timestamp")
    with np.load(rig_path, allow_pickle=False) as data:
        k, c, crop = crop_rig(data["K"], data["c2w"], data["resolution_wh"], rgb.shape[1:3])
        ids = data["camera_ids"].astype(str).tolist()
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("Four unique ordered camera IDs are required; source is first")
    declared_source_camera = source.get("camera_id", source.get("source_camera_id"))
    if declared_source_camera is not None and str(declared_source_camera) != ids[0]:
        raise ValueError("Observed source camera ID differs from rig camera_ids[0]")
    x, y, cw, ch = crop
    pixels = np.zeros((17, 160, 320, 3), np.uint8)
    pixels[0] = cv2.resize(rgb[0, y:y+ch, x:x+cw], (320, 160), interpolation=cv2.INTER_AREA)
    output.mkdir(parents=True)
    np.savez_compressed(output / "condition.npz", rgb=pixels, K=k, c2w=c,
                        timestamps=times, source_frame_indices=np.r_[0, np.full(16, -1)])
    child = copy.deepcopy(manifest)
    child["artifacts"] = _absolute_artifacts(manifest, manifest_path.parent)
    child["artifacts"].append({"id": "published_camera_requests", "kind": "cameras", "origin": "known_calibration",
        "parent_ids": [source_id, rig_id], "path": "condition.npz", "sha256": file_sha256(output / "condition.npz")})
    child.update(method_inputs=["published_camera_requests"], strict_rgb_claim=False, strict_forecasting_claim=False)
    validate_manifest(child, output, verify_files=True)
    (output / "input-manifest.json").write_text(json.dumps(child, indent=2))
    prep = {"mode": "reconstruction", "diagnostic": "first_source_frame_published_camera_response",
        "condition_sha256": file_sha256(output / "condition.npz"), "conditioning_artifact_id": "published_camera_requests",
        "camera_ids": ids, "source_camera_id": ids[0], "source_crop_xywh": crop,
        "source_camera_identity_policy": "verified against source artifact camera ID" if declared_source_camera is not None else "caller declares rig camera_ids[0] is the source; source artifact has no camera-ID metadata",
        "supplied_dataset_cameras": True, "observed_source_frames": 1,
        "camera_design_horizon": "Supplied fixed four-camera rig; no estimated object track",
        "conditioning_horizon": "Only source frame0 pixels; later timestamps are output labels, not observed frames",
        "training_caveat": "Calibrated camera-response diagnostic; no strict RGB-only or forecast claim"}
    (output / "preparation.json").write_text(json.dumps(prep, indent=2))
    return prep


def validate_generation(condition, conditioning, text_guidance, hist_guidance):
    """Guard the explicit generator opt-in before any model computation."""
    condition = Path(condition).resolve()
    prep = json.loads((condition / "preparation.json").read_text())
    report = load_manifest(condition / "input-manifest.json", [("cameras", condition / "condition.npz")])
    if report["condition"] != "calibrated_rgb" or not report["calibration_dependencies"] or report["task_mode"] != "reconstruction":
        raise ValueError("Calibrated diagnostic requires verified known-calibration lineage")
    if prep.get("diagnostic") not in DIAGNOSTICS or prep.get("supplied_dataset_cameras") is not True:
        raise ValueError("Unrecognized calibrated-camera diagnostic preparation")
    if conditioning != "first_frame_diagnostic" or text_guidance != 0 or hist_guidance != 0:
        raise ValueError("Calibrated camera diagnostic requires first-frame conditioning and guidance 0/0")
    if prep.get("mode") != "reconstruction" or file_sha256(condition / "condition.npz") != prep["condition_sha256"]:
        raise ValueError("Calibrated condition changed")
    with np.load(condition / "condition.npz", allow_pickle=False) as data:
        rgb, times = data["rgb"], data["timestamps"]
        checked_cameras(data["K"], data["c2w"])
        if (rgb.shape != (17, 160, 320, 3) or rgb.dtype != np.uint8 or np.any(rgb[1:])
                or times.shape != (17,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError("Expected 17-frame diagnostic with zero future/target conditioning pixels")
        if not np.allclose(data["c2w"][0], np.eye(4), atol=1e-6):
            raise ValueError("Cameras must be source-relative")
        if np.min(np.linalg.norm(data["c2w"][1:, :3, 3], axis=1)) <= 1e-3:
            raise ValueError("Target camera baselines must be nonzero")
    return prep, report


def cross(*, estimated_condition, published_condition, output):
    """Create both declared K/C crosses; keep original ancestry and no future pixels."""
    estimated, published, output = map(lambda p: Path(p).resolve(), [estimated_condition, published_condition, output])
    if output.exists():
        raise FileExistsError(output)
    pub_prep, _ = validate_generation(published, "first_frame_diagnostic", 0, 0)
    est_report = load_manifest(estimated / "input-manifest.json", [("cameras", estimated / "condition.npz")])
    if est_report["condition"] != "rgb_only" or est_report["task_mode"] != "reconstruction":
        raise ValueError("Estimated control must preserve RGB-only reconstruction ancestry")
    est_prep = json.loads((estimated / "preparation.json").read_text())
    if file_sha256(estimated / "condition.npz") != est_prep["condition_sha256"]:
        raise ValueError("Estimated camera control changed")
    with np.load(published / "condition.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in ["rgb", "K", "c2w", "timestamps", "source_frame_indices"]}
    with np.load(estimated / "condition.npz", allow_pickle=False) as z:
        ek, ec = checked_cameras(z["K"], z["c2w"])
        if not np.array_equal(z["rgb"][0], arrays["rgb"][0]) or not np.array_equal(z["timestamps"], arrays["timestamps"]):
            raise ValueError("Crossing requires identical source frame and output timestamps")
        if not np.allclose(ec[0], np.eye(4), atol=1e-6):
            raise ValueError("Estimated cameras must already be source-relative")
    artifacts = []
    for prefix, folder in [("estimated_", estimated), ("published_", published)]:
        artifacts.extend(_absolute_artifacts(json.loads((folder / "input-manifest.json").read_text()), folder, prefix))
    parents = ["estimated_" + est_prep.get("conditioning_artifact_id", "virtual_camera_requests"),
               "published_" + pub_prep["conditioning_artifact_id"]]
    output.mkdir(parents=True)
    results = []
    for name, k, c in [("estimatedK_publishedC", ek, arrays["c2w"]), ("publishedK_virtualC", arrays["K"], ec)]:
        folder = output / name; folder.mkdir()
        np.savez_compressed(folder / "condition.npz", **{**arrays, "K": k, "c2w": c})
        manifest = {"schema_version": 1, "condition": "calibrated_rgb", "task_mode": "reconstruction",
            "strict_rgb_claim": False, "strict_forecasting_claim": False, "method_inputs": ["mixed_camera_requests"],
            "artifacts": copy.deepcopy(artifacts) + [{"id": "mixed_camera_requests", "kind": "cameras", "origin": "model_prediction",
                "parent_ids": parents, "path": "condition.npz", "sha256": file_sha256(folder / "condition.npz"),
                "component_policy": name}],
            "metric_scale": {"status": "unresolved", "artifact_ids": [], "description": "Mixed conditioning requests, not a reconstructed scene scale"},
            "robot_registration": {"status": "unresolved", "artifact_ids": [], "description": "No robot registration in generation"}}
        validate_manifest(manifest, folder, verify_files=True)
        (folder / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
        prep = {**pub_prep, "diagnostic": "first_source_frame_camera_cross", "conditioning_artifact_id": "mixed_camera_requests",
            "condition_sha256": file_sha256(folder / "condition.npz"), "camera_design": name,
            "camera_design_horizon": name + "; original estimated component retains full-source camera-design ancestry; only source frame0 pixels condition generation",
            "training_caveat": "Calibrated K/C crossover, not strict RGB-only or forecast; estimated-camera ancestry is retained"}
        (folder / "preparation.json").write_text(json.dumps(prep, indent=2))
        validate_generation(folder, "first_frame_diagnostic", 0, 0)
        results.append({"condition": name, "path": str(folder), "condition_sha256": prep["condition_sha256"]})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    for name in ["input-manifest", "source-id", "rig-id", "timestamps-json", "output"]:
        a.add_argument("--" + name, required=True)
    a = sub.add_parser("cross")
    for name in ["estimated-condition", "published-condition", "output"]:
        a.add_argument("--" + name, required=True)
    args = vars(p.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["timestamps"] = json.loads(Path(args.pop("timestamps_json")).read_text())
    print(json.dumps({"prepare": prepare, "cross": cross}[command](**args), indent=2))


if __name__ == "__main__":
    main()
