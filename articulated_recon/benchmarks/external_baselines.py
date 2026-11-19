"""RGB-only launch adapters for pinned SceneRig and OVOW releases.

The default CLI only reports a launch plan. --execute runs a separately installed
baseline; this module never installs weights, passes oracle inputs, or plans robot
actions. Artifact completeness is deliberately distinct from geometric accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import ProtocolError, validate_manifest

RELEASES = {
    "scenerig": {
        "repository": "https://github.com/Red-Fairy/SceneRig",
        "commit": "2defb4b112d5abce28ab1f2bea8aac014b31533e",
        "entry": "lib/runners/static_scene.py",
        "entry_sha256": "093fb266a2eb6b31c8435eced476ff8f5b1e9c6e9b38f2fd3f1c51e96fc3d8dc",
        "input_scope": "first_frame",
        "installation_gb": 150,
    },
    "ovow": {
        "repository": "https://github.com/SparcAI-Inc/OVOW",
        "commit": "0078fa2bab641f6dd334b80381fced05a273bb10",
        "entry": "scripts/run_video.sh",
        "entry_sha256": "1a29e42fab4fc4ba8fcdcc80706e429455d4b728d82217f0a140f67ec48933b8",
        "input_scope": "full_clip",
        "installation_gb": 120,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value, base):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def build_command(request, base_dir):
    """Validate input provenance and return an argv list; no shell interpolation."""
    if request.get("schema_version") != 1:
        raise ValueError("baseline request schema_version must be 1")
    baseline = request.get("baseline")
    if baseline not in RELEASES:
        raise ValueError("baseline must be scenerig or ovow")
    allowed = {"schema_version", "baseline", "checkout", "input_manifest", "rgb_artifact_id",
               "input_scope", "output_dir", "python", "blender", "model", "gpu",
               "timeout_seconds", "minimum_free_gb", "required_paths", "required_environment",
               "allow_localized_checkout"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"Unsupported request fields (no passthrough arguments): {sorted(unknown)}")
    release = RELEASES[baseline]
    if request.get("input_scope") != release["input_scope"]:
        raise ValueError(f"{baseline} requires input_scope={release['input_scope']}")
    base_dir = Path(base_dir).resolve()
    manifest_path = _path(request["input_manifest"], base_dir)
    manifest = json.loads(manifest_path.read_text())
    report = validate_manifest(manifest, base_dir=manifest_path.parent, verify_files=True)
    if manifest["condition"] != "rgb_only":
        raise ProtocolError("External RGB-only baseline requires condition=rgb_only")
    source_id = request["rgb_artifact_id"]
    sources = [item for item in manifest["artifacts"] if item["id"] == source_id]
    if len(sources) != 1:
        raise ProtocolError("rgb_artifact_id must identify exactly one artifact")
    source = sources[0]
    if source["kind"] != "rgb" or source["origin"] != "observed_rgb" or source.get("generated") is not False:
        raise ProtocolError("External baseline requires observed RGB, not generated views or geometry")
    if source_id not in manifest["method_inputs"]:
        raise ProtocolError("Selected RGB must be a declared method input")
    source_path = _path(source["path"], manifest_path.parent)
    if sha256(source_path) != source["sha256"]:
        raise ProtocolError("RGB digest mismatch")
    suffixes = {"scenerig": {".png", ".jpg", ".jpeg"}, "ovow": {".mp4", ".mov", ".mkv"}}
    if source_path.suffix.lower() not in suffixes[baseline]:
        raise ValueError("Supply the first RGB frame for SceneRig or the source RGB video for OVOW")
    bounds = source.get("time_range_s")
    if baseline == "scenerig" and (not bounds or bounds[0] != bounds[1]):
        raise ProtocolError("SceneRig first-frame input must declare one source timestamp")
    if baseline == "ovow" and (not bounds or bounds[1] <= bounds[0]):
        raise ProtocolError("OVOW video input must declare a nonzero observed time span")
    checkout = _path(request["checkout"], base_dir)
    output = _path(request["output_dir"], base_dir)
    if output == checkout or checkout in output.parents:
        raise ValueError("Keep benchmark output outside the external source checkout")
    gpu = str(request.get("gpu", "0"))
    if not re.fullmatch(r"\d+", gpu):
        raise ValueError("gpu must be one nonnegative device index")
    if baseline == "scenerig":
        python = _path(request.get("python", str(checkout / ".venv/bin/python")), base_dir)
        blender = _path(request.get("blender", str(checkout / "lib/utils/third_party/blender-4.5/blender")), base_dir)
        # Bypass run_e2e.sh, which evaluates key exports from the user's shell rc.
        # These are the only data-bearing arguments; no depth/camera option exists.
        command = [str(python), str(checkout / release["entry"]), "--image", str(source_path),
                   "--output-dir", str(output / "native"), "--gpu", gpu,
                   "--model", request.get("model", "claude-opus-5"), "--blender-command", str(blender)]
    else:
        command = ["bash", str(checkout / release["entry"]), "--name", "source",
                   "--video", str(source_path), "--out", str(output / "native")]
    return {"baseline": baseline, "release": release, "checkout": str(checkout),
            "output_dir": str(output), "command": command,
            "source": {"artifact_id": source_id, "path": str(source_path), "sha256": source["sha256"],
                       "input_scope": request["input_scope"], "time_range_s": bounds},
            "input_protocol_report": report, "input_manifest_sha256": sha256(manifest_path),
            "gpu": gpu}


def preflight(plan, request):
    """Read-only static/runtime checks. Model import/inference viability is untested."""
    checkout = Path(plan["checkout"])
    errors, warnings = [], []
    release = plan["release"]
    if not (checkout / release["entry"]).is_file():
        errors.append("missing external runner")
    elif sha256(checkout / release["entry"]) != release["entry_sha256"]:
        errors.append("external runner differs from the verified pinned entry point")
    commit, diff_digest = None, None
    try:
        commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        if commit != release["commit"]:
            errors.append("external checkout differs from pinned release")
        diff = subprocess.check_output(["git", "-C", str(checkout), "diff", "HEAD", "--"], stderr=subprocess.DEVNULL)
        diff_digest = hashlib.sha256(diff).hexdigest()
        if diff and not request.get("allow_localized_checkout", False):
            errors.append("tracked external edits require allow_localized_checkout=true and a reviewed diff")
    except (OSError, subprocess.CalledProcessError):
        errors.append("external checkout is not a readable Git repository")
    if platform.system() != "Linux":
        errors.append("released baseline environments require Linux")
    executable = plan["command"][0]
    if not (shutil.which(executable) or (Path(executable).is_file() and os.access(executable, os.X_OK))):
        errors.append("baseline interpreter/shell not executable")
    if plan["baseline"] == "scenerig":
        blender = Path(plan["command"][-1])
        if not blender.is_file() or not os.access(blender, os.X_OK):
            errors.append("SceneRig Blender binary is missing or not executable")
        if not any(os.environ.get(key) for key in ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")):
            errors.append("SceneRig agent API credential is absent from environment")
    else:
        if not (checkout / ".env").is_file():
            errors.append("OVOW localized .env configuration is missing")
        warnings.append("OVOW sources its local .env; review model/runtime configuration before launch")
    for value in request.get("required_paths", []):
        if not _path(value, checkout).exists():
            errors.append(f"missing declared runtime dependency: {value}")
    for name in request.get("required_environment", []):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError("required_environment must contain variable names, never values")
        if not os.environ.get(name):
            errors.append(f"missing required environment variable: {name}")
    if not request.get("required_paths"):
        errors.append("declare required_paths for installed model weights/runtime before execution")
    output = Path(plan["output_dir"])
    if output.exists():
        errors.append("output directory already exists; use a fresh run directory")
    parent = output.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    available_gb = shutil.disk_usage(parent).free / 1e9
    minimum_gb = float(request.get("minimum_free_gb", 20))
    if minimum_gb < 1:
        raise ValueError("minimum_free_gb must be >=1")
    if available_gb < minimum_gb:
        errors.append("insufficient free disk for requested run reserve")
    gpu_info = None
    if shutil.which("nvidia-smi"):
        probe = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if probe.returncode == 0:
            rows = [line.split(",") for line in probe.stdout.strip().splitlines()]
            rows = [[part.strip() for part in row] for row in rows]
            selected = [row for row in rows if row[0] == plan["gpu"]]
            if selected:
                gpu_info = selected[0]
                if plan["baseline"] == "ovow" and float(gpu_info[2]) < 40000:
                    warnings.append("OVOW recommends at least 40 GB VRAM")
            else:
                errors.append("requested GPU index unavailable")
        else:
            errors.append("nvidia-smi could not query GPU")
    else:
        errors.append("NVIDIA GPU runtime not detected")
    return {"ready_to_launch": not errors, "errors": errors, "warnings": warnings,
            "checkout_commit": commit, "tracked_diff_sha256": diff_digest,
            "gpu": gpu_info, "free_disk_gb": available_gb,
            "dependency_validation": "static presence only; upstream imports and model inference not yet tested"}


def inspect_outputs(baseline, native_root):
    """Inspect native artifact inventory, never label physical execution successful."""
    native_root = Path(native_root)
    if baseline == "scenerig":
        scene = native_root / "scene"
        required = [scene / name for name in ("preprocess_manifest.json", "blender_file.blend", "final/pipeline_result.json")]
        mesh_files = sorted(p for p in (scene / "meshes").rglob("*") if p.is_file() and p.suffix in {".ply", ".obj", ".glb"}) if (scene / "meshes").exists() else []
        errors = [] if mesh_files else ["no object meshes"]
    elif baseline == "ovow":
        scenes = sorted(native_root.glob("ws/source_*"))
        if len(scenes) != 1:
            return {"artifact_complete": False, "errors": ["expected exactly one new OVOW source workspace"], "files": []}
        scene = scenes[0]
        names = ("scene_anim_blender_physics_with_deform.glb",
                 "fullscene_deform_objseq_white_left_rgb_right_split10_h264.mp4",
                 "fullscene_deform_objseq_textured_left_rgb_right_split10_h264.mp4")
        required = []
        errors = []
        for name in names:
            matches = list(scene.rglob(name))
            if len(matches) != 1:
                errors.append(f"expected one {name}")
            else:
                required.append(matches[0])
        mesh_files = []
    else:
        raise ValueError("Unknown baseline")
    files = []
    for path in required + mesh_files:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing/empty artifact: {path.name}")
            continue
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, dict) or not value:
                    errors.append(f"invalid result object: {path.name}")
                if value.get("success") is False or value.get("status") in {"failed", "error"}:
                    errors.append(f"native failure status: {path.name}")
            except (OSError, ValueError):
                errors.append(f"invalid JSON: {path.name}")
        files.append({"path": str(path.relative_to(native_root)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"artifact_complete": not errors, "errors": errors, "files": files,
            "geometry_validated": False, "robot_plan_executed": False,
            "completion_scope": "native reconstruction artifacts only"}


def execute(plan, request, checks):
    if not checks["ready_to_launch"]:
        raise RuntimeError("Baseline preflight failed; no inference launched")
    if sha256(plan["source"]["path"]) != plan["source"]["sha256"]:
        raise ProtocolError("RGB changed after launch preparation; no inference launched")
    timeout = int(request.get("timeout_seconds", 1800))
    if timeout <= 0:
        raise ValueError("timeout_seconds must be positive")
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "status": "running", "plan": plan, "preflight": checks,
              "simulator_feedback_to_robot_planner": False, "robot_plan_executed": False}
    receipt = output / "baseline-receipt.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = plan["gpu"]
    if plan["baseline"] == "ovow":
        environment["OVOW_GPU"] = "0"
    started = time.monotonic()
    with (output / "process.log").open("w") as log:
        try:
            process = subprocess.Popen(plan["command"], cwd=plan["checkout"], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Stop only the process group created for this run, including
                # model subprocesses. A timeout must not leave GPU jobs running.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            record["returncode"] = returncode
            artifacts = inspect_outputs(plan["baseline"], output / "native")
            record["artifacts"] = artifacts
            record["status"] = "complete" if returncode == 0 and artifacts["artifact_complete"] else "failed"
            if sha256(plan["source"]["path"]) != plan["source"]["sha256"]:
                record.update(status="failed", error="source_rgb_changed_during_execution")
        except (OSError, subprocess.TimeoutExpired) as error:
            record.update(status="failed", error=type(error).__name__, returncode=None)
        finally:
            record["wall_seconds"] = time.monotonic() - started
            receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional preflight JSON, outside the future run directory")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    plan = build_command(request, args.request.resolve().parent)
    checks = preflight(plan, request)
    result = execute(plan, request, checks) if args.execute else {"status": "not_run", "plan": plan, "preflight": checks}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"not_run", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
