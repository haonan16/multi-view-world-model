"""Renderer-only migration of the two never-executed November 13 block plans.

This is a legacy calibrated control, not the new RGB-only method. Old seals and
plans remain untouched. The executor loses rendering only; actions, integration
frequency, controller, initial state, fixture and evaluation policy are retained.
"""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

EXECUTOR_SHA = "2a908b2aff07ed48840e8bc0ad7ddda2178c0ec0430a8e35a45d3286bc922abd"
PROBE_SHA = "4e36a53189a2d6d1f605f885c88405779eb75925fd9799b2f4aae58a877eb036"
VARIANTS = ("stationary_minimal_rotation", "stable_support")
RENDER_CHANGES = (
    ("color=sapien.render.RenderMaterial(base_color=[.84,.56,.2,1],roughness=.6)", "color=None"),
    ("(cap,'cap',sapien.render.RenderMaterial(base_color=[.2,.65,.3,1]))", "(cap,'cap',None)"),
    ("b.add_visual_from_file(str(self.case_dir/(name+'.obj')),material=col);", ""),
    ("   b.add_visual_from_file(str(self.case_dir/'object.obj'),material=color)\n", ""),
    ("render_mode='rgb_array',sim_backend='cpu',human_render_camera_configs={'shader_pack':'minimal'},",
     "render_mode=None,sim_backend='cpu',render_backend='none',"),
    (";writer=imageio.get_writer(out/'physics.mp4',fps=freq,codec='libx264',quality=8,macro_block_size=16)", ""),
    ("records.append(row);frame=arr(e.render_rgb_array())[0].astype(np.uint8);writer.append_data(frame)\n   if len(records)%40==0:imageio.imwrite(out/f'frame-{len(records):03d}.png',frame)",
     "records.append(row)"),
    ("writer.close();writer=None;np.save(out/'actions.npy'", "np.save(out/'actions.npy'"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def render_only_executor(source):
    """Accept exactly the sealed historical source and apply visual-only edits."""
    if hashlib.sha256(source.encode()).hexdigest() != EXECUTOR_SHA:
        raise ValueError("Unexpected predecessor executor; refusing a heuristic migration")
    for old, new in RENDER_CHANGES:
        if source.count(old) != 1:
            raise ValueError("Renderer migration did not match exactly once")
        source = source.replace(old, new)
    compile(source, "execute_headless.py", "exec")
    if "RenderMaterial(" in source or "add_visual_from_file(" in source:
        raise ValueError("Headless executor still creates visual resources")
    return source


def generic_collision_probe(source):
    """Add one synthetic collision-only box; no task geometry or commands are read."""
    if hashlib.sha256(source.encode()).hexdigest() != PROBE_SHA:
        raise ValueError("Unexpected generic probe source")
    edits = (
        ("manipulated_objects=0, task_data_read=False,", "manipulated_objects=1, task_objects=0, generic_dummy_objects=1, task_data_read=False,"),
        ("    import sapien\n", "    import sapien\n    import trimesh\n"),
        ("            self.table_scene.build()\n", """            self.table_scene.build()
            mesh = trimesh.creation.box(extents=[.04, .03, .02])
            builder = self.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose(p=[.30, .20, .01])
            material = sapien.physx.PhysxMaterial(1., 1., 0.)
            builder.add_box_collision(pose=sapien.Pose(mesh.bounds.mean(0)), half_size=mesh.extents/2,
                                      material=material, density=500)
            self.dummy = builder.build(name="generic_dummy_box")
"""),
        ("    report[\"sensor_count\"] = len(env._sensors)\n", "    assert \"generic_dummy_box\" in report[\"scene_actor_names\"]\n    report[\"generic_dummy_collision_only\"] = True\n    report[\"sensor_count\"] = len(env._sensors)\n"),
    )
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("Generic collision probe edit is ambiguous")
        source = source.replace(old, new)
    compile(source, "generic_headless_probe.py", "exec")
    return source


def verify_bundle(folder, executor):
    frozen = read(folder / "frozen-plan.json")
    if sha(executor) != frozen["executor_sha256"] or frozen["friction"] != 1.:
        raise ValueError("Executor or physics configuration changed")
    for name, expected in frozen["files"].items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise ValueError(f"Sealed file changed: {name}")
    if (folder / "execution-started.json").exists():
        raise ValueError("A measured execution was already claimed")
    return frozen


def prepare(source_root, output, failed_infrastructure=None):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    original_executor = source_root / "code/planning/execute_fixed_reference.py"
    source = original_executor.read_text()
    headless = render_only_executor(source)
    probe = source_root / "cluster/headless-probe/generic_headless_probe.py"
    if sha(probe) != PROBE_SHA:
        raise ValueError("Generic probe source changed")
    collision_probe = generic_collision_probe(probe.read_text())
    infrastructure_failures = []
    if failed_infrastructure is not None:
        failed_infrastructure = Path(failed_infrastructure).resolve()
        for variant in VARIANTS:
            previous_output = failed_infrastructure / "physics" / variant / "block"
            for name in ("records.json", "partial-records.json", "actions.npy", "summary.json"):
                if (previous_output / name).exists():
                    raise ValueError("Previous attempt may contain task outcomes; do not migrate as infrastructure-only")
            log = failed_infrastructure / "physics" / (variant + ".log")
            text = log.read_text()
            if "self._load_scene(options)" not in text or "color=sapien.render.RenderMaterial(" not in text or "RuntimeError: failed to find a rendering device" not in text:
                raise ValueError("Previous attempt is not the known pre-task visual-material construction failure")
            receipt = failed_infrastructure / "inputs" / variant / "block/execution-started.json"
            infrastructure_failures.append({"variant": variant, "log": str(log), "log_sha256": sha(log),
                "receipt": str(receipt), "receipt_sha256": sha(receipt), "task_control_steps": 0,
                "evidence": "Constructor failed in _load_scene before object construction/env.step; no records/actions/summary exist"})
    paired_index = read(source_root / "audit/paired-pre-execution-index.json")
    predecessors = []
    for variant in VARIANTS:
        folder = source_root / "planning/fixed-reference-inputs" / (variant + "__se3") / "block"
        frozen = verify_bundle(folder, original_executor)
        entry = next(x for x in paired_index["variants"] if x["variant"] == variant)
        if sha(folder / "frozen-plan.json") != entry["frozen_plan_sha256"] or sha(entry["seal"]) != entry["seal_sha256"]:
            raise ValueError("Pre-execution audit seal changed")
        check = read(folder / "fixed-reference-check.json")
        if not all(check[k] for k in ("passed", "commands_exactly_preserved", "reference_attached_after_command_seal",
                                      "initial_pose_exact", "geometry_unchanged", "no_reference_recenter_resize_or_support_snap")):
            raise ValueError("Independent fixture packaging contract failed")
        predecessors.append((variant, folder, frozen, entry))

    output.mkdir(parents=True)
    (output / "code").mkdir()
    (output / "probe").mkdir()
    executor = output / "code/execute_headless.py"
    executor.write_text(headless)
    shutil.copyfile(original_executor, output / "code/predecessor_executor.py")
    shutil.copyfile(source_root / "code/planning/freeze_plan.py", output / "code/freeze_plan.py")
    shutil.copyfile(Path(__file__), output / "code/legacy_headless.py")
    (output / "probe/generic_headless_probe.py").write_text(collision_probe)
    write(output / "prior-infrastructure-failures.json", infrastructure_failures)
    (output / "code/renderer-only.diff").write_text("".join(difflib.unified_diff(
        source.splitlines(True), headless.splitlines(True), fromfile="predecessor_executor.py", tofile="execute_headless.py")))
    entries = []
    for variant, original, frozen, previous in predecessors:
        folder = output / "inputs" / variant / "block"
        folder.mkdir(parents=True)
        for name in frozen["files"]:
            shutil.copyfile(original / name, folder / name)
        shutil.copyfile(original / "frozen-plan.json", folder / "predecessor-frozen-plan.json")
        migration = {
            "schema": "renderer-only-legacy-migration-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "predecessor_bundle": str(original), "predecessor_frozen_sha256": sha(original / "frozen-plan.json"),
            "predecessor_pre_execution_seal": previous["seal"], "predecessor_pre_execution_sha256": previous["seal_sha256"],
            "predecessor_task_execution_receipt_exists": False,
            "supersedes_infrastructure_jobs": {"renderer_failed": "60338367", "task_cancelled_before_start": "60352240"},
            "preserved_input_hashes": frozen["files"], "predecessor_executor_sha256": EXECUTOR_SHA,
            "headless_executor_sha256": sha(executor), "migration_code_sha256": sha(Path(__file__)),
            "changed": ["render_backend=none", "render_mode=None", "disable image/video output", "omit visual materials and visual mesh builders"],
            "prior_infrastructure_failures": infrastructure_failures,
            "commands_fixture_controller_physics_unchanged": True,
            "classification": "Legacy calibrated control; independent fixed recorded-RGBD reconstruction proxy, not exact GT or strict RGB-only",
            "video_policy": "Any later video must be marked offline replay of measured states, not a second physics run",
        }
        write(folder / "infrastructure-migration.json", migration)
        replacement = dict(frozen, created_utc=datetime.now(timezone.utc).isoformat(), executor_sha256=sha(executor))
        replacement["files"] = dict(frozen["files"], **{
            "predecessor-frozen-plan.json": sha(folder / "predecessor-frozen-plan.json"),
            "infrastructure-migration.json": sha(folder / "infrastructure-migration.json"),
        })
        write(folder / "frozen-plan.json", replacement)
        verify_bundle(folder, executor)
        entries.append({"variant": variant, "folder": str(folder), "frozen_plan_sha256": sha(folder / "frozen-plan.json")})
    index = {
        "schema": "legacy-headless-paired-execution-v1", "source_root": str(source_root), "variants": entries,
        "executor": str(executor), "executor_sha256": sha(executor), "probe_sha256": sha(output / "probe/generic_headless_probe.py"),
        "runtime_env": str(source_root / "cluster/runtime.env"), "runtime_env_sha256": sha(source_root / "cluster/runtime.env"),
        "freeze_helper_sha256": sha(output / "code/freeze_plan.py"),
        "migration_code_sha256": sha(output / "code/legacy_headless.py"),
        "task_rollouts_per_variant": 1, "automatic_retries": 0,
        "task_execution_started": False, "controller_or_physics_changed": False,
    }
    write(output / "migration-index.json", index)
    return index


def run(root, preflight):
    """Run both unchanged plans once, only after a successful generic CPU probe."""
    root = Path(root).resolve()
    index = read(root / "migration-index.json")
    probe = read(preflight)
    if not (probe["status"] == "passed" and probe["task_data_read"] is False and probe["task_objects"] == 0
            and probe["generic_dummy_objects"] == 1 and probe["generic_dummy_collision_only"] is True
            and probe["script_sha256"] == index["probe_sha256"] and probe["render_backend"] == "none"
            and probe["sim_backend"] == "physx_cpu" and probe["control_steps"] == 10):
        raise ValueError("Required generic no-task headless probe did not pass")
    if sha(index["runtime_env"]) != index["runtime_env_sha256"] or probe["runtime_env_sha256"] != index["runtime_env_sha256"]:
        raise ValueError("Runtime environment changed since preflight")
    if {k: importlib.metadata.version(k) for k in probe["versions"]} != probe["versions"]:
        raise ValueError("Runtime package versions changed since probe")
    if sha(index["executor"]) != index["executor_sha256"] or sha(root / "code/freeze_plan.py") != index["freeze_helper_sha256"]:
        raise ValueError("Sealed executor/helper changed")
    if sha(Path(__file__)) != index["migration_code_sha256"]:
        raise ValueError("Batch runner changed")
    for item in index["variants"]:
        folder = Path(item["folder"])
        if sha(folder / "frozen-plan.json") != item["frozen_plan_sha256"]:
            raise ValueError("Migrated plan seal changed")
        verify_bundle(folder, index["executor"])
    output = root / "physics"
    output.mkdir(exist_ok=False)
    write(output / "batch-seal.json", {"preflight": str(preflight), "preflight_sha256": sha(preflight),
                                       "migration_index_sha256": sha(root / "migration-index.json"),
                                       "automatic_retries": 0, "all_variants_sealed_before_execution": True})
    results = []
    for item in index["variants"]:
        folder = Path(item["folder"])
        variant_output = output / item["variant"]
        log = output / (item["variant"] + ".log")
        try:
            with log.open("x") as stream:
                process = subprocess.run([sys.executable, index["executor"], "--inputs", str(folder.parent),
                    "--output", str(variant_output), "--cases", "block", "--friction", "1.", "--require-frozen"],
                    stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
            summary_path = variant_output / "block/summary.json"
            summary = read(summary_path) if summary_path.exists() else None
            complete = process.returncode == 0 and summary is not None and summary.get("status") == "complete"
            result = dict(variant=item["variant"], status="complete" if complete else "failed", summary=summary,
                          process_returncode=process.returncode, log=str(log), retries=0)
        except subprocess.TimeoutExpired:
            result = dict(variant=item["variant"], status="timeout", log=str(log), retries=0)
        results.append(result)
        # Each incremental result is a new file; completed evidence is never replaced.
        write(output / (item["variant"] + "-result.json"), result)
        print(json.dumps(result), flush=True)
    write(output / "results.json", results)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--failed-infrastructure", type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("--root", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    args = vars(p.parse_args())
    command = args.pop("command")
    result = (prepare if command == "prepare" else run)(**args)
    print(json.dumps(result, indent=2))
    if command == "run" and any(x["status"] != "complete" for x in result):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
