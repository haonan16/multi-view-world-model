import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "benchmarks/external_baselines.py"
    spec = importlib.util.spec_from_file_location("external_baselines", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_request(tmp_path, baseline="scenerig"):
    adapter = module()
    source = tmp_path / ("source.png" if baseline == "scenerig" else "source.mp4")
    source.write_bytes(b"fixed RGB evidence for argument/provenance test")
    bounds = [0, 0] if baseline == "scenerig" else [0, 5]
    protocol = {"schema_version": 1, "condition": "rgb_only", "task_mode": "reconstruction",
                "method_inputs": ["rgb"], "artifacts": [{"id": "rgb", "kind": "rgb", "origin": "observed_rgb",
                "generated": False, "parent_ids": [], "time_range_s": bounds, "path": source.name,
                "sha256": adapter.sha256(source)}]}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    request = {"schema_version": 1, "baseline": baseline, "checkout": "checkout",
               "input_manifest": "protocol.json", "rgb_artifact_id": "rgb",
               "input_scope": "first_frame" if baseline == "scenerig" else "full_clip", "output_dir": "run"}
    return adapter, request, protocol


@pytest.mark.parametrize("baseline", ["scenerig", "ovow"])
def test_external_command_uses_only_verified_rgb(tmp_path, baseline):
    adapter, request, _ = fixture_request(tmp_path, baseline)
    plan = adapter.build_command(request, tmp_path)
    assert plan["source"]["artifact_id"] == "rgb"
    assert "--depth" not in plan["command"] and "--camera" not in plan["command"]
    assert "--image" in plan["command"] if baseline == "scenerig" else "--video" in plan["command"]
    assert not Path(plan["output_dir"]).exists()


def test_external_cannot_inject_oracle_flags(tmp_path):
    adapter, request, _ = fixture_request(tmp_path)
    request["extra_args"] = ["--depth", "oracle.npy"]
    with pytest.raises(ValueError, match="Unsupported request fields"):
        adapter.build_command(request, tmp_path)


def test_external_refuses_changed_rgb_or_generated_evidence(tmp_path):
    adapter, request, protocol = fixture_request(tmp_path)
    (tmp_path / "source.png").write_bytes(b"different input")
    with pytest.raises(ValueError):
        adapter.build_command(request, tmp_path)
    adapter, request, protocol = fixture_request(tmp_path)
    protocol["artifacts"][0]["generated"] = True
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        adapter.build_command(request, tmp_path)


def test_scenerig_zero_exit_artifacts_are_not_enough_without_meshes(tmp_path):
    adapter = module()
    scene = tmp_path / "scene"
    (scene / "final").mkdir(parents=True)
    (scene / "preprocess_manifest.json").write_text('{"schema_version":1}')
    (scene / "final/pipeline_result.json").write_text('{"status":"complete"}')
    (scene / "blender_file.blend").write_bytes(b"native scene")
    assert not adapter.inspect_outputs("scenerig", tmp_path)["artifact_complete"]
    (scene / "meshes").mkdir()
    (scene / "meshes/object.ply").write_bytes(b"ply\n")
    report = adapter.inspect_outputs("scenerig", tmp_path)
    assert report["artifact_complete"]
    assert not report["geometry_validated"] and not report["robot_plan_executed"]


def test_ovow_requires_all_outputs_from_one_workspace(tmp_path):
    adapter = module()
    scene = tmp_path / "ws/source_123"
    scene.mkdir(parents=True)
    (scene / "scene_anim_blender_physics_with_deform.glb").write_bytes(b"glTF")
    assert not adapter.inspect_outputs("ovow", tmp_path)["artifact_complete"]
    for kind in ("white", "textured"):
        (scene / f"fullscene_deform_objseq_{kind}_left_rgb_right_split10_h264.mp4").write_bytes(b"video")
    assert adapter.inspect_outputs("ovow", tmp_path)["artifact_complete"]
    (tmp_path / "ws/source_456").mkdir()
    assert not adapter.inspect_outputs("ovow", tmp_path)["artifact_complete"]


def test_execute_does_not_run_when_preflight_fails(tmp_path):
    adapter, request, _ = fixture_request(tmp_path)
    plan = adapter.build_command(request, tmp_path)
    with pytest.raises(RuntimeError, match="preflight failed"):
        adapter.execute(plan, request, {"ready_to_launch": False})
    assert not Path(plan["output_dir"]).exists()


def test_execute_refuses_input_changed_after_preflight(tmp_path):
    adapter, request, _ = fixture_request(tmp_path)
    plan = adapter.build_command(request, tmp_path)
    (tmp_path / "source.png").write_bytes(b"changed after planning")
    with pytest.raises(ValueError, match="RGB changed"):
        adapter.execute(plan, request, {"ready_to_launch": True})
    assert not Path(plan["output_dir"]).exists()
