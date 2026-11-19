import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import legacy_headless as migration


SOURCE = '''def run():
 env=make(render_mode='rgb_array',sim_backend='cpu',human_render_camera_configs={'shader_pack':'minimal'},sim_config={'control_freq':20,'sim_freq':120})
 env.reset(seed=0);freq=e.control_freq;writer=imageio.get_writer(out/'physics.mp4',fps=freq,codec='libx264',quality=8,macro_block_size=16)
 def step():
   env.step(action)
   records.append(row);frame=arr(e.render_rgb_array())[0].astype(np.uint8);writer.append_data(frame)
   if len(records)%40==0:imageio.imwrite(out/f'frame-{len(records):03d}.png',frame)
 writer.close();writer=None;np.save(out/'actions.npy',np.array(actions))
def scene():
 color=sapien.render.RenderMaterial(base_color=[.84,.56,.2,1],roughness=.6)
 for b,name,col in [(root,'housing',color),(cap,'cap',sapien.render.RenderMaterial(base_color=[.2,.65,.3,1]))]:
   b.add_visual_from_file(str(self.case_dir/(name+'.obj')),material=col);b.add_convex_collision_from_file('collision.obj')
 for b in builders:
   b.add_visual_from_file(str(self.case_dir/'object.obj'),material=color)
   b.add_box_collision(half_size=mesh.extents/2,density=500)
'''


def test_renderer_migration_retains_steps_control_settings_and_logs(monkeypatch):
    monkeypatch.setattr(migration, "EXECUTOR_SHA", hashlib.sha256(SOURCE.encode()).hexdigest())
    changed = migration.render_only_executor(SOURCE)
    assert "render_backend='none'" in changed
    assert "env.step(action)" in changed and "env.reset(seed=0)" in changed
    assert "sim_config={'control_freq':20,'sim_freq':120}" in changed
    assert "records.append(row)" in changed and "np.save(out/'actions.npy'" in changed
    assert "render_rgb_array()" not in changed and "get_writer(" not in changed
    assert "RenderMaterial(" not in changed and "add_visual_from_file(" not in changed
    assert "b.add_convex_collision_from_file('collision.obj')" in changed
    assert "b.add_box_collision(half_size=mesh.extents/2,density=500)" in changed


def test_unknown_predecessor_rejected():
    with pytest.raises(ValueError, match="Unexpected predecessor"):
        migration.render_only_executor(SOURCE)


@pytest.mark.parametrize("change", ["commands", "executor", "receipt"])
def test_sealed_bundle_rejects_mutation_or_repeat(tmp_path, change):
    executor = tmp_path / "executor.py"
    executor.write_text("fixed controller")
    commands = tmp_path / "plan.npz"
    commands.write_text("fixed commands")
    (tmp_path / "frozen-plan.json").write_text(json.dumps({"executor_sha256": migration.sha(executor),
        "friction": 1., "files": {commands.name: migration.sha(commands)}}))
    migration.verify_bundle(tmp_path, executor)
    if change == "receipt":
        (tmp_path / "execution-started.json").write_text("{}")
    else:
        (commands if change == "commands" else executor).write_text("changed")
    with pytest.raises(ValueError):
        migration.verify_bundle(tmp_path, executor)
