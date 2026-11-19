"""The published checkout can regenerate its port without archived run code."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_compiles_port_against_adjacent_upstream(tmp_path):
    run = tmp_path / 'run'
    subprocess.run([sys.executable, str(ROOT / 'scripts/bootstrap_runtime.py'),
                    '--run', str(run), '--upstream', str(ROOT.parent)], check=True)
    runtime = run / 'runtime/amani_comparison'
    for path in runtime.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    provenance = json.loads((runtime / 'robotfitter-mano-provenance.json').read_text())
    assert provenance['upstream_sha256'] != provenance['port_sha256']
    assert (runtime / 'mano_adapter.py').is_file()
    assert not (run / 'assets').exists()
    assert not (run / 'inputs').exists()
    # A stale run must not silently receive a different fitter source.
    source = run / 'upstream/artirecon/fit_robot.py'
    source.write_text(source.read_text() + '\n# changed source\n')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/bootstrap_runtime.py'),
                             '--run', str(run), '--upstream', str(ROOT.parent)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Refusing to replace different upstream code' in result.stderr
