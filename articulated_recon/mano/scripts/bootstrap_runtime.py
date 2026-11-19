"""Build the tested MANO port from the adjacent articulated_recon source checkout.

Only code is copied. Inputs, model weights and licensed MANO data are supplied
separately. Source hashes and exact generated diffs remain in the run directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(run, upstream=None, models=None):
    run = Path(run).resolve()
    upstream = Path(upstream or ROOT.parent).resolve()
    if models is not None:
        models = Path(models).resolve()
        if not (models / 'MANO_RIGHT.npz').is_file():
            raise FileNotFoundError('Expected licensed converted MANO_RIGHT.npz in --models')
        link = run / 'assets/mano'
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.resolve() != models:
                raise FileExistsError(f'Existing model directory differs: {link}')
        else:
            link.symlink_to(models, target_is_directory=True)
    names = ['artirecon/fit_robot.py', 'artirecon/objects_v2.py', 'seg_sam3.py']
    for name in names:
        source = upstream / name
        if not source.is_file():
            raise FileNotFoundError(f'Missing upstream source: {source}')
    hashes = {}
    for name in names:
        source = upstream / name
        target = run / 'upstream' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError(f'Refusing to replace different upstream code in existing run: {target}')
        target.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_sam_comparison_runtime.py'),
                    '--run', str(run)], check=True)
    dest = run / 'runtime/amani_comparison'
    shutil.copyfile(ROOT / 'mano_recon/robotfitter_adapter.py', dest / 'mano_adapter.py')
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_robotfitter_mano.py'),
                    '--run', str(run)], check=True)
    (dest / 'bootstrap-provenance.json').write_text(json.dumps({
        'upstream_directory': str(upstream), 'sources_sha256': hashes,
        'assets_copied': False, 'inputs_copied': False,
    }, indent=2))
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=ROOT.parent)
    parser.add_argument('--models', type=Path, help='Link user-licensed converted MANO directory into this run; no files copied')
    args = parser.parse_args()
    print(bootstrap(args.run, args.upstream, args.models))


if __name__ == '__main__':
    main()
