"""Seal offline plans and executor configuration before any simulator evaluation."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(case_dir, executor, friction=1.):
    case_dir=Path(case_dir)
    if (case_dir/'execution-started.json').exists():
        raise ValueError('This input directory has already been evaluated')
    files={p.name:digest(p) for p in sorted(case_dir.iterdir()) if p.is_file() and p.name!='frozen-plan.json'}
    if 'plan.npz' not in files or 'manifest.json' not in files:
        raise ValueError('Compile the complete observation-only plan first')
    payload=dict(schema='frozen-observation-plan-v1',created_utc=datetime.now(timezone.utc).isoformat(),
                 files=files,executor_sha256=digest(executor),friction=friction,
                 protocol='One execution. No plan, object geometry, grasp, controller or physics parameter changes after simulator feedback.',
                 allowed_feedback='Robot proprioception for fixed IK/PD execution only; object state and contacts are evaluation-only.',
                 limitation='Development informed by prior experiments; this is a new frozen evaluation, not evidence of never having used simulation during development.')
    with open(case_dir/'frozen-plan.json','x') as f:json.dump(payload,f,indent=2)
    return payload


def verify(case_dir, executor, friction):
    case_dir=Path(case_dir)
    payload=json.loads((case_dir/'frozen-plan.json').read_text())
    if payload['executor_sha256']!=digest(executor) or payload['friction']!=friction:
        raise ValueError('Frozen executor/configuration changed; refusing evaluation')
    for name,expected in payload['files'].items():
        if digest(case_dir/name)!=expected:
            raise ValueError(f'Frozen input changed: {name}')
    return payload


def claim_execution(case_dir, executor, friction, output):
    """Atomically consume a frozen plan before constructing the simulator.

    A failed infrastructure start still leaves its receipt for explicit audit;
    changing the output directory cannot silently retry a measured rollout.
    """
    case_dir=Path(case_dir)
    payload=verify(case_dir,executor,friction)
    receipt=dict(started_utc=datetime.now(timezone.utc).isoformat(),
                 frozen_plan_sha256=digest(case_dir/'frozen-plan.json'),
                 output=str(Path(output).resolve()))
    with open(case_dir/'execution-started.json','x') as f:
        json.dump(receipt,f,indent=2)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case_dir',type=Path);p.add_argument('--executor',type=Path,default=Path(__file__).with_name('execute_maniskill.py'));p.add_argument('--friction',type=float,default=1.);a=p.parse_args()
    print(json.dumps(freeze(a.case_dir,a.executor,a.friction),indent=2))
