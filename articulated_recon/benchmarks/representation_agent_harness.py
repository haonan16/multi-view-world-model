"""Prepare and instrument matched fresh Codex agents; no launch without --execute.

Adapted from the local archived agent_reconstruction/run.py isolation pattern.
Only source code was consulted, not archived candidates or evaluation metrics.
CLI turns are NOT individual model API requests; unavailable effort stays null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

from representation_packet import verify_packet, write_json
from representation_agent_tools import validate_export


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    for name in ['time_limit_seconds', 'max_agent_turns', 'max_tool_calls', 'max_reported_tokens', 'max_scene_checkpoints']:
        value = config.get(name)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
            raise ValueError(name + ' must be positive')
    if not config.get('model') or not Path(config['python']).is_file():
        raise ValueError('Config requires explicit model and existing Python runtime')


PREFLIGHT_TEXT='source inputs read-only; output writable; disallowed reads/network denied'


def assert_fresh_packet(workspace):
    """Never silently delete a previous reconstruction or reuse a measured run."""
    workspace = Path(workspace)
    verify_packet(workspace)
    if (workspace/'execution-receipt.json').exists():
        raise ValueError('Packet already has an execution receipt; prepare a fresh replica')
    output = workspace/'output'
    if output.exists():
        for path in output.rglob('*'):
            if path.is_dir():
                continue
            if path != output/'preflight.txt' or path.read_text() != PREFLIGHT_TEXT:
                raise ValueError('Fresh agent output is not empty: '+str(path))


def prepare_suite(packets, output, seed=20261119):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    runs, configs, prompts, groups = [], set(), set(), {}
    for i, path in enumerate(packets):
        path = Path(path).resolve()
        frozen = verify_packet(path)
        assert_fresh_packet(path)
        packet = json.loads((path/'packet.json').read_text())
        config = json.loads((path/'config.json').read_text())
        validate_config(config)
        configs.add(frozen['config_sha256'])
        prompts.add(frozen['prompt_sha256'])
        # Only immutable source evidence and selected frames define a matched pair.
        key = (packet['source_parents']['observations_sha256'], tuple(packet['native_frame_ids']))
        groups.setdefault(key, []).append((packet, frozen))
        runs.append(dict(id=f'{i:03d}-{packet["condition"]}-{packet["selected_frame_count"]}frames',
            workspace=str(path), condition=packet['condition'], native_frame_ids=packet['native_frame_ids'],
            frozen_inputs_sha256=digest(path/'frozen-inputs.json'),
            status='prepared'))
    if len(configs) != 1 or len(prompts) != 1:
        raise ValueError('Every comparison requires the exact same frozen config and prompt')
    for group in groups.values():
        conditions = {packet['condition'] for packet,_ in group}
        if conditions != {'A_raw', 'B_object'} or len(group) != 2:
            raise ValueError('Each source/frame selection requires exactly one A_raw and B_object packet')
        first, second = group
        if first[1]['common_evidence_sha256'] != second[1]['common_evidence_sha256']:
            raise ValueError('Matched representations must contain byte-identical selected observations')
        for key in ['shared_asset_sha256', 'masks_sha256', 'tracks_sha256']:
            if first[0]['source_parents'].get(key) != second[0]['source_parents'].get(key):
                raise ValueError('Matched representations differ in raw evidence: ' + key)
        if first[1]['files'].get('tracks.npz') != second[1]['files'].get('tracks.npz'):
            raise ValueError('Matched representations differ in selected point correspondences')
        if first[0]['frontend_context_frame_ids'] != second[0]['frontend_context_frame_ids']:
            raise ValueError('Matched representations differ in upstream frontend context')
    random.Random(seed).shuffle(runs)
    output.mkdir(parents=True)
    suite = dict(schema_version=1, runs=runs, seed=seed, config=config,
        config_sha256=next(iter(configs)), prompt_sha256=next(iter(prompts)),
        runner_sha256=digest(__file__), agent_status='not_started',
        experimental_factors=['representation A_raw vs B_object', 'agent selected-frame budget'],
        limitations=['No claim of sparse frontend inference from a sparse agent packet.',
                     'Source agreement is not GT geometry/contact/execution accuracy.',
                     'One stochastic attempt per cell is descriptive, not a stable ranking.'])
    write_json(output/'suite.json', suite)
    return suite


def permission_overrides(config, workspace, frozen):
    workspace = Path(workspace).resolve()
    python = Path(config['python'])
    # Source files are readable. Only output/ is writable. Sibling packets and
    # the repository are absent from the allowlist, not just forbidden in prose.
    fs = {':minimal':'read', str(workspace):'read', str(workspace/'output'):'write',
          str(python.parent.parent):'read', str(python.resolve().parents[1]):'read'}
    if str(python.resolve()).startswith('/opt/homebrew/'):
        fs['/opt/homebrew'] = 'read'
    filesystem = '{' + ', '.join(json.dumps(k)+' = '+json.dumps(v) for k,v in fs.items()) + '}'
    values = ['default_permissions="source_reconstruction"',
        'permissions.source_reconstruction.filesystem='+filesystem,
        'permissions.source_reconstruction.network.enabled=false',
        'approval_policy="never"', 'web_search="disabled"', 'project_doc_max_bytes=0',
        'shell_environment_policy.inherit="none"',
        'shell_environment_policy.set={PATH="/usr/bin:/bin:/usr/sbin:/sbin",PYTHONDONTWRITEBYTECODE="1"}',
        'model_reasoning_effort='+json.dumps(config['reasoning_effort'])]
    for name in ['apps','plugins','memories','multi_agent','browser_use','browser_use_external',
                 'computer_use','shell_snapshot','skill_search']:
        values.append(f'features.{name}=false')
    values.append('features.skip_host_skill_discovery=true')
    return [arg for value in values for arg in ['-c', value]]


def preflight(codex, config, workspace, frozen, blocked_paths):
    """Run the actual selected sandbox, including denied reads/writes/network."""
    script = r'''
import json, os, pathlib, socket, sys
import numpy, scipy, PIL
root=pathlib.Path.cwd()
assert (root/'observations.npz').is_file()
assert (root/'PROMPT.txt').read_text()
for path in json.loads(sys.argv[1]):
    try:
        with open(path,'rb') as stream: stream.read(1)
    except PermissionError: pass
    else: raise RuntimeError('Read isolation failed: '+path)
for name in ['observations.npz','config.json','scene_tools.py','frozen-inputs.json']:
    try: fd=os.open(root/name, os.O_WRONLY)
    except PermissionError: pass
    else:
        os.close(fd)
        raise RuntimeError('Input write protection failed: '+name)
try: socket.create_connection(('1.1.1.1',443),timeout=2)
except OSError: pass
else: raise RuntimeError('Network isolation failed')
(root/'output'/'preflight.txt').write_text('source inputs read-only; output writable; disallowed reads/network denied')
print('PASS: source-only sandbox')
'''
    cmd = [codex,'sandbox','-P','source_reconstruction','-C',str(workspace),
           *permission_overrides(config,workspace,frozen),'--',config['python'],'-c',script,json.dumps(blocked_paths)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Source-only sandbox preflight failed: '+result.stdout+result.stderr)
    return result.stdout.strip()


def event_accounting(path):
    """Handle JSONL partial tails, duplicate item updates and absent usage."""
    turns, tokens, completed, started_tools = set(), [], {}, set()
    parse_errors, turn_count = 0, 0
    if not Path(path).exists():
        lines = []
    else:
        lines = Path(path).read_text(errors='replace').splitlines()
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # A running process may currently be writing the final line.
            parse_errors += int(index != len(lines)-1)
            continue
        kind = event.get('type')
        if kind == 'turn.completed':
            turn_count += 1
            if isinstance(event.get('usage'), dict):
                tokens.append(event['usage'])
        item = event.get('item', {})
        item_type = item.get('type')
        if kind in {'item.started','item.completed'} and item_type in {
                'command_execution','mcp_tool_call','web_search','image_view','file_change','tool_call'}:
            identifier = item.get('id', f'event_{index}')
            started_tools.add(identifier)
            if kind == 'item.completed':
                completed[identifier] = item
    usage = {name: sum(x.get(name,0) for x in tokens) if tokens else None
             for name in ['input_tokens','cached_input_tokens','output_tokens']}
    return dict(agent_invocations=1, completed_agent_turns=turn_count,
        model_requests=None, model_request_count_status='not_exposed_by_cli_turn_events',
        tool_calls_started=len(started_tools), tool_calls_completed=len(completed),
        failed_tool_calls=sum(x.get('status')=='failed' or (x.get('exit_code') not in {None,0}) for x in completed.values()),
        parse_errors=parse_errors, monetary_cost=None, **usage)


def budget_exceeded(accounting, elapsed, config):
    if elapsed >= config['time_limit_seconds']:
        return 'wall_time'
    if accounting['tool_calls_started'] > config['max_tool_calls']:
        return 'tool_calls'
    if accounting['completed_agent_turns'] > config['max_agent_turns']:
        return 'agent_turns'
    if accounting['input_tokens'] is not None and accounting['output_tokens'] is not None:
        if accounting['input_tokens']+accounting['output_tokens'] > config['max_reported_tokens']:
            return 'reported_tokens'
    return None


def checkpoint(scene_path, packet, directory, records, elapsed, maximum):
    if not scene_path.exists():
        return
    try:
        raw = scene_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if any(r['sha256']==sha for r in records):
            return
        scene = json.loads(raw)
        counts = validate_export(scene, packet)
    except (ValueError, OSError, KeyError):
        return
    if len(records) >= maximum:
        return
    name = f'{len(records):03d}.json'
    (directory/name).write_bytes(raw)
    records.append(dict(path=name, sha256=sha, first_observed_elapsed_seconds=elapsed, counts=counts))
    write_json(directory/'index.json', records)


def terminate_group(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def run_one(suite_path, run_id, codex='codex', execute=False):
    suite_path = Path(suite_path).resolve()
    suite = json.loads((suite_path/'suite.json').read_text())
    if suite['runner_sha256'] != digest(__file__):
        raise ValueError('Runner changed since suite freeze')
    run = next((r for r in suite['runs'] if r['id']==run_id), None)
    if run is None:
        raise ValueError('Unknown run ID')
    workspace = Path(run['workspace'])
    if digest(workspace/'frozen-inputs.json') != run['frozen_inputs_sha256']:
        raise ValueError('Frozen input inventory changed')
    frozen = verify_packet(workspace)
    assert_fresh_packet(workspace)
    config = suite['config']
    validate_config(config)
    if digest(workspace/'config.json') != suite['config_sha256']:
        raise ValueError('Frozen config changed')
    packet = json.loads((workspace/'packet.json').read_text())
    (workspace/'output').mkdir(exist_ok=True)
    blocked = [str(Path(r['workspace'])/'observations.npz') for r in suite['runs'] if r['id'] != run_id]
    blocked += [str(suite_path/'suite.json'), str(Path(__file__).resolve())]
    result = dict(run_id=run_id, executed=False, preflight=preflight(codex,config,workspace,frozen,blocked))
    if not execute:
        return result
    logs = suite_path/'logs'/run_id
    logs.mkdir(parents=True, exist_ok=False)
    checkpoints = logs/'checkpoints'
    checkpoints.mkdir()
    # No retries/resume: one exclusive log directory per pre-registered run.
    cmd = [codex,'exec','--ignore-user-config','--strict-config','--ephemeral',
        '--skip-git-repo-check','--json','--color','never','-m',config['model'],
        '-C',str(workspace),'-o',str(logs/'final.txt'),
        *permission_overrides(config,workspace,frozen),'-']
    start_wall, start_mono = time.time(), time.monotonic()
    receipt = dict(schema_version=1,run_id=run_id,suite_sha256=digest(suite_path/'suite.json'),
                   frozen_inputs_sha256=run['frozen_inputs_sha256'],started_unix=start_wall,
                   description='Exclusive fresh-agent invocation receipt; consumed even if launch fails')
    with (workspace/'execution-receipt.json').open('x') as stream:
        json.dump(receipt,stream,indent=2)
    result.update(executed=True,status='running',started_unix=start_wall, model=config['model'],
                  config_sha256=suite['config_sha256'], prompt_sha256=suite['prompt_sha256'])
    write_json(logs/'status.json', result)
    records = []
    with (logs/'events.jsonl').open('w') as events, (logs/'stderr.txt').open('w') as errors:
        proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=events,stderr=errors,text=True,start_new_session=True)
        try:
            proc.stdin.write((workspace/'PROMPT.txt').read_text())
            proc.stdin.close()
            while proc.poll() is None:
                active, wall = time.monotonic()-start_mono, time.time()-start_wall
                checkpoint(workspace/'output/scene-sequence.json',packet,checkpoints,records,active,config['max_scene_checkpoints'])
                reason = budget_exceeded(event_accounting(logs/'events.jsonl'),max(active,wall),config)
                if abs(active-wall)>5:
                    reason = 'runtime_clock_interruption'
                if reason:
                    result.update(status='stopped_at_budget',stop_reason=reason)
                    terminate_group(proc)
                    break
                try:
                    proc.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            else:
                result['status'] = 'completed' if proc.returncode==0 else 'failed'
        except BaseException:
            terminate_group(proc)
            result['status']='interrupted'
            write_json(logs/'status.json',result)
            raise
    elapsed = time.monotonic()-start_mono
    checkpoint(workspace/'output/scene-sequence.json',packet,checkpoints,records,elapsed,config['max_scene_checkpoints'])
    accounting = event_accounting(logs/'events.jsonl')
    result.update(exit_code=proc.returncode, elapsed_seconds=elapsed, wall_elapsed_seconds=time.time()-start_wall,
                  effort=accounting, distinct_valid_scene_checkpoints=len(records),
                  scene_edits=max(0,len(records)-1) if records else None,
                  edit_count_scope='externally sampled distinct valid final-scene hashes; not every file write',
                  final_scene_sha256=None, final_scene_valid=False,
                  source_geometry_accuracy=None, contact_accuracy=None, execution_success=None)
    final = workspace/'output/scene-sequence.json'
    if final.exists():
        result['final_scene_sha256'] = digest(final)
        try:
            result['final_scene_counts'] = validate_export(json.loads(final.read_text()),packet)
            result['final_scene_valid'] = True
        except (ValueError, KeyError) as exc:
            result['final_scene_error'] = str(exc)
    try:
        verify_packet(workspace)
    except ValueError as exc:
        result.update(status='invalid_input_mutation',input_error=str(exc))
    reason = budget_exceeded(accounting,elapsed,config)
    if reason and result['status']=='completed':
        result.update(status='completed_over_reported_budget',stop_reason=reason)
    result['source_check_reports'] = [str(p.relative_to(workspace)) for p in sorted((workspace/'output').rglob('source-check.json'))]
    result['render_inspection_status']='agent_report_required; file existence does not prove visual inspection'
    write_json(logs/'status.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--packet', type=Path, action='append', required=True)
    prepare.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--run-id', required=True)
    run.add_argument('--codex', default='codex')
    run.add_argument('--execute', action='store_true')
    a = p.parse_args()
    result = prepare_suite(a.packet,a.output) if a.command=='prepare' else run_one(a.suite,a.run_id,a.codex,a.execute)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
