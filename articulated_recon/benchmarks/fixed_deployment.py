"""Choose one virtual Panda deployment from a declared RGB-derived initializer.

This defines a virtual benchmark setup, not measured real-robot calibration. It
must be reused unchanged across methods. The independent evaluator transforms
its own reference by the SAME transform; its object errors are not aligned away.
Full-video-derived setup cannot be passed to a first-frame forecast protocol.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256,load_manifest,validate_manifest
from artirecon.scene_sequence import load_scene


def virtual_transform(scene,workspace_xy=(.45,0.)):
    if scene['coordinate_frame']['name']!='source_camera' or scene['support'] is None:
        raise ValueError('Virtual setup requires source-camera geometry and an estimated plane')
    if len(scene['objects'])!=1:
        raise ValueError('Declare an explicit target before using multi-object virtual deployment')
    obj=scene['objects'][0]
    if not obj['observed'][0]:raise ValueError('Initial object observation is missing')
    n=np.asarray(scene['support']['normal_world'],float);d=float(scene['support']['offset_m'])
    p=np.asarray(obj['T_world_object'][0],float)[:3,3]
    # Only rigid coordinate setup. No shape rescaling or individual pose edits.
    forward=np.array([0.,0.,1.]);forward-=n*(n@forward)
    if np.linalg.norm(forward)<1e-5:
        forward=np.array([1.,0.,0.]);forward-=n*(n@forward)
    forward/=np.linalg.norm(forward)
    rotation=np.stack([forward,np.cross(n,forward),n])
    table_foot=p-n*(p@n+d)
    transform=np.eye(4);transform[:3,:3]=rotation
    transform[:3,3]=np.array([*workspace_xy,0.])-rotation@table_foot
    return transform


def absolute_manifest(path):
    path=Path(path).resolve();report=load_manifest(path)
    manifest=json.loads(path.read_text())
    for a in manifest['artifacts']:
        if 'path' in a:a['path']=str((path.parent/a['path']).resolve())
    return manifest,report


def create(scene_path,input_manifest,source_camera_id,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    manifest,report=absolute_manifest(input_manifest)
    if report['condition']!='rgb_only' or report['task_mode']!='reconstruction':
        raise ValueError('Current virtual setup is derived from the RGB-only reconstruction baseline')
    scene_path=Path(scene_path).resolve()
    matches=[a['id'] for a in manifest['artifacts'] if a['id'] in report['method_artifact_ids']
             and a['kind']=='object_geometry' and a.get('path')==str(scene_path)]
    if len(matches)!=1:raise ValueError('Bind the actual source initializer as one active object_geometry artifact')
    scene=load_scene(scene_path,require_metric_scale=True)
    transform=virtual_transform(scene)
    deployment={'schema':'deployment-registration-v1','source_frame':'source_camera','target_frame':'panda_base',
        'T_robot_scene':transform.tolist(),'deployment_type':'virtual','fixed_across_methods':True,
        'derived_from_evaluation_geometry':False,'chosen_before_evaluation':True,
        'source_camera_id':source_camera_id,'source_coordinate_convention':'opencv',
        'procedure':'One source-only baseline defines table-up and initial estimated table foot. Place that foot at Panda workspace x=.45m,y=0; preserve all relative poses and shape. Reuse this exact transform for every method and the independent reference.',
        'measured_physical_calibration':False,'source_initializer_sha256':file_sha256(scene_path),
        'limitations':['Virtual deployment convention, not a claim about a real Panda in the human recording.',
                      'Estimated table orientation/scale errors remain and affect the independent scene.',
                      'Full-video-derived provenance; cannot be reused as RGB-only first-frame forecasting input.']}
    output.mkdir(parents=True);path=output/'deployment.json';path.write_text(json.dumps(deployment,indent=2))
    manifest['artifacts'].append({'id':'fixed_deployment','kind':'robot_registration','origin':'model_prediction',
        'parent_ids':matches,'path':str(path),'sha256':file_sha256(path)})
    manifest['method_inputs'].append('fixed_deployment')
    manifest['robot_registration']={'status':'estimated','artifact_ids':['fixed_deployment'],
        'description':'Fixed virtual Panda placement from the declared source-only baseline; no evaluator geometry'}
    validate_manifest(manifest,output,verify_files=True)
    (output/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    return deployment


def attach(method_manifest,deployment_manifest,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    method,method_report=absolute_manifest(method_manifest)
    deployment,dep_report=absolute_manifest(deployment_manifest)
    if method['task_mode']!=deployment['task_mode']:
        raise ValueError('Do not leak full-video setup into first-frame forecasting')
    if method['condition']!=deployment['condition']:
        raise ValueError('Deployment and method conditions differ')
    def sources(manifest, report):
        return {a['sha256'] for a in manifest['artifacts'] if a['id'] in report['method_artifact_ids']
                and a['kind']=='rgb' and a['origin']=='observed_rgb'}
    if sources(method,method_report)!=sources(deployment,dep_report):
        raise ValueError('Estimated virtual deployment must use the identical observed source clip')
    active=set(dep_report['method_artifact_ids'])
    prefix='deployment__'
    if any(a['id'].startswith(prefix) for a in method['artifacts']):raise ValueError('Deployment already attached')
    for a in deployment['artifacts']:
        if a['id'] not in active:continue
        copied=deepcopy(a);copied['id']=prefix+a['id'];copied['parent_ids']=[prefix+x for x in a['parent_ids']]
        method['artifacts'].append(copied)
    method['method_inputs'].append(prefix+'fixed_deployment')
    method['robot_registration']=deepcopy(deployment['robot_registration'])
    method['robot_registration']['artifact_ids']=[prefix+x for x in method['robot_registration']['artifact_ids']]
    validate_manifest(method,output.parent,verify_files=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(method,f,indent=2)
    return {'manifest':str(output),'registration_artifact_id':prefix+'fixed_deployment',
            'shared_registration_sha256':next(a['sha256'] for a in deployment['artifacts'] if a['id']=='fixed_deployment')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    c=subs.add_parser('create')
    for n in ['scene-path','input-manifest','output']:c.add_argument('--'+n,type=Path,required=True)
    c.add_argument('--source-camera-id',required=True)
    a=subs.add_parser('attach')
    for n in ['method-manifest','deployment-manifest','output']:a.add_argument('--'+n,type=Path,required=True)
    args=vars(p.parse_args());command=args.pop('command');print(json.dumps((create if command=='create' else attach)(**args),indent=2))
