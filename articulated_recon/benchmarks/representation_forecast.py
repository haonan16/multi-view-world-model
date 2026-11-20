"""Freeze paired fresh-agent forecast inputs; no model or agent is launched.

Raw evidence is DA3 plus one first-image metric ratio. Fitted future scale is
part of the initializer/candidate, never silently folded into the raw condition.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
from artirecon.input_protocol import file_sha256,load_manifest,validate_manifest
from artirecon.scene_sequence import load_scene
from representation_packet import normalize_observations,default_config,write_json,verify_packet
from representation_agent_tools import contact_sheet,validate_export,candidate_scales
import forecast_geometry as fg
from da3_helper_views import read_inputs


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def comparison_initial_scene(control,scene_path):
    """Export a controller-only first-image table reference for sealed scoring."""
    control,scene_path=Path(control),Path(scene_path).resolve()
    output=control/'initial_scene_comparison';output.mkdir()
    manifest=json.loads((control/'initial_scene/input-manifest.json').read_text())
    shutil.copy2(scene_path,output/'scene-sequence.json')
    for a in manifest['artifacts']:
        if a.get('path')==str(scene_path):a['path']=str((output/'scene-sequence.json').resolve())
    write_json(output/'input-manifest.json',manifest)
    load_manifest(output/'input-manifest.json',[('object_geometry',output/'scene-sequence.json')])
    return output


def portable_inputs(condition,helper,predictions,masks,initializer,output,remote_prefix=None,local_prefix=None):
    """Materialize controller-only manifest clones with an auditable path map.

    Original manifests/preparation are never changed. The strict existing
    first-image validator is run on the clones, including singleton ancestry and
    byte-exact reconstruction of the conditioning arrays from the first image.
    """
    if bool(remote_prefix)!=bool(local_prefix):raise ValueError('Provide both relocation prefixes')
    output=Path(output);output.mkdir(parents=True)
    def local(path):
        path=Path(path)
        if remote_prefix and path.is_absolute() and path.is_relative_to(remote_prefix):
            path=Path(local_prefix)/path.relative_to(remote_prefix)
        return path.resolve()
    condition,helper,predictions,masks,initializer=map(local,(condition,helper,predictions,masks,initializer))
    prep=json.loads((condition/'preparation.json').read_text())
    for key in ['geometry_manifest','scene_manifest','scene_path']:
        expected=prep['scene_sha256'] if key=='scene_path' else prep[key+'_sha256']
        if file_sha256(local(prep[key]))!=expected:raise ValueError('Original preparation ancestry changed: '+key)
    gm,sm=local(prep['geometry_manifest']),local(prep['scene_manifest'])
    originals=dict(condition=condition/'input-manifest.json',helper=helper/'input-manifest.json',
        predictions=predictions/'input-manifest.json',masks=masks/'input-manifest.json',
        initial_geometry=gm,initial_scene=sm,initializer=initializer/'input-manifest.json')
    replacements={}
    for label,source in [('condition',condition/'condition.npz'),('helper',helper/'observations.npz'),
                         ('initial_geometry',gm.parent/'geometry.npz')]:
        dest=output/label/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);replacements[source.resolve()]=dest.resolve()
    receipts=[]
    for label,original in originals.items():
        manifest=json.loads(original.read_text())
        for artifact in manifest['artifacts']:
            if 'path' in artifact:
                path=Path(artifact['path'])
                path=local(path if path.is_absolute() else original.parent/path)
                artifact['path']=str(replacements.get(path,path))
        target=output/label/'input-manifest.json';target.parent.mkdir(parents=True,exist_ok=True)
        write_json(target,manifest);load_manifest(target)
        receipts.append(dict(original=str(original),original_sha256=file_sha256(original),
            derived=str(target.resolve()),derived_sha256=file_sha256(target),operation='path-only relocation; artifact bytes/hashes unchanged'))
    changed=deepcopy(prep)
    for key,label in [('geometry_manifest','initial_geometry'),('scene_manifest','initial_scene')]:
        changed[key]=str((output/label/'input-manifest.json').resolve())
        changed[key+'_sha256']=file_sha256(changed[key])
    changed['scene_path']=str(local(prep['scene_path']))
    write_json(output/'condition/preparation.json',changed)
    write_json(output/'relocation-receipt.json',dict(original_preparation_sha256=file_sha256(condition/'preparation.json'),
        derived_preparation_sha256=file_sha256(output/'condition/preparation.json'),
        remote_prefix=str(remote_prefix) if remote_prefix else None,local_prefix=str(local_prefix) if local_prefix else None,
        manifest_clones=receipts,artifact_hashes_preserved=True))
    job=read_inputs(output/'helper',output/'condition')
    from benchmarks.strict_forecast import validate_forecast
    _,strict_report=validate_forecast(output/'condition')
    relocation_path=output/'relocation-receipt.json'
    relocation=json.loads(relocation_path.read_text());relocation['condition_recreation']=strict_report['condition_recreation']
    write_json(relocation_path,relocation)
    comparison_initial_scene(output,changed['scene_path'])
    return job,changed,dict(predictions=predictions,masks=masks,initializer=initializer)


def pair(*,condition,helper,predictions,masks,initializer,output,config,remote_prefix=None,local_prefix=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    control=output/'controller'
    job,prep,paths=portable_inputs(condition,helper,predictions,masks,initializer,control,remote_prefix,local_prefix)
    predictions,masks,initializer=[paths[k] for k in ['predictions','masks','initializer']]
    values=read_npz(predictions/'predictions.npz');mask_values=read_npz(masks/'masks.npz')
    receipt=json.loads((predictions/'execution.json').read_text());branch=receipt.get('condition')
    if (branch not in {'source_unposed','helper_unposed'} or receipt.get('status')!='complete'
            or receipt.get('task_mode')!='forecast' or receipt.get('output_sha256')!=file_sha256(predictions/'predictions.npz')):
        raise ValueError('Require completed authenticated source_unposed/helper_unposed predictions')
    views=1 if branch=='source_unposed' else 4
    if not np.array_equal(values['rgb'],job['rgb'][:,:views]) or not np.array_equal(values['timestamps'],job['timestamps']):
        raise ValueError('Predictions must match the selected generated pixels and timestamps')
    for label,kind,file in [('predictions','depth',predictions/'predictions.npz'),('predictions','cameras',predictions/'predictions.npz'),('masks','masks',masks/'masks.npz')]:
        load_manifest(control/label/'input-manifest.json',[(kind,file)])
    manifest=fg.merge_inputs([control/k/'input-manifest.json' for k in ['predictions','masks','initial_geometry','initial_scene']])
    validation=validate_manifest(manifest,control,verify_files=True)
    if branch=='source_unposed' and any(a['id'] in validation['method_artifact_ids'] and a.get('view_role')=='helper' for a in manifest['artifacts']):
        raise ValueError('Source-only masks/predictions must not consume helper RGB')
    write_json(control/'common-manifest.json',manifest)
    initial=load_scene(prep['scene_path'],require_metric_scale=True)
    fit_config=json.loads((initializer/'fit-config.json').read_text())
    fitted=load_scene(initializer/'scene-sequence.json',require_metric_scale=True)
    fit_receipt=json.loads((initializer/'execution.json').read_text())
    if fit_receipt.get('status')!='complete' or fit_receipt.get('scene_sha256')!=file_sha256(initializer/'scene-sequence.json') or fit_receipt.get('branch')!=branch:
        raise ValueError('Initializer needs its completed scene/branch receipt')
    init_manifest=json.loads((control/'initializer/input-manifest.json').read_text())
    init_report=load_manifest(control/'initializer/input-manifest.json',[('object_geometry',initializer/'scene-sequence.json')])
    common_records={a['id']:a for a in manifest['artifacts']}
    # Numerical initialization may add derived artifacts, but may not add evidence.
    for a in init_manifest['artifacts']:
        if a['id'] not in init_report['method_artifact_ids']:continue
        if a['id'] in common_records:
            if {k:v for k,v in a.items() if k!='path'}!={k:v for k,v in common_records[a['id']].items() if k!='path'}:
                raise ValueError('Initializer used different common evidence: '+a['id'])
        elif a.get('path') and Path(a['path']).parent!=initializer:
            raise ValueError('Initializer has additional external evidence: '+a['id'])
    frames=fg.prepare_evidence(values,mask_values,fit_config)
    initial_values=read_npz(control/'initial_geometry/geometry.npz')
    initial_depth=initial_values.get('depth_m',initial_values.get('depth'))[0,0]
    scale,anchor=fg.initial_scale_anchor(initial_depth,prep['source_crop_xywh'],values['depth'][0,0],frames[0][0]['object_mask'])
    raw_c=values['c2w'].copy();scaled_c=raw_c.copy();scaled_c[:,:,:3,3]*=scale
    observations=normalize_observations(dict(rgb=values['rgb'],depth_m=values['depth']*scale,
        K=values['K'],c2w=scaled_c,timestamps=values['timestamps'],observed_mask=values['observed_mask'],
        source_frame_indices=values['source_frame_indices'],object_masks=np.asarray([[v['object_mask'] for v in f] for f in frames]),
        object_mask_known=np.asarray([[v['known'] for v in f] for f in frames]),
        occlusion_mask=np.asarray([[v['hand_mask'] for v in f] for f in frames])),task_mode='forecast')
    count=len(values['timestamps']);ids=np.arange(count)
    observations.update(raw_depth=values['depth'],raw_c2w=raw_c,native_frame_ids=ids,geometry_row_ids=ids)
    source_ids=sorted({a['id'] for a in manifest['artifacts'] if a['id'] in validation['method_artifact_ids']
        and (a['kind'] in {'depth','cameras','masks'} or a.get('generated') is True)})
    parents=dict(observations_sha256=file_sha256(predictions/'predictions.npz'),masks_sha256=file_sha256(masks/'masks.npz'),
        initial_scene_sha256=file_sha256(prep['scene_path']),optimizer_config_sha256=file_sha256(initializer/'fit-config.json'))
    # Save one array archive, then copy it exactly for the matched pair.
    np.savez_compressed(control/'common-observations.npz',**observations)
    for name in ['A_raw','B_object']:
        dest=output/name;dest.mkdir()
        shutil.copy2(control/'common-observations.npz',dest/'observations.npz')
        for source,target in [(predictions/'predictions.npz','raw-predictions.npz'),(masks/'masks.npz','raw-masks.npz'),
                (Path(prep['scene_path']),'first-image-scene.json'),(initializer/'fit-config.json','optimizer-config.json'),
                (HERE/'representation_agent_tools.py','scene_tools.py'),(HERE.parent/'artirecon/scene_sequence.py','scene_sequence.py'),
                (HERE/'representation_forecast_prompt.txt','PROMPT.txt')]:shutil.copy2(source,dest/target)
        for subdir,names in [('benchmarks',['forecast_geometry.py','da3_helper_views.py','representation_agent_tools.py']),
                             ('artirecon',['input_protocol.py','scene_sequence.py'])]:
            folder=dest/'optimizer'/subdir;folder.mkdir(parents=True)
            for file in names:shutil.copy2(HERE.parent/subdir/file,folder/file)
        (dest/'images').mkdir()
        for v in range(views):
            labels=[]
            for t in ids:
                Image.fromarray(values['rgb'][t,v]).save(dest/'images'/f'{t:03d}-view{v}.png')
                labels.append(f'{t}: {values["timestamps"][t]:.3f}s v{v} '+('observed' if t==v==0 else 'generated'))
            contact_sheet(list(values['rgb'][:,v]),labels,dest/'images'/f'contact-sheet-view-{v}.png')
        packet=dict(schema_version=1,task_mode='forecast',condition=name,evidence_branch=branch,
            source_parents=parents,coordinate_frame=initial['coordinate_frame'],metric_scale_status='estimated',
            observed_until_s=float(values['timestamps'][0]),timestamps=values['timestamps'].tolist(),
            native_frame_ids=ids.tolist(),geometry_row_ids=ids.tolist(),frontend_context_frame_ids=ids.tolist(),
            frame_id_semantics='forecast timeline row IDs; recorded source_frame_indices are 0 then -1, not observed future frames',
            requested_frame_budget=count,selected_frame_count=count,object_id=prep['object_id'],task_text=prep['task_text'],
            prediction_artifact_ids=source_ids,initializer='initializer-scene.json' if name=='B_object' else None,
            initializer_kind='shared_cuboid_SE3_and_per_time_scale' if name=='B_object' else None,
            forecast=dict(first_image_ratio=scale,common_depth_scale=scale,scale_anchor=anchor,
                common_reference='raw DA3 times one fixed first-image ratio; no fitted future scale',
                camera_gauge='Each unposed frame gauged to source camera; fixed source camera is an assumption; native K is retained'),
            evidence='observations.npz',images='images',point_tracks=None,canonical_asset=None,
            preprocessing_timing_seconds=dict(structured_initializer=fit_receipt.get('total_seconds') if name=='B_object' else None),
            measurement_template=dict(gt_geometry_accuracy=None,contact_accuracy=None,execution_success=None,monetary_cost=None),
            forbidden_inputs=['GT','recorded future frames','other agent outputs','simulator feedback','internet'])
        if name=='B_object':
            validate_export(fitted,packet);shutil.copy2(initializer/'scene-sequence.json',dest/'initializer-scene.json')
        write_json(dest/'packet.json',packet);write_json(dest/'config.json',config)
        hashes={str(p.relative_to(dest)):file_sha256(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        write_json(dest/'frozen-inputs.json',dict(schema_version=1,files=hashes,common_evidence_sha256=hashes['observations.npz'],
            prompt_sha256=hashes['PROMPT.txt'],config_sha256=hashes['config.json'],packet_builder_sha256=file_sha256(__file__)))
    write_json(control/'pair-receipt.json',dict(status='prepared',branch=branch,initializer=str(initializer),
        initial_preprocessing_seconds=fit_receipt.get('total_seconds'),common_depth_scale=scale,
        shared_raw_predictions_sha256=parents['observations_sha256'],shared_masks_sha256=parents['masks_sha256'],
        fitted_future_scales_in_common_evidence=False,agent_invocations=0))
    return dict(status='prepared',packets=[str(output/n) for n in ['A_raw','B_object']],branch=branch)


def export_bundle(*,packet,scene,output):
    packet,scene,output=map(lambda p:Path(p).resolve(),(packet,scene,output));verify_packet(packet)
    if output.exists():raise FileExistsError(output)
    meta=json.loads((packet/'packet.json').read_text());candidate=json.loads(scene.read_text());validate_export(candidate,meta)
    obs=read_npz(packet/'observations.npz');scales=candidate_scales(candidate,meta)
    c=obs['raw_c2w'].copy();c[:,:,:3,3]*=scales[:,None,None]
    depth=obs['raw_depth']*scales[:,None,None,None];obj=next(o for o in candidate['objects'] if o['id']==meta['object_id'])
    output.mkdir(parents=True);shutil.copy2(scene,output/'scene-sequence.json')
    np.savez_compressed(output/'geometry.npz',rgb=obs['rgb'],depth_m=depth,K=obs['K'],c2w=c,
        timestamps=obs['timestamps'],source_frame_indices=obs['source_frame_indices'],observed_mask=obs['observed_mask'],
        object_masks=obs['object_masks'],object_mask_known=obs['object_mask_known'],occlusion_mask=obs['occlusion_mask'],
        estimated_depth_scale=scales,prediction_supported=np.asarray(obj['prediction_supported']))
    control=packet.parent/'controller';manifest=json.loads((control/'common-manifest.json').read_text())
    parents=manifest['method_inputs'].copy()
    if meta['condition']=='B_object':
        manifest['artifacts'].append(dict(id='agent_initializer',kind='object_geometry',origin='model_prediction',parent_ids=parents,
            path=str(packet/'initializer-scene.json'),sha256=file_sha256(packet/'initializer-scene.json')))
        parents=parents+['agent_initializer']
    for aid,kind,file in [('agent_scene','object_geometry','scene-sequence.json'),('agent_depth','depth','geometry.npz'),('agent_cameras','cameras','geometry.npz')]:
        manifest['artifacts'].append(dict(id=aid,kind=kind,origin='model_prediction',parent_ids=parents,path=file,
            sha256=file_sha256(output/file),description='Agent hypothesis; display depth/cameras use candidate scales, not a common evaluation reference'))
    manifest['method_inputs']=['agent_scene','agent_depth','agent_cameras']
    manifest['metric_scale']=dict(status='estimated',artifact_ids=['agent_depth'],description='First-image RGB anchor plus agent scale hypothesis')
    validate_manifest(manifest,output,verify_files=True);write_json(output/'input-manifest.json',manifest)
    receipt=dict(status='complete',scene_sha256=file_sha256(output/'scene-sequence.json'),branch=meta['evidence_branch'],
        condition=meta['condition'],packet_sha256=file_sha256(packet/'frozen-inputs.json'),
        geometry_scope='candidate-conditioned visualization; primary comparison must use frozen packet observations',
        preprocessing_timing_seconds=meta['preprocessing_timing_seconds'],GT_used=False,simulator_used=False)
    write_json(output/'execution.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('config');c.add_argument('--python',required=True);c.add_argument('--model',default='gpt-6.2-sol');c.add_argument('--output',type=Path,required=True)
    b=sub.add_parser('pair')
    for name in ['condition','helper','predictions','masks','initializer','config','output']:b.add_argument('--'+name,type=Path,required=True)
    for name in ['remote-prefix','local-prefix']:b.add_argument('--'+name,type=Path)
    e=sub.add_parser('export')
    for name in ['packet','scene','output']:e.add_argument('--'+name,type=Path,required=True)
    a=vars(p.parse_args());command=a.pop('command')
    if command=='config':
        if a['output'].exists():raise FileExistsError(a['output'])
        value=default_config(a['model'],a['python']);value.update(seed=20261120,max_reported_tokens=1000000)
        write_json(a['output'],value);result=value
    elif command=='pair':a['config']=json.loads(a['config'].read_text());result=pair(**a)
    else:result=export_bundle(**a)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
