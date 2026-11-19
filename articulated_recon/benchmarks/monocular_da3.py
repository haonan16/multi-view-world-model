"""DA3 Metric monocular depth with an explicitly learned focal-length anchor.

The standalone metric head predicts focal-normalized depth. Following DA3's
released apply_metric_scaling, multiply by mean focal length / 300 on its output
grid. Here the focal length comes from UniDepth's source-camera prediction, never
dataset calibration. This is not an independent camera/metric-scale oracle.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256,load_manifest,validate_manifest


def metric_depth(raw, estimated_k, native_hw):
    raw=np.asarray(raw,np.float32); k=np.asarray(estimated_k,float)
    if raw.ndim!=2 or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('Expected a depth grid and finite positive estimated focal lengths')
    if len(native_hw)!=2 or min(native_hw)<=0:raise ValueError('Invalid original image size')
    h,w=native_hw
    mean_focal=(k[0,0]*raw.shape[1]/w+k[1,1]*raw.shape[0]/h)/2
    return raw*(mean_focal/300),float(mean_focal/300)


def run(*, source_geometry, checkpoint, output, process_res=518):
    import cv2
    from depth_anything_3.api import DepthAnything3
    source_geometry,checkpoint,output=map(lambda p:Path(p).resolve(),(source_geometry,checkpoint,output))
    if output.exists():raise FileExistsError(output)
    report=load_manifest(source_geometry/'input-manifest.json',[('cameras',source_geometry/'geometry.npz')])
    if report['condition']!='rgb_only':raise ValueError('Focal anchor must be estimated from RGB')
    manifest=json.loads((source_geometry/'input-manifest.json').read_text())
    for a in manifest['artifacts']:
        if 'path' in a:a['path']=str((source_geometry/a['path']).resolve())
    artifacts={a['id']:a for a in manifest['artifacts']}
    source=artifacts['source_rgb']
    with np.load(source['path'],allow_pickle=False) as z:
        if set(z.files)!={'rgb','timestamps'}:raise ValueError('Source must be an isolated RGB bundle')
        rgb,times=z['rgb'].copy(),z['timestamps'].copy()
    with np.load(source_geometry/'geometry.npz',allow_pickle=False) as z:
        k=z['K'].copy();c=z['c2w'].copy()
        if not np.array_equal(rgb,z['rgb'][:,0]) or not np.array_equal(times,z['timestamps']):
            raise ValueError('Focal estimate belongs to a different source RGB')
    if k.shape!=(1,3,3) or c.shape!=(1,4,4) or not np.allclose(c[0],np.eye(4)):
        raise ValueError('Expected one fixed source camera, identity gauge')
    cfg=json.loads((checkpoint/'config.json').read_text())
    if cfg.get('model_name')!='da3metric-large':raise ValueError('Use the standalone DA3 Metric Large checkpoint')
    output.mkdir(parents=True);start=time.monotonic()
    receipt=dict(status='running',frames=len(rgb),process_res=process_res,
        method='DA3 Metric Large focal-normalized head + UniDepth-estimated fixed source K',
        source_K_estimated=True,supplied_dataset_calibration=False,sensor_depth_used=False,
        confidence_available=False,neutral_weight_interpretation='ones are neutral weights, not confidence ranks or probabilities',
        code_sha256=file_sha256(__file__),checkpoint_revision=checkpoint.name)
    try:
        model=DepthAnything3.from_pretrained(str(checkpoint),local_files_only=True).to('cuda').eval()
        receipt['model_load_seconds']=time.monotonic()-start
        frames=[];seconds=[];scales=[];native=[]
        for i,im in enumerate(rgb):
            before=time.monotonic()
            pred=model.inference([im],extrinsics=None,intrinsics=None,process_res=process_res,
                                 process_res_method='upper_bound_resize',ref_view_strategy='first')
            if bool(pred.is_metric):
                raise ValueError('API already reports scaled metric depth; refuse possible double focal scaling')
            raw=np.asarray(pred.depth,np.float32)
            if raw.ndim!=3 or len(raw)!=1:raise ValueError('Unexpected standalone metric-head shape')
            depth,focal_scale=metric_depth(raw[0],k[0],im.shape[:2])
            frames.append(cv2.resize(depth,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR))
            native.append(list(raw.shape[-2:]));scales.append(focal_scale);seconds.append(time.monotonic()-before)
            print(f'DA3 Metric {i+1}/{len(rgb)}',flush=True)
        depth=np.stack(frames)[:,None];valid=np.isfinite(depth)&(depth>0)
        if not valid.any():raise ValueError('No positive finite predicted depth')
        np.savez_compressed(output/'geometry.npz',rgb=rgb[:,None],timestamps=times,depth_m=depth,
                            valid=valid,K=k,c2w=c,confidence_rank=np.ones_like(depth),
                            confidence_available=np.array(False),focal_scale=np.array(scales))
        # Retain full focal-estimation ancestry; old depth is not an active method
        # input unless its camera estimator itself declares that dependency.
        active=set(report['method_artifact_ids'])
        manifest['artifacts']=[a for a in manifest['artifacts'] if a['id'] in active]
        for a in manifest['artifacts']:
            if a['id']=='predicted_geometry':a['id']='unidepth_depth_unused'
            if a['id']=='predicted_cameras':a['id']='unidepth_focal_anchor'
            a['parent_ids']=[{'predicted_geometry':'unidepth_depth_unused','predicted_cameras':'unidepth_focal_anchor'}.get(p,p) for p in a['parent_ids']]
        manifest['artifacts'].append(dict(id='da3_metric_weights',kind='model_prior',origin='prior',parent_ids=[],
            prior_type='learned_model_weights',scope='global',model_name='depth-anything/DA3METRIC-LARGE',revision=checkpoint.name))
        for aid,kind in [('predicted_geometry','depth'),('predicted_cameras','cameras')]:
            manifest['artifacts'].append(dict(id=aid,kind=kind,origin='model_prediction',
                parent_ids=['source_rgb','unidepth_focal_anchor','da3_metric_weights'],path='geometry.npz',
                sha256=file_sha256(output/'geometry.npz')))
        manifest['method_inputs']=['source_rgb','predicted_geometry','predicted_cameras']
        manifest['metric_scale']=dict(status='estimated',artifact_ids=['predicted_geometry'],
            description='Learned focal-normalized DA3 Metric depth times UniDepth-estimated focal/300; no reference alignment')
        manifest['robot_registration']=dict(status='unresolved',artifact_ids=[],description='No target-robot registration')
        receipt['validation']=validate_manifest(manifest,output,verify_files=True)
        (output/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
        receipt.update(status='complete',frame_inference_seconds=seconds,native_depth_hw=native,
                       focal_multiplier=scales,geometry_sha256=file_sha256(output/'geometry.npz'))
    except Exception as e:
        receipt.update(status='failed',error=repr(e));raise
    finally:
        receipt['seconds']=time.monotonic()-start
        (output/'inference.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source-geometry','checkpoint','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--process-res',type=int,default=518)
    run(**vars(p.parse_args()))
