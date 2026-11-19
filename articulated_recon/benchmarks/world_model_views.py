"""Generate helper views with the existing author Wan model from one RGB clip.

The virtual cameras are designed around RGB-derived scene geometry. They are
conditioning requests, not measured cameras or proof that generated pixels obey
those poses. No target-view first frames, dataset K, or evaluator data enter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import load_manifest, file_sha256, validate_manifest
from artirecon.scene_sequence import load_scene


def look_at_camera(centre, target):
    forward = np.asarray(target) - np.asarray(centre)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0., -1., 0.]))
    if np.linalg.norm(right) < 1e-6:
        raise ValueError('Virtual camera is parallel to the chosen image-up axis')
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = centre
    return pose


def virtual_cameras(target, angle_degrees=20.):
    target = np.asarray(target, float)
    if target.shape != (3,) or not np.isfinite(target).all() or target[2] <= .05:
        raise ValueError('Target must be a finite positive-depth source-camera point')
    theta = np.deg2rad(angle_degrees)
    if not 5 <= angle_degrees <= 40:
        raise ValueError('Use a declared moderate virtual baseline between 5 and 40 degrees')
    baseline = target[2] * np.tan(theta)
    centres = [[baseline, 0, 0], [-baseline, 0, 0], [0, -baseline*.65, 0]]
    return np.stack([np.eye(4)] + [look_at_camera(c, target) for c in centres])


def source_tensor(rgb, views=4, forecast=False, total_frames=None):
    frames = total_frames if forecast else len(rgb)
    if forecast and (len(rgb) != 1 or frames is None):
        raise ValueError('Forecast inputs must contain exactly one RGB frame')
    if frames < 1 or (frames - 1) % 4:
        raise ValueError('Wan temporal VAE requires 4k+1 frames')
    h, w = rgb.shape[1:3]
    pixels = np.zeros((1, views, frames, 3, h, w), np.float32)
    known = np.zeros((frames, views), bool)
    if forecast:
        pixels[0, 0, 0] = rgb[0].transpose(2, 0, 1).astype(np.float32)/127.5-1
        known[0, 0] = True
    else:
        pixels[0, 0] = rgb.transpose(0, 3, 1, 2).astype(np.float32)/127.5-1
        known[:, 0] = True
    return pixels, known


def prepare(*, geometry_bundle, scene_path, scene_manifest, output, frames=25):
    """Prepare reconstruction helper views from estimated source geometry only."""
    import cv2
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    geometry_bundle = Path(geometry_bundle).resolve()
    manifest_path = geometry_bundle/'input-manifest.json'
    report = load_manifest(manifest_path, [('depth', geometry_bundle/'geometry.npz'),
                                          ('cameras', geometry_bundle/'geometry.npz')])
    scene_report = load_manifest(scene_manifest)
    if report['condition'] != 'rgb_only' or scene_report['condition'] != 'rgb_only':
        raise ValueError('Helper preparation requires RGB-only reconstruction ancestry')
    if report['task_mode'] != 'reconstruction' or scene_report['task_mode'] != 'reconstruction':
        raise ValueError('This preparation currently supports full-video reconstruction only')
    original_scene = load_scene(scene_path, require_metric_scale=True)
    if original_scene['coordinate_frame']['name'] != 'source_camera':
        raise ValueError('Initializer must explicitly use the source_camera frame')
    scene_manifest_data = json.loads(Path(scene_manifest).read_text())
    geometry_manifest_data = json.loads(manifest_path.read_text())
    source_hashes = lambda m, r: {a.get('sha256') for a in m['artifacts'] if a['id'] in r['method_artifact_ids'] and a['kind']=='rgb' and a['origin']=='observed_rgb'}
    if not source_hashes(geometry_manifest_data,report) or source_hashes(geometry_manifest_data,report)!=source_hashes(scene_manifest_data,scene_report):
        raise ValueError('Geometry and initializer must derive from the identical observed source RGB')
    active = set(scene_report['method_artifact_ids'])
    scene_record = next((a for a in scene_manifest_data['artifacts'] if a['id'] in active
                         and a.get('path') and (Path(scene_manifest).resolve().parent/a['path']).resolve() == Path(scene_path).resolve()), None)
    if scene_record is None:
        raise ValueError('Initializer scene must be a materialized active manifest artifact')
    obj = original_scene['objects'][0]
    usable = np.asarray(obj['observed'], bool)
    if not usable.any():
        raise ValueError('No supported object pose for designing helper cameras')
    target = np.median(np.asarray(obj['T_world_object'])[usable, :3, 3], axis=0)
    with np.load(geometry_bundle/'geometry.npz', allow_pickle=False) as data:
        rgb = data['rgb'][:, 0].copy()
        k = data['K'][0].copy()
        c = data['c2w'][0]
        timestamps = data['timestamps'].copy()
    if not np.allclose(c, np.eye(4)):
        raise ValueError('Source coordinate convention must be camera identity')
    if frames > len(rgb) or frames < 5 or (frames-1) % 4:
        raise ValueError('Select 4k+1 existing frames, at least five')
    ids = np.rint(np.linspace(0, len(rgb)-1, frames)).astype(int)
    h, w = rgb.shape[1:3]
    # The checkpoint was trained on320x160. Preserve aspect by a centred2:1 crop.
    crop_h, crop_w = min(h, w//2), min(w, h*2)
    y0, x0 = (h-crop_h)//2, (w-crop_w)//2
    rgb = np.stack([cv2.resize(im[y0:y0+crop_h, x0:x0+crop_w], (320,160), interpolation=cv2.INTER_AREA) for im in rgb[ids]])
    k[0,2] -= x0; k[1,2] -= y0
    k[0] *= 320/crop_w; k[1] *= 160/crop_h
    cameras = virtual_cameras(target)
    output.mkdir(parents=True)
    np.savez_compressed(output/'condition.npz', rgb=rgb, K=np.repeat(k[None],4,axis=0),
                        c2w=cameras, timestamps=timestamps[ids], source_frame_indices=ids)
    for artifact in scene_manifest_data['artifacts']:
        if 'path' in artifact:
            artifact['path']=str((Path(scene_manifest).resolve().parent/artifact['path']).resolve())
    existing={a['id']:a for a in scene_manifest_data['artifacts']}
    for artifact in geometry_manifest_data['artifacts']:
        if artifact['id'] not in report['method_artifact_ids']:continue
        if 'path' in artifact:artifact['path']=str((manifest_path.parent/artifact['path']).resolve())
        if artifact['id'] in existing:
            for field in ['kind','origin','sha256','parent_ids']:
                if existing[artifact['id']].get(field)!=artifact.get(field):
                    raise ValueError('Conflicting geometry/initializer artifact lineage: '+artifact['id'])
        else:scene_manifest_data['artifacts'].append(artifact)
    scene_manifest_data['artifacts'].append({'id':'virtual_camera_requests','kind':'cameras',
        'origin':'model_prediction','parent_ids':list(dict.fromkeys([scene_record['id']]+report['method_inputs'])),
        'path':'condition.npz','sha256':file_sha256(output/'condition.npz'),
        'assumption':'Designed conditioning poses, not measured generated-view camera extrinsics'})
    scene_manifest_data['method_inputs'].append('virtual_camera_requests')
    validate_manifest(scene_manifest_data,output,verify_files=True)
    (output/'input-manifest.json').write_text(json.dumps(scene_manifest_data,indent=2))
    meta = {'mode':'reconstruction', 'observed_source_frames':len(rgb), 'extra_observed_views':0,
            'condition_sha256':file_sha256(output/'condition.npz'),
            'geometry_manifest':str(manifest_path), 'geometry_manifest_sha256':file_sha256(manifest_path),
            'scene_manifest':str(Path(scene_manifest).resolve()), 'scene_manifest_sha256':file_sha256(scene_manifest),
            'scene_sha256':file_sha256(scene_path), 'target_source_camera_m':target.tolist(),
            'camera_design':'Source identity plus three virtual cameras at fixed20degree baselines; depth-scaled, aim at estimated track median',
            'camera_interpretation':'Requested conditioning poses, not measured generated-view calibration',
            'source_crop_xywh':[x0,y0,crop_w,crop_h], 'supplied_dataset_cameras':False,
            'evaluator_geometry_used':False,'source_timestamps':timestamps[ids].tolist(),
            'training_caveat':'Configured training horizon17; source-only full-video history is accepted by sampler but differs from target-first-frame training history.'}
    (output/'preparation.json').write_text(json.dumps(meta,indent=2))
    return meta


def prepare_prefix(*, condition, output, frames=17):
    """Shorten source conditioning only; preserve the predeclared camera design."""
    condition, output = Path(condition).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    load_manifest(condition/'input-manifest.json',[('cameras',condition/'condition.npz')])
    prep=json.loads((condition/'preparation.json').read_text())
    if prep['mode']!='reconstruction' or prep.get('conditioning_artifact_id','virtual_camera_requests')!='virtual_camera_requests':
        raise ValueError('Prefix requires an original full-video reconstruction condition')
    if file_sha256(condition/'condition.npz')!=prep['condition_sha256']:
        raise ValueError('Original condition changed')
    with np.load(condition/'condition.npz',allow_pickle=False) as z:
        arrays={name:z[name].copy() for name in ['rgb','K','c2w','timestamps','source_frame_indices']}
    original_times=arrays['timestamps'].copy()
    if type(frames) is not int or frames<5 or (frames-1)%4 or frames>=len(original_times):
        raise ValueError('Use a shorter existing 4k+1 prefix of at least five frames')
    for name in ['rgb','timestamps','source_frame_indices']: arrays[name]=arrays[name][:frames]
    output.mkdir(parents=True)
    np.savez_compressed(output/'condition.npz',**arrays)
    manifest=json.loads((condition/'input-manifest.json').read_text())
    for artifact in manifest['artifacts']:
        if 'path' in artifact: artifact['path']=str((condition/artifact['path']).resolve())
    manifest['artifacts'].append({'id':'prefix_camera_requests','kind':'cameras','origin':'model_prediction',
        'parent_ids':['virtual_camera_requests'],'path':'condition.npz','sha256':file_sha256(output/'condition.npz'),
        'assumption':'Same predeclared full-clip camera requests; only source conditioning time prefix is shortened'})
    manifest['method_inputs'].append('prefix_camera_requests')
    validate_manifest(manifest,output,verify_files=True)
    (output/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    prep.update(observed_source_frames=frames,condition_sha256=file_sha256(output/'condition.npz'),
        conditioning_artifact_id='prefix_camera_requests',source_timestamps=arrays['timestamps'].tolist(),
        original_condition=str(condition),original_condition_sha256=file_sha256(condition/'condition.npz'),
        conditioning_horizon='Prefix of the original sampled source sequence; not a full-clip endpoint comparison',
        original_source_time_range_s=[float(original_times[0]),float(original_times[-1])],
        conditioned_source_time_range_s=[float(arrays['timestamps'][0]),float(arrays['timestamps'][-1])],
        camera_design_horizon='Unchanged full-clip source-derived initializer; camera design may consume later source frames',
        training_caveat='17-frame prefix matches training frame count when frames=17, but source-only full-video history still omits target-view first-frame anchors used by training view_history')
    (output/'preparation.json').write_text(json.dumps(prep,indent=2))
    return prep


def checked_guidance(text_guidance, hist_guidance, sample_steps):
    for value in [text_guidance,hist_guidance]:
        if not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=5:
            raise ValueError('Guidance must be a finite declared value between zero and five')
    # The no-history branch has an all-false mask and is excluded from source-clamp checks.
    return int(sample_steps)*(1+int(bool(text_guidance)))


def validate_text_embeddings(path, receipt_path, prompt_key):
    """Accept the author's cached-embedding format only with a completed receipt."""
    import torch
    path,receipt_path=Path(path).resolve(),Path(receipt_path).resolve()
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('status')!='complete' or receipt.get('key')!=prompt_key:
        raise ValueError('Custom text embedding needs a completed matching-key receipt')
    if receipt.get('embedding_sha256')!=file_sha256(path):
        raise ValueError('Custom text embedding differs from its receipt')
    for key in ['encoder_sha256','encoder_source_sha256','tokenizer_source_sha256','tokenizer_config_sha256']:
        value=receipt.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Custom text receipt must bind the author encoder/tokenizer implementation')
    data=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(data,dict) or set(data)!={prompt_key,''}:
        raise ValueError('Custom embedding dictionary must contain the requested task and null context')
    for key,text in [(prompt_key,receipt.get('text')),('','')]:
        row=data[key]
        if not isinstance(row,dict):raise ValueError('Custom embedding entry must contain text and embed')
        tensor=row.get('embed')
        if (not isinstance(text,str) or (key and not text.strip()) or row.get('text')!=text
                or not isinstance(tensor,torch.Tensor) or tensor.ndim!=2 or not 1<=tensor.shape[0]<=512
                or tensor.shape[1]!=4096 or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())):
            raise ValueError('Custom text embedding must have matching text and finite [L,4096] tensors')
        schema=receipt.get('schema',{}).get(key,{})
        if schema.get('text')!=text or schema.get('shape')!=list(tensor.shape):
            raise ValueError('Embedding text/tensor shape differs from the author encoding receipt')
        if receipt.get('dtype')!=str(tensor.dtype).removeprefix('torch.'):
            raise ValueError('Embedding dtype differs from the author encoding receipt')
    return {'mode':'custom_task_prompt','prompt_key':prompt_key,'task_text_used':True,'text':receipt['text'],
        'embedding_path':str(path),'embedding_sha256':file_sha256(path),'receipt_path':str(receipt_path),
        'receipt_sha256':file_sha256(receipt_path),'encoder_sha256':receipt['encoder_sha256'],
        'encoder_source_sha256':receipt['encoder_source_sha256'],'tokenizer_source_sha256':receipt['tokenizer_source_sha256'],
        'tokenizer_config_sha256':receipt['tokenizer_config_sha256'],
        'interpretation':'Explicit task instruction encoded with the cached author UMT5; no observed 3D geometry'}


def configure_text_conditioning(algo, prompt_key, *, null_text, text_guidance, custom=None):
    """Guidance zero still uses task text unless null context is explicit."""
    if null_text:
        if text_guidance!=0:
            raise ValueError('Explicit null text requires --text-guidance 0')
        algo.use_text=False
        return {'mode':'null_context','prompt_key':None,'task_text_used':False}
    if prompt_key not in algo.text_embeds:
        raise ValueError('Unknown fixed task prompt; use an appropriate cached prompt or explicit null text')
    algo.use_text=True
    if custom is not None:return custom
    return {'mode':'cached_task_prompt','prompt_key':prompt_key,'task_text_used':True}


def generate(*, condition, repository, config, checkpoint, checkpoint_receipt, output,
             prompt_key='task_0004', seed=20261119, text_guidance=2., hist_guidance=1., null_text=False,
             text_embeddings=None,text_embedding_receipt=None):
    """Run the unchanged author sampler, verifying source-latent clamping."""
    import torch
    import imageio.v2 as imageio
    from omegaconf import OmegaConf
    condition, output, repository = Path(condition).resolve(),Path(output).resolve(),Path(repository).resolve()
    if output.exists(): raise FileExistsError(output)
    prep=json.loads((condition/'preparation.json').read_text())
    load_manifest(condition/'input-manifest.json',[('cameras',condition/'condition.npz')])
    if prep['mode']!='reconstruction' or file_sha256(condition/'condition.npz')!=prep['condition_sha256']:
        raise ValueError('Condition changed or unsupported forecast preparation')
    for key in ['geometry_manifest','scene_manifest']:
        if file_sha256(prep[key])!=prep[key+'_sha256']:raise ValueError('Preparation ancestry changed')
        load_manifest(prep[key])
    proof=json.loads(Path(checkpoint_receipt).read_text()); checkpoint=Path(checkpoint).resolve();config=Path(config).resolve()
    if proof['checkpoint']!=str(checkpoint) or proof['checkpoint_bytes']!=checkpoint.stat().st_size or checkpoint.stat().st_mtime_ns>Path(checkpoint_receipt).stat().st_mtime_ns:
        raise ValueError('Existing full checkpoint hash receipt no longer applies')
    if proof['config_sha256']!=file_sha256(config):raise ValueError('Model config changed')
    if file_sha256(repository/'algorithms/wan/wan_multiview.py')!='b95040653675f7517276aecbe055470f4260074562cac16a6031f61c3e9c8def':
        raise ValueError('Author sampler changed; review before running')
    if file_sha256(repository/'algorithms/wan/modules/vae.py')!='e3210afa96b708ba6eb790b3f1c6f5c92dfc11a3aa8e725a83291ac52cc6a7f9':
        raise ValueError('Author VAE changed; review before running')
    sys.path.insert(0,str(repository))
    from algorithms.wan.wan_multiview import WanMultiView
    from datasets.aria_new import camera_to_raymap
    with np.load(condition/'condition.npz',allow_pickle=False) as z:
        rgb,k,c,times,ids=[z[n].copy() for n in ['rgb','K','c2w','timestamps','source_frame_indices']]
    tensor,known=source_tensor(rgb)
    cfg=OmegaConf.load(config)
    expected_conditioned_calls=checked_guidance(text_guidance,hist_guidance,cfg.algorithm.sample_steps)
    if null_text and text_guidance!=0: raise ValueError('Explicit null text requires --text-guidance 0')
    if bool(text_embeddings)!=bool(text_embedding_receipt) or (null_text and text_embeddings):
        raise ValueError('Custom task embeddings require both file and receipt, and cannot be combined with null text')
    custom_text=validate_text_embeddings(text_embeddings,text_embedding_receipt,prompt_key) if text_embeddings else None
    if custom_text:
        cfg.algorithm.text.enabled=True
        cfg.algorithm.text.embeds_path=str(Path(text_embeddings).resolve())
    if cfg.algorithm.design!='tokens' or (cfg.dataset.height,cfg.dataset.width)!=(160,320):
        raise ValueError('This driver requires the reviewed tokens design and320x160 model')
    cfg.algorithm.model.tuned_ckpt_path=None;cfg.algorithm.gradient_checkpointing_rate=0.
    output.mkdir(parents=True)
    sampling_config=dict(schema='source-only-sampling-config-v1',text_guidance=float(text_guidance),hist_guidance=float(hist_guidance),
        seed=seed,prompt_key=None if null_text else prompt_key,text_conditioning='null_context' if null_text else 'cached_task_prompt',sample_steps=int(cfg.algorithm.sample_steps),
        author_validation_text_guidance=float(cfg.algorithm.get('val_text_guidance',0.)),
        author_validation_hist_guidance=float(cfg.algorithm.get('val_hist_guidance',0.)),
        model_config_sha256=file_sha256(config),checkpoint_sha256=proof['checkpoint_sha256'],
        driver_sha256=file_sha256(__file__),condition_sha256=file_sha256(condition/'condition.npz'),
        preparation_sha256=file_sha256(condition/'preparation.json'),conditioning_frames=len(rgb),
        source_time_range_s=[float(times[0]),float(times[-1])],target_view_observed_anchors=0,
        extra_observed_views=0,camera_design_horizon=prep.get('camera_design_horizon','Full source-clip initializer'),
        conditioning_horizon=prep.get('conditioning_horizon','Full sampled source clip'),custom_text_conditioning=custom_text)
    if custom_text:sampling_config['text_conditioning']='custom_task_prompt'
    (output/'sampling-config.json').write_text(json.dumps(sampling_config,indent=2))
    sampling_config_sha256=file_sha256(output/'sampling-config.json')
    start=time.monotonic()
    algo=WanMultiView(cfg.algorithm);algo.configure_model()
    weights=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if int(weights['global_step'])!=10000:raise ValueError('Unexpected author checkpoint step')
    state={n[6:]:v for n,v in weights['state_dict'].items() if n.startswith('model.')}
    algo.model.load_state_dict(state,strict=True);del weights,state
    algo=algo.to(dtype=torch.bfloat16).to('cuda').eval()
    text_conditioning=configure_text_conditioning(algo,prompt_key,null_text=null_text,text_guidance=text_guidance,custom=custom_text)
    K=torch.tensor(k,dtype=torch.float32);C=torch.tensor(c,dtype=torch.float32)
    scale=float(C[1:,:3,3].norm(dim=-1).mean());C[:,:3,3]/=scale
    kr=K.clone();kr[:,0]*=40/320;kr[:,1]*=20/160
    directions,moments=camera_to_raymap(kr,C,20,40)
    batch=dict(videos=torch.from_numpy(tensor),raymaps=torch.cat([directions,moments],dim=1)[None],intrinsics=K[None],c2w=C[None],prompts=[prompt_key])
    checks=[];torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        lat,cond=algo.pack(batch)
        if lat.ndim!=6 or lat.shape[1]!=4 or lat.shape[2]!=16 or lat.shape[3]!=(len(rgb)-1)//4+1:
            raise ValueError('Unexpected packed multiview VAE layout')
        hm=torch.zeros(lat.shape[:2]+lat.shape[3:4],dtype=torch.bool,device=lat.device);hm[:,0,:]=True
        expanded=hm[:,:,None,:,None,None].expand_as(lat);fixed=lat[expanded].clone()
        def check(module,args,kwargs):
            if torch.equal(kwargs['hist_mask'],hm):
                err=float((args[0][expanded]-fixed).abs().max())
                if err!=0 or not bool(torch.all(args[2][hm]==0)):raise ValueError('Observed source latent changed during sampling')
                checks.append(err)
        hook=algo.model.register_forward_pre_hook(check,with_kwargs=True)
        torch.manual_seed(seed)
        try:pred=algo.sample(lat,cond,batch=batch,hist_mask=hm,text_guidance=text_guidance,hist_guidance=hist_guidance)
        finally:hook.remove()
        final_error=float((pred[expanded]-fixed).abs().max())
        if final_error!=0 or len(checks)!=expected_conditioned_calls:raise ValueError('Source clamping verification failed')
        raw=((algo._views(pred,1)[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,0,3,4,2).cpu().numpy()
    if raw.shape!=(len(rgb),4,160,320,3):
        raise ValueError('Unexpected decoded multiview RGB layout')
    observations=raw.copy();observations[:,0]=rgb
    if custom_text and (file_sha256(text_embeddings)!=custom_text['embedding_sha256']
                        or file_sha256(text_embedding_receipt)!=custom_text['receipt_sha256']):
        raise ValueError('Custom text evidence changed during generation')
    if file_sha256(output/'sampling-config.json')!=sampling_config_sha256:
        raise ValueError('Sampling configuration changed during generation')
    np.savez_compressed(output/'generated.npz',rgb=raw,K=k,c2w=c,timestamps=times,observed_mask=known)
    np.savez_compressed(output/'observations.npz',rgb=observations,K=k,c2w=c,timestamps=times,observed_mask=known,source_frame_indices=ids)
    manifest=json.loads((condition/'input-manifest.json').read_text())
    manifest['task_text_conditioning']=text_conditioning
    for artifact in manifest['artifacts']:
        if 'path' in artifact:artifact['path']=str((condition/artifact['path']).resolve())
    manifest['artifacts'] += [
        {'id':'world_model_weights','kind':'model_prior','origin':'prior','prior_type':'learned_model_weights',
         'scope':'global','parent_ids':[],'model_name':'Author WanMultiView RGB step10000',
         'revision':proof['checkpoint_sha256']},
        {'id':'generated_helper_rgb','kind':'rgb','origin':'model_prediction','generated':True,
         'view_role':'helper','parent_ids':[prep.get('conditioning_artifact_id','virtual_camera_requests'),'world_model_weights'],
         'path':'generated.npz','sha256':file_sha256(output/'generated.npz'),
         'time_range_s':[float(times[0]),float(times[-1])]},
        {'id':'reconstruction_rgb','kind':'rgb','origin':'model_prediction','generated':True,
         'view_role':'helper','parent_ids':['generated_helper_rgb'],
         'path':'observations.npz','sha256':file_sha256(output/'observations.npz'),
         'time_range_s':[float(times[0]),float(times[-1])],
         'mixed_origin_rule':'observed_mask is true for original source view0 only; remaining pixels generated'}]
    manifest['method_inputs'] += ['reconstruction_rgb']
    validate_manifest(manifest,output,verify_files=True)
    (output/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    fps=(len(times)-1)/(times[-1]-times[0])
    with imageio.get_writer(output/'source-and-generated.mp4',fps=fps,codec='libx264',quality=8) as writer:
        for row in observations:writer.append_data(np.concatenate(row,axis=1))
    report=dict(status='complete',mode='observed-video reconstruction',seconds=time.monotonic()-start,
                checkpoint=str(checkpoint),checkpoint_sha256=proof['checkpoint_sha256'],driver_sha256=file_sha256(__file__),
                preparation_sha256=file_sha256(condition/'preparation.json'),seed=seed,prompt_key=text_conditioning['prompt_key'],
                text_conditioning=text_conditioning,
                text_guidance=float(text_guidance),hist_guidance=float(hist_guidance),sampling_config_sha256=sampling_config_sha256,
                conditioning_horizon=sampling_config['conditioning_horizon'],source_time_range_s=sampling_config['source_time_range_s'],
                camera_design_horizon=sampling_config['camera_design_horizon'],
                source_frames=len(rgb),total_views=4,generated_views=3,observed_extra_views=0,
                supplied_dataset_cameras=False,measured_depth_used=False,simulator_used=False,
                source_clamp_max_error=final_error,conditional_forward_checks=len(checks),
                generated_sha256=file_sha256(output/'generated.npz'),observations_sha256=file_sha256(output/'observations.npz'),
                virtual_camera_scale_m=scale,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9,
                source_pixels='Exact supplied RGB restored only in view0; raw decoded source retained in generated.npz',
                caveats=[prep['training_caveat'],'Generated camera consistency and object accuracy are unmeasured.'])
    (output/'generation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for name in ['geometry-bundle','scene-path','scene-manifest','output']:a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--frames',type=int,default=25)
    prefix=sub.add_parser('prepare-prefix')
    for name in ['condition','output']:prefix.add_argument('--'+name,type=Path,required=True)
    prefix.add_argument('--frames',type=int,default=17)
    g=sub.add_parser('generate')
    for name in ['condition','repository','config','checkpoint','checkpoint-receipt','output']:g.add_argument('--'+name,type=Path,required=True)
    g.add_argument('--prompt-key',default='task_0004');g.add_argument('--seed',type=int,default=20261119)
    g.add_argument('--text-guidance',type=float,default=2.);g.add_argument('--hist-guidance',type=float,default=1.)
    g.add_argument('--null-text',action='store_true',help='Use null text context rather than a cached task prompt; requires text guidance zero')
    g.add_argument('--text-embeddings',type=Path,help='Author-format task/null embedding dictionary; requires a completed receipt')
    g.add_argument('--text-embedding-receipt',type=Path)
    args=vars(p.parse_args());command=args.pop('command')
    print(json.dumps({'prepare':prepare,'prepare-prefix':prepare_prefix,'generate':generate}[command](**args),indent=2))


if __name__=='__main__':main()
