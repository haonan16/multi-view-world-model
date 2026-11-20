"""Native SAM3 video tracking of strict forecast RGB from predicted t0 masks.

Source view seeds come from the authenticated original-resolution first image;
helper seeds come from their independently generated t0 detections. No later
recorded frame, manual coordinate, geometry fit, or simulator supplies a seed.
Only object rows are replaced. Hands/table retain framewise SAM3 predictions.
"""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256,load_manifest,validate_manifest
from benchmarks.forecast_masks import load_forecast
from benchmarks.forecast_geometry import merge_inputs
from benchmarks.strict_forecast import _first_image_manifest


OBJECT_PROMPT='wooden block on patterned mat'


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name].copy() for name in archive.files}


def select_seed(masks,view,prompt,*,minimum_score=.3,duplicate_iou=.85,minimum_pixels=12):
    """Accept a unique first-frame candidate after removing duplicate masks."""
    prompts=list(masks['prompts'])
    if prompt not in prompts:
        return None,dict(status='missing_prompt',view=view,candidate_rows=[])
    shape=tuple(int(x) for x in masks['shape'])
    t,p,score=np.asarray(masks['t']),np.asarray(masks['p']),np.asarray(masks['score'])
    v=np.asarray(masks.get('v',np.zeros(len(t),np.int64)))
    rows=np.flatnonzero((t==0)&(v==view)&(p==prompts.index(prompt))&(score>=minimum_score))
    unique=[]
    for row in sorted(rows,key=lambda r:(-float(score[r]),int(r))):
        mask=np.unpackbits(masks['mask'][row])[:np.prod(shape)].reshape(shape).astype(bool)
        if mask.sum()<minimum_pixels:continue
        if any((mask&old).sum()/max(1,(mask|old).sum())>=duplicate_iou for _,old in unique):continue
        unique.append((int(row),mask))
    report=dict(view=view,candidate_rows=[row for row,_ in unique],
                status='unique' if len(unique)==1 else 'missing' if not unique else 'ambiguous')
    if len(unique)!=1:return None,report
    row,mask=unique[0]
    report.update(selected_row=row,detector_score=float(score[row]),seed_pixels=int(mask.sum()))
    return mask,report


def source_seed(mask,crop,shape):
    x,y,w,h=map(int,crop)
    clipped=mask[y:y+h,x:x+w]
    if clipped.shape!=(h,w):raise ValueError('Initial-mask crop is outside the native first image')
    return np.asarray(Image.fromarray(clipped).resize((shape[1],shape[0]),Image.Resampling.NEAREST),bool)


def associate_source_seed(native,config_path):
    """Reuse the initializer's mask-only rule on exactly its first source image.

    The saved configuration is explicit, hashed algorithm input. Neither the
    initializer's selected row nor its fitted depth/scene is read here.
    """
    from benchmarks import source_geometry_v2 as selector
    config_path=Path(config_path).resolve()
    config=json.loads(config_path.read_text());selector.validate_config(config)
    if config['object_prompt']!=OBJECT_PROMPT:
        raise ValueError('Source association config uses a different object prompt')
    times=np.asarray(native['timestamps'])
    if times.shape!=(1,) or ('v' in native and np.any(native['v']!=0)):
        raise ValueError('Source seed association requires exactly one source image')
    candidates=selector.MaskCandidates(native,1,tuple(native['shape']),times)
    selected,diagnostics=selector.associate_masks(candidates,1,config)
    audit=diagnostics[0];row=selected[0]
    scores=audit['candidates']
    report=dict(view=0,status='heuristic_selected' if row is not None else audit['reason'],
        method='source_geometry_v2.associate_masks; first-image masks only',
        heuristic_not_ground_truth=True,selected_row=row,candidate_rows=[r['row'] for r in scores],
        candidate_margin_required=float(config['candidate_margin']),
        selection_margin=(float(scores[0]['selection_score']-scores[1]['selection_score']) if len(scores)>1 else None),
        association=audit,config_path=str(config_path),config_sha256=file_sha256(config_path),
        association_code_sha256=file_sha256(selector.__file__),frames_used=[0],depth_used=False)
    if row is None:return None,report
    mask=candidates.mask(row)
    report.update(detector_score=float(candidates.score[row]),seed_pixels=int(mask.sum()))
    return mask,report


def transport_seed(source,target,seed,allowed,*,match_fn=None):
    """Estimate a background-plane correspondence, never a known camera pose."""
    import cv2
    from benchmarks.helper_camera_audit import feature_matches
    if source.shape!=target.shape or seed.shape!=source.shape[:2] or allowed.shape!=seed.shape:
        raise ValueError('Seed transport requires matching image/mask grids')
    a,b,counts=(match_fn or feature_matches)(source,target,allowed)
    report=dict(method='Mutual SIFT .75 ratio + background RANSAC3px homography',
        geometric_assumption='Estimated background-plane correspondence; object parallax and hallucination remain possible',
        ground_truth_used=False,requested_cameras_used=False,source_frame=0,target_frame=0,
        **counts,status='unsupported',thresholds=dict(min_inliers=12,min_inlier_fraction=.35,
        min_source_hull_fraction=.08,min_target_hull_fraction=.05,max_median_error_px=2.,max_p90_error_px=3.5,
        min_local_inliers=3,local_radius_px=40.,max_centroid_extrapolation_px=8.,min_pixels=12,min_area_ratio=.25,max_area_ratio=4.))
    if len(a)<12 or len(a)/max(1,counts['mutual_matches'])<.35:
        report['reason']='insufficient_mutual_background_inliers';return None,report
    matrix,_=cv2.findHomography(a,b,0)
    if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2,2])<1e-9:
        report['reason']='invalid_homography';return None,report
    matrix=matrix/matrix[2,2]
    projected=cv2.perspectiveTransform(a.astype(np.float64)[None],matrix)[0]
    errors=np.linalg.norm(projected-b,axis=1)
    source_hull=cv2.convexHull(a.astype(np.float32));target_hull=cv2.convexHull(b.astype(np.float32))
    centroid=np.argwhere(seed).mean(0)[::-1] if seed.any() else np.array([np.nan,np.nan])
    source_area=float(cv2.contourArea(source_hull)/seed.size);target_area=float(cv2.contourArea(target_hull)/seed.size)
    local=int(np.count_nonzero(np.linalg.norm(a-centroid,axis=1)<=40))
    hull_distance=float(cv2.pointPolygonTest(source_hull,tuple(centroid.astype(float)),True)) if np.isfinite(centroid).all() else -1e9
    report.update(homography=matrix.tolist(),median_error_px=float(np.median(errors)),p90_error_px=float(np.quantile(errors,.9)),
        source_hull_fraction=source_area,target_hull_fraction=target_area,local_inliers=local,centroid_inside_hull_distance_px=hull_distance)
    if (source_area<.08 or target_area<.05 or np.median(errors)>2 or np.quantile(errors,.9)>3.5
            or local<3 or hull_distance < -8):
        report['reason']='poor_reprojection_or_spatial_support';return None,report
    yy,xx=np.nonzero(seed);xy=np.c_[xx,yy,np.ones(len(xx))]@matrix.T
    if not len(xy) or np.any(xy[:,2]<=1e-8):
        report['reason']='invalid_seed_projection';return None,report
    mapped=cv2.warpPerspective(seed.astype(np.uint8),matrix,(seed.shape[1],seed.shape[0]),flags=cv2.INTER_NEAREST)>0
    ratio=float(mapped.sum()/max(1,seed.sum()))
    report.update(projected_pixels=int(mapped.sum()),area_ratio=ratio)
    if mapped.sum()<12 or not .25<=ratio<=4:
        report['reason']='projected_seed_missing_or_implausible_area';return None,report
    report.update(status='transported_hypothesis',reason='background_correspondence_supported',
                  centroid_xy=np.argwhere(mapped).mean(0)[::-1].tolist(),seed_pixels=int(mapped.sum()))
    return mapped,report


def source_background(native,crop,shape,seed):
    import cv2
    prompts=list(native['prompts'])
    if 'hand' not in prompts or OBJECT_PROMPT not in prompts:
        raise ValueError('Source transport needs predicted hand and object categories for background exclusion')
    exclude=np.zeros(tuple(native['shape']),bool)
    rows=np.flatnonzero((native['t']==0)&np.isin(native['p'],[prompts.index('hand'),prompts.index(OBJECT_PROMPT)])&(native['score']>=.3))
    for row in rows:exclude|=np.unpackbits(native['mask'][row])[:exclude.size].reshape(exclude.shape).astype(bool)
    exclude=source_seed(exclude,crop,shape)|seed
    return ~(cv2.dilate(exclude.astype(np.uint8),np.ones((9,9),np.uint8))>0)


def validate_mask_packet(data,job,*,source_only):
    views=1 if source_only else job['rgb'].shape[1]
    if (tuple(data['shape'])!=tuple(job['rgb'].shape[2:4])
            or not np.array_equal(data['timestamps'],job['timestamps'])
            or not np.array_equal(data['observed_mask'],job['observed_mask'][:,:views])
            or not np.array_equal(data['source_frame_indices'],job['ids'])):
        raise ValueError('Baseline masks must match exact strict forecast pixels/times/view roles')
    if np.any((data['v']<0)|(data['v']>=views)):
        raise ValueError('Mask view indices differ from their product')
    import hashlib
    expected=np.asarray([[hashlib.sha256(job['rgb'][t,v].tobytes()).hexdigest()
                          for v in range(views)] for t in range(len(job['timestamps']))])
    if not np.array_equal(data['image_rgb_sha256'],expected):
        raise ValueError('Framewise masks were computed on different generated RGB images')


def load_inputs(helper,condition,baseline_masks,initial_masks,source_association_config=None,helper_seed_transport=False):
    job=load_forecast(helper,condition)
    baseline_masks,initial_masks=Path(baseline_masks).resolve(),Path(initial_masks).resolve()
    before={}
    for folder in [baseline_masks,baseline_masks/'source-only',initial_masks]:
        load_manifest(folder/'input-manifest.json',[('masks',folder/'masks.npz')])
        for name in ['input-manifest.json','masks.npz']:
            path=folder/name;before[str(path)]=file_sha256(path)
    initial,_,root,_,t0=_first_image_manifest(initial_masks/'input-manifest.json')
    if root['sha256']!=job['observed']['sha256'] or t0!=job['manifest']['observed_until_s']:
        raise ValueError('Initial tracker seed must share the same sole observed image')
    baseline=arrays(baseline_masks/'masks.npz');source=arrays(baseline_masks/'source-only/masks.npz')
    validate_mask_packet(baseline,job,source_only=False);validate_mask_packet(source,job,source_only=True)
    source_rows=baseline['v']==0
    for name in ['mask','score','box','t','v','p']:
        if not np.array_equal(source[name],baseline[name][source_rows]):
            raise ValueError('Source-only baseline must be the exact independent view-0 candidate subset: '+name)
    # Both products must be independently attributable; source-only cannot
    # inherit helpers merely because the same GPU job also computed them.
    manifests={}
    for name,folder in [('all',baseline_masks),('source',baseline_masks/'source-only')]:
        manifest=merge_inputs([folder/'input-manifest.json',initial_masks/'input-manifest.json'])
        if name=='source':
            # The independently computed framewise source subset originally
            # shares its ID with the all-view product. Give this terminal
            # evidence a distinct name before later helper+source fusion.
            ids={a['id']:'forecast_tracker_source_framewise_masks' for a in manifest['artifacts']
                 if a['kind']=='masks' and a.get('path')==str(folder/'masks.npz')}
            if len(ids)!=1:raise ValueError('Expected one bound source-only framewise mask product')
            if any(a['id'] in ids.values() for a in manifest['artifacts']):raise ValueError('Source mask alias already exists')
            for a in manifest['artifacts']:
                a['id']=ids.get(a['id'],a['id']);a['parent_ids']=[ids.get(p,p) for p in a['parent_ids']]
            manifest['method_inputs']=[ids.get(p,p) for p in manifest['method_inputs']]
        report=validate_manifest(manifest,verify_files=True)
        if name=='source' and any(a.get('view_role')=='helper' for a in manifest['artifacts']
                                 if a['id'] in report['method_artifact_ids']):
            raise ValueError('Source-only tracker input contains generated-helper ancestry')
        manifests[name]=manifest
    prep=json.loads((Path(condition)/'preparation.json').read_text())
    native=arrays(initial_masks/'masks.npz')
    if tuple(native['shape'])!=tuple(_first_image_manifest(initial_masks/'input-manifest.json')[3].shape[1:3]):
        raise ValueError('Native initial mask grid differs from the observed image')
    if not np.array_equal(native['timestamps'],[t0]):raise ValueError('Native seed masks contain later times')
    if source_association_config is not None:
        source_association_config=Path(source_association_config).resolve()
        before[str(source_association_config)]=file_sha256(source_association_config)
    seeds=[];reports=[]
    for view in range(job['rgb'].shape[1]):
        if view==0 and source_association_config is not None:
            mask,report=associate_source_seed(native,source_association_config)
        elif view>0 and helper_seed_transport:
            if seeds[0] is None:
                mask,report=None,dict(status='unsupported',reason='source_seed_unavailable',view=view)
            else:
                allowed=source_background(native,prep['source_crop_xywh'],job['rgb'].shape[2:4],seeds[0])
                mask,report=transport_seed(job['rgb'][0,0],job['rgb'][0,view],seeds[0],allowed)
        else:
            mask,report=select_seed(native if view==0 else baseline,0 if view==0 else view,OBJECT_PROMPT)
        report.update(view=view,source='native_first_observed_image' if view==0 else
            'source_mask_transported_to_generated_helper_t0' if helper_seed_transport else 'generated_helper_first_frame')
        if view==0 and mask is not None:
            mask=source_seed(mask,prep['source_crop_xywh'],job['rgb'].shape[2:4])
            report.update(crop_xywh=prep['source_crop_xywh'],seed_pixels_after_crop=int(mask.sum()))
            if mask.sum()<12:mask=None;report['status']='insufficient_seed_after_crop'
        seeds.append(mask);reports.append(report)
    return dict(job=job,baseline=baseline,source=source,manifests=manifests,seeds=seeds,
                seed_reports=reports,input_hashes=before,baseline_masks=baseline_masks,initial_masks=initial_masks)


def candidate(logits,object_logit,*,presence_threshold=.5,minimum_pixels=12):
    logits=np.asarray(logits,np.float32)
    if logits.ndim!=2 or not np.isfinite(logits).all() or not np.isfinite(object_logit):
        raise ValueError('Tracker returned nonfinite mask or object-presence logits')
    presence=float(expit(object_logit));mask=logits>0
    present=presence>presence_threshold and mask.sum()>=minimum_pixels
    return mask,presence,bool(present)


def combine(baseline,tracked,*,source_only=False):
    """Replace object detections only; retain untouched hand/table detections."""
    prompts=list(baseline['prompts']);object_id=prompts.index(OBJECT_PROMPT)
    keep=baseline['p']!=object_id
    names=['mask','score','box','t','v','p']
    result={name:baseline[name][keep].copy() for name in names}
    scores_kind=[0]*int(keep.sum());track_ids=[-1]*int(keep.sum())
    raw_logit=[float('nan')]*int(keep.sum())
    rows=[r for r in tracked if r['present'] and (not source_only or r['view']==0)]
    if rows:
        additions={name:[] for name in names}
        for row in rows:
            mask=row['mask'];yy,xx=np.nonzero(mask)
            entries=dict(mask=np.packbits(mask.reshape(-1)),score=row['presence'],
                box=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
                t=row['frame'],v=row['view'],p=object_id)
            for name in names:additions[name].append(entries[name])
            scores_kind.append(1);track_ids.append(1);raw_logit.append(row['object_logit'])
        for name in names:
            extra=np.asarray(additions[name],dtype=result[name].dtype)
            result[name]=np.concatenate([result[name],extra],axis=0)
    for name in ['prompts','shape','timestamps','observed_mask','source_frame_indices','image_rgb_sha256']:
        result[name]=baseline[name].copy()
    result.update(candidate_row_id=np.arange(len(result['t']),dtype=np.int64),
        score_kind=np.asarray(scores_kind,np.int8),track_id=np.asarray(track_ids,np.int64),
        object_score_logit=np.asarray(raw_logit,np.float32))
    return result


def write_product(output,data,manifest,checkpoint,raw_path,*,source_only,helper_seed_transport=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'masks.npz',**data)
    manifest=deepcopy(manifest);parents=manifest['method_inputs'].copy()
    weights='forecast_video_tracker_weights'
    masks='forecast_source_video_tracker_masks' if source_only else 'forecast_video_tracker_masks'
    raw='forecast_source_video_tracker_logits' if source_only else 'forecast_video_tracker_logits'
    if any(a['id'] in {weights,masks,raw} for a in manifest['artifacts']):
        raise ValueError('Tracker artifact IDs already exist')
    manifest['artifacts'] += [
        dict(id=weights,kind='model_prior',origin='prior',scope='global',prior_type='learned_model_weights',parent_ids=[],
             model_name='facebook/sam3 Sam3TrackerVideoModel',revision=checkpoint.name,config_sha256=file_sha256(checkpoint/'config.json')),
        dict(id=raw,kind='masks',origin='model_prediction',parent_ids=parents+[weights],path=str(raw_path),sha256=file_sha256(raw_path),
             description='Native tracker mask/object-presence logits; unavailable views/frames remain NaN'),
        dict(id=masks,kind='masks',origin='model_prediction',parent_ids=[raw]+parents,path='masks.npz',sha256=file_sha256(output/'masks.npz'),
             view_indices=[0] if source_only else [0,1,2,3],temporal_object_identity_estimated=True,
             cross_view_identity_estimated=bool(helper_seed_transport and not source_only),hand_table_tracking=False,
             cross_view_identity_interpretation='Optional background-plane source-mask transport is a hypothesis, not verified semantic identity',
             score_interpretation='score_kind0: retained framewise detector score; score_kind1: sigmoid native tracker object-presence logit, not IoU/confidence calibration')]
    manifest['method_inputs']=[masks]
    report=validate_manifest(manifest,output,verify_files=True)
    write_json(output/'input-manifest.json',manifest)
    return dict(mask_artifact_id=masks,masks_sha256=file_sha256(output/'masks.npz'),validation=report)


def run(*,helper,condition,baseline_masks,initial_masks,output,checkpoint,track,
        presence_threshold=.5,minimum_pixels=12,model_load_seconds=None,source_association_config=None,helper_seed_transport=False):
    output,checkpoint=Path(output).resolve(),Path(checkpoint).resolve()
    if output.exists():raise FileExistsError(output)
    if not np.isfinite(presence_threshold) or not 0<presence_threshold<1 or type(minimum_pixels)is not int or minimum_pixels<1:
        raise ValueError('Declare valid fixed visibility/area thresholds')
    inputs=load_inputs(helper,condition,baseline_masks,initial_masks,source_association_config,helper_seed_transport);job=inputs['job']
    rgb=job['rgb'];frames,views,h,w=rgb.shape[:4]
    output.mkdir(parents=True);started=time.monotonic()
    receipt=dict(status='running',tracker='native HF Sam3TrackerVideoModel',first_frame_seed_only=True,forward_only=True,
        tracker_executed=any(seed is not None for seed in inputs['seeds']),
        seed_policy=('Explicit source_geometry_v2 first-image mask-only association for source. Missing/ambiguous seeds remain unsupported'
                     if source_association_config is not None else
                     'Unique predicted first-image source object mask; missing or ambiguous seed remains unsupported'),
        seeds=inputs['seed_reports'],presence_threshold=presence_threshold,mask_logit_threshold=0.,minimum_pixels=minimum_pixels,
        initial_seed_is_ground_truth=False,GT_future_used=False,manual_prompts_used=False,simulator_used=False,
        helper_seed_policy='Background-plane source-mask transport; quality failures unsupported' if helper_seed_transport else 'unique_generated_t0_candidate',
        initial_support_or_depth_used_by_tracker=False,model_load_seconds=model_load_seconds,
        input_files=inputs['input_hashes'],condition_files=job['input_files'],producer_sha256=file_sha256(__file__),
        frames=frames,views=views,cross_view_identity_estimated=bool(helper_seed_transport),
        score_interpretation='Native object-presence logit; sigmoid is not a calibrated correctness probability. Seed logits are conditioned on supplied predicted mask.')
    raw=np.full((frames,views,h,w),np.nan,np.float32);object_logits=np.full((frames,views),np.nan,np.float32)
    processed=np.zeros((frames,views),bool);rows=[];frame_reports=[]
    try:
        for view,seed in enumerate(inputs['seeds']):
            if seed is None:continue
            seen=set()
            for result in track(rgb[:,view].copy(),seed.copy()):
                frame=result['frame_index']
                if type(frame)is not int or frame<0 or frame>=frames or frame in seen:
                    raise ValueError('Tracker produced duplicate or out-of-range frame index')
                seen.add(frame)
                logits=np.asarray(result['mask_logits'],np.float32);logit=float(result['object_score_logit'])
                if logits.shape!=(h,w):raise ValueError('Tracker logits were not resized to the forecast pixel grid')
                mask,presence,present=candidate(logits,logit,presence_threshold=presence_threshold,minimum_pixels=minimum_pixels)
                raw[frame,view]=logits;object_logits[frame,view]=logit;processed[frame,view]=True
                rows.append(dict(frame=frame,view=view,mask=mask,presence=presence,present=present,object_logit=logit))
                frame_reports.append(dict(frame=frame,view=view,object_score_logit=logit,presence_score=presence,
                    pixels=int(mask.sum()),retained=present,visibility='predicted_present' if present else 'absent_occluded_or_too_small'))
                write_json(output/'progress.json',dict(status='running',processed_frames=int(processed.sum()),total_requested_frames=frames*sum(s is not None for s in inputs['seeds']),last=frame_reports[-1]))
                print(f'SAM3 tracker view{view} frame{frame}: pixels={int(mask.sum())} presence={presence:.4f} kept={present}',flush=True)
            if seen!=set(range(frames)):raise ValueError('Native tracker stopped before processing all seeded-view frames')
        # No stored old/future observation is promoted after tracking finishes.
        final=load_inputs(helper,condition,baseline_masks,initial_masks,source_association_config,helper_seed_transport)
        if final['input_hashes']!=inputs['input_hashes'] or final['job']['input_files']!=job['input_files']:
            raise ValueError('Tracker inputs changed during inference')
        source_output=output/'source-only';source_output.mkdir()
        for folder,sl in [(output,slice(None)),(source_output,slice(0,1))]:
            np.savez_compressed(folder/'tracker-logits.npz',mask_logits=raw[:,sl],object_score_logits=object_logits[:,sl],
                processed=processed[:,sl],timestamps=job['timestamps'],observed_mask=job['observed_mask'][:,sl],source_frame_indices=job['ids'])
        all_product=write_product(output,combine(inputs['baseline'],rows),inputs['manifests']['all'],checkpoint,output/'tracker-logits.npz',source_only=False,helper_seed_transport=helper_seed_transport)
        source_product=write_product(source_output,combine(inputs['source'],rows,source_only=True),inputs['manifests']['source'],checkpoint,source_output/'tracker-logits.npz',source_only=True)
        write_json(source_output/'execution.json',dict(status='complete',mode='independent_source_view_video_track',
            helper_pixels_used=False,frames=frames,views=1,seed=inputs['seed_reports'][0],**source_product))
        receipt.update(status='complete',processed_frames=int(processed.sum()),seeded_views=[v for v,s in enumerate(inputs['seeds']) if s is not None],
            unsupported_views=[v for v,s in enumerate(inputs['seeds']) if s is None],frame_reports=frame_reports,
            source_only_product=source_product,**all_product,
            completion_meaning='All seeded views propagated by native video memory; absent/occluded masks excluded, not filled; correctness is not established')
        write_json(output/'progress.json',dict(status='complete',processed_frames=int(processed.sum()),
            total_requested_frames=frames*sum(s is not None for s in inputs['seeds']),
            seeded_views=receipt['seeded_views'],unsupported_views=receipt['unsupported_views']))
    except Exception as error:
        receipt.update(status='failed',error=repr(error),processed_frames=int(processed.sum()));raise
    finally:
        receipt['seconds']=time.monotonic()-started;write_json(output/'execution.json',receipt)
    return receipt


def native_track(model,processor,device='cuda'):
    import torch
    def track(video,seed):
        session=processor.init_video_session(video=list(video),inference_device=device,dtype=torch.bfloat16)
        processor.add_inputs_to_inference_session(inference_session=session,frame_idx=0,obj_ids=[1],input_masks=seed.astype(np.float32))
        with torch.inference_mode():
            for result in model.propagate_in_video_iterator(session,start_frame_idx=0,reverse=False):
                if result.object_ids!=[1] or result.object_score_logits is None:
                    raise ValueError('Native tracker output has unexpected identity or missing visibility logit')
                logits=processor.post_process_masks([result.pred_masks],original_sizes=[[video.shape[1],video.shape[2]]],binarize=False)[0]
                value=logits.detach().float().cpu().numpy()
                while value.ndim>2 and value.shape[0]==1:value=value[0]
                scores=result.object_score_logits.detach().float().cpu().numpy().reshape(-1)
                if len(scores)!=1:raise ValueError('Native tracker returned multiple presence scores')
                yield dict(frame_index=int(result.frame_idx),mask_logits=value,object_score_logit=float(scores[0]))
    return track


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['helper','condition','baseline-masks','initial-masks','output','checkpoint']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--presence-threshold',type=float,default=.5);parser.add_argument('--minimum-pixels',type=int,default=12)
    parser.add_argument('--source-association-config',type=Path,
        help='Explicit saved source_geometry_v2 initializer config; recompute mask-only source t0 association instead of requiring a single candidate')
    parser.add_argument('--helper-seed-transport',action='store_true',
        help='Estimate t0 background homography to transport the source target mask into each helper; never use requested cameras')
    parser.add_argument('--device',default='cuda');parser.add_argument('--preflight',action='store_true')
    args=vars(parser.parse_args());preflight=args.pop('preflight');device=args.pop('device')
    inputs=load_inputs(args['helper'],args['condition'],args['baseline_masks'],args['initial_masks'],args['source_association_config'],args['helper_seed_transport'])
    checkpoint=args['checkpoint'].resolve()
    for name in ['config.json','model.safetensors']:
        if not (checkpoint/name).is_file() or not (checkpoint/name).stat().st_size:raise ValueError('Missing cached tracker file: '+name)
    if args['output'].exists():raise FileExistsError(args['output'])
    if preflight:
        print(json.dumps(dict(status='inputs_ready',seeds=inputs['seed_reports'],shape=list(inputs['job']['rgb'].shape)),indent=2));return
    if not any(seed is not None for seed in inputs['seeds']):
        result=run(**args,track=lambda video,seed:iter(()),model_load_seconds=0.)
        print(json.dumps(dict(status=result['status'],tracker_executed=False,unsupported_views=result['unsupported_views']),indent=2));return
    import torch
    from transformers import Sam3TrackerVideoModel,Sam3TrackerVideoProcessor
    start=time.monotonic()
    model=Sam3TrackerVideoModel.from_pretrained(str(checkpoint),dtype=torch.bfloat16,local_files_only=True).to(device).eval()
    processor=Sam3TrackerVideoProcessor.from_pretrained(str(checkpoint),local_files_only=True)
    result=run(**args,track=native_track(model,processor,device),model_load_seconds=time.monotonic()-start)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],seeded_views=result['seeded_views'],
        transformers=importlib.metadata.version('transformers')),indent=2))


if __name__=='__main__':main()
