"""Shared cuboid/SE(3) baseline from strict forecast RGB, DA3 and SAM3.

No future recorded image, GT geometry, contact label or simulator is consumed.
Requested-camera depth inherits the first-image model scale. Independently
unposed DA3 groups instead need a first-image depth-ratio anchor and estimated
per-time scales constrained by one shared shape; those scales are not observed.
This numerical initializer is an honest baseline for subsequent agent editing.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from artirecon.input_protocol import file_sha256, load_manifest, validate_manifest
from artirecon.scene_sequence import load_scene, object_from_arrays, scene_from_arrays, validate_scene
from da3_helper_views import read_inputs
from representation_agent_tools import agreement, geometry_arrays, raster_depth


def default_config():
    return dict(schema_version=1, object_id='block', object_prompt='wooden block on patterned mat',
        hand_prompt='hand', minimum_score=.3, duplicate_iou=.85, minimum_mask_pixels=12,
        minimum_object_points=12, points_per_view=256, boundary_samples=96, surface_grid=5,
        depth_sigma_m=.012, silhouette_sigma_px=3., shape_log_sigma=.35,
        scale_log_sigma=1.2, scale_ratio_bounds=[.1,10.], shape_ratio_bounds=[.35,3.],
        maximum_translation_m=1.5, pose_translation_prior_sigma_m=.5,
        pose_rotation_prior_sigma_rad=2., nonpenetration_weight=0., nonpenetration_sigma_m=.005,
        max_nfev=70, random_seed=20261120)


def default_initial_rest_prior():
    """Optional block only: legacy default_config bytes/keys remain unchanged."""
    return dict(enabled=False,minimum_observations=2,minimum_score=.5,
        maximum_centroid_shift_object_diagonals=.1,maximum_area_change_fraction=.15,
        minimum_mask_iou=.7,minimum_hand_gap_object_diagonals=.75,
        maximum_hand_gap_drop_object_diagonals=.25,maximum_interval_s=.5)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def _sample(values, count):
    if len(values) <= count:
        return values
    return values[np.unique(np.rint(np.linspace(0,len(values)-1,count)).astype(int))]


def _decode(data, row, shape):
    return np.unpackbits(data['mask'][row])[:np.prod(shape)].reshape(shape).astype(bool)


def associate_masks(mask_data, shape, timestamps, views, config):
    """Independent unique detections; duplicate masks are not distinct identities.

    We do not select an ambiguous block using an expected trajectory or a depth
    fit. Missing/ambiguous frames remain unsupported. No mask is carried forward.
    """
    if tuple(mask_data['shape']) != tuple(shape) or not np.array_equal(mask_data['timestamps'],timestamps):
        raise ValueError('Forecast mask grid/timestamps differ from DA3 pixels')
    prompts=list(mask_data['prompts'])
    if config['object_prompt'] not in prompts:
        raise ValueError('Object prompt absent from generated-image masks')
    object_p=prompts.index(config['object_prompt'])
    hand_p=prompts.index(config['hand_prompt']) if config['hand_prompt'] in prompts else None
    result=[]
    for t in range(len(timestamps)):
        per_view=[]
        for v in range(views):
            valid=(mask_data['t']==t)&(mask_data['v']==v)&(mask_data['score']>=config['minimum_score'])
            rows=np.flatnonzero(valid&(mask_data['p']==object_p))
            unique=[]
            for row in sorted(rows,key=lambda r:(-float(mask_data['score'][r]),int(r))):
                mask=_decode(mask_data,row,shape)
                if mask.sum()<config['minimum_mask_pixels']:
                    continue
                if all((mask&old).sum()/max(1,(mask|old).sum()) < config['duplicate_iou'] for _,old in unique):
                    unique.append((int(row),mask))
            hand=np.zeros(shape,bool);hand_rows=[]
            if hand_p is not None:
                for row in np.flatnonzero(valid&(mask_data['p']==hand_p)):
                    hand_candidate=_decode(mask_data,row,shape)
                    hand|=hand_candidate
                    if hand_candidate.sum()>=config['minimum_mask_pixels']:
                        hand_rows.append(int(row))
            chosen=unique[0] if len(unique)==1 else None
            per_view.append(dict(object_mask=chosen[1] if chosen else np.zeros(shape,bool),
                hand_mask=hand, known=chosen is not None, selected_row=chosen[0] if chosen else None,
                object_score=float(mask_data['score'][chosen[0]]) if chosen else None,
                hand_rows=hand_rows,hand_scores=[float(mask_data['score'][row]) for row in hand_rows],
                candidate_rows=[row for row,_ in unique],
                status='unique_candidate' if chosen else 'missing' if not unique else 'ambiguous_candidates'))
        result.append(per_view)
    return result


def lift(depth,k,c2w,mask,maximum):
    y,x=np.nonzero(mask&np.isfinite(depth)&(depth>0))
    pixels=_sample(np.column_stack([x,y]),maximum)
    if not len(pixels):
        return np.empty((0,3))
    x,y=pixels.T
    camera=(np.column_stack([x,y,np.ones(len(x))])@np.linalg.inv(k).T)*depth[y,x,None]
    return camera@c2w[:3,:3].T+c2w[:3,3]


def prepare_evidence(predictions, masks, config):
    rgb,depth=np.asarray(predictions['rgb']),np.asarray(predictions['depth'],float)
    times=np.asarray(predictions['timestamps'],float)
    if depth.shape!=rgb.shape[:-1] or rgb.ndim!=5 or rgb.dtype!=np.uint8:
        raise ValueError('DA3 must contain RGB/depth [T,V,H,W,(3)] on one image grid')
    count,views,h,w=depth.shape
    k,c=np.asarray(predictions['K'],float),np.asarray(predictions['c2w'],float)
    if k.shape!=(count,views,3,3) or c.shape!=(count,views,4,4):
        raise ValueError('DA3 requires explicit per-time/view K and c2w')
    if not np.isfinite(k).all() or not np.isfinite(c).all() or not np.allclose(c[:,0],np.eye(4),atol=1e-5):
        raise ValueError('DA3 groups must use the fixed source-camera gauge')
    associations=associate_masks(masks,(h,w),times,views,config)
    frames=[]
    for t in range(count):
        rows=[]
        for v in range(views):
            entry=associations[t][v]
            visible=entry['object_mask']&~entry['hand_mask']
            interior=binary_erosion(visible,iterations=1)
            if interior.sum()<config['minimum_object_points']:
                interior=visible
            points=lift(depth[t,v],k[t,v],c[t,v],interior,config['points_per_view'])
            hand_points=lift(depth[t,v],k[t,v],c[t,v],entry['hand_mask'],config['points_per_view'])
            boundary=visible&~binary_erosion(visible)
            boundary&=~binary_dilation(entry['hand_mask'],iterations=2)
            yy,xx=np.nonzero(boundary)
            boundary_uv=_sample(np.column_stack([xx,yy]),config['boundary_samples'])
            rows.append(dict(**entry,points=points,hand_points=hand_points,boundary_uv=boundary_uv,
                outside_distance=distance_transform_edt(~entry['object_mask']),
                K=k[t,v],c2w=c[t,v],shape=(h,w),source_view=v))
        frames.append(rows)
    return frames


def initial_rest_regime(frames,timestamps,config):
    """Select one contiguous RGB-only stationary prefix, never re-entering it.

    This is a fixed-source-camera model hypothesis. It uses no depth, fitted
    position, contact outcome or simulator feedback. A missing hand is unknown,
    not proof that the object is untouched. Multiple detected hands must all be
    far because the union-mask distance is the minimum over those hands.
    """
    policy=deepcopy(config.get('initial_rest_prior',default_initial_rest_prior()))
    expected=default_initial_rest_prior()
    if set(policy)!=set(expected) or type(policy['enabled']) is not bool:
        raise ValueError('Use the complete saved initial_rest_prior configuration')
    if type(policy['minimum_observations']) is not int or policy['minimum_observations']<2:
        raise ValueError('Initial rest needs at least two contiguous observations')
    for key in set(policy)-{'enabled','minimum_observations'}:
        if not isinstance(policy[key],(int,float)) or isinstance(policy[key],bool) or not np.isfinite(policy[key]) or policy[key]<=0:
            raise ValueError('Invalid initial-rest threshold: '+key)
    if max(policy['minimum_score'],policy['minimum_mask_iou'])>1:
        raise ValueError('Initial-rest score/IoU thresholds cannot exceed one')
    anchored=np.zeros(len(frames),bool)
    report=dict(enabled=policy['enabled'],status='disabled',anchored_frame_indices=[],
        source_view=0,method='contiguous stationary source-mask + detected distant-hand prefix',
        assumption='Fixed source camera; stationary image evidence supports an initial-rest hypothesis, not contact truth',
        depth_or_fitted_poses_used=False,intrinsics_modified=False,postfit_pose_snap=False,
        policy=policy,evidence=[],stop_frame=None,stop_reason=None)
    if not policy['enabled']:return anchored,report
    initial_mask=None;initial_centre=None;initial_area=None;diagonal=None;initial_gap=None;previous_gap=None
    for t,frame in enumerate(frames):
        view=frame[0];row=dict(frame=t,timestamp_s=float(timestamps[t]),status='eligible')
        reason=None
        if t and timestamps[t]-timestamps[t-1]>policy['maximum_interval_s']+1e-9:
            reason='temporal_gap'
        elif not view['known'] or not view['object_mask'].any():
            reason='missing_or_ambiguous_object'
        elif view.get('object_score') is None or view['object_score']<policy['minimum_score']:
            reason='object_score_below_threshold'
        elif not view['hand_mask'].any() or not view.get('hand_scores'):
            reason='missing_hand_is_unknown'
        elif min(view['hand_scores'])<policy['minimum_score']:
            reason='hand_score_below_threshold'
        if reason is None:
            mask=view['object_mask'];xy=np.argwhere(mask)[:,::-1];centre=xy.mean(0);area=int(mask.sum())
            gap=float(distance_transform_edt(~mask)[view['hand_mask']].min())
            if initial_mask is None:
                initial_mask=mask.copy();initial_centre=centre;initial_area=area
                diagonal=max(1.,float(np.linalg.norm(np.ptp(xy,axis=0))));initial_gap=gap;previous_gap=gap
            shift=float(np.linalg.norm(centre-initial_centre)/diagonal)
            area_change=abs(area/initial_area-1.)
            iou=float((mask&initial_mask).sum()/max(1,(mask|initial_mask).sum()))
            gap_drop=(max(initial_gap,previous_gap)-gap)/diagonal
            row.update(centroid_shift_object_diagonals=shift,area_change_fraction=float(area_change),
                mask_iou_to_initial=iou,hand_gap_object_diagonals=gap/diagonal,
                hand_gap_drop_object_diagonals=float(gap_drop),object_score=view['object_score'],
                selected_object_row=view.get('selected_row'),hand_rows=view.get('hand_rows',[]))
            if gap/diagonal<policy['minimum_hand_gap_object_diagonals']:reason='hand_near_object'
            elif t and gap_drop>policy['maximum_hand_gap_drop_object_diagonals']:reason='hand_approaching'
            elif shift>policy['maximum_centroid_shift_object_diagonals']:reason='object_image_motion'
            elif area_change>policy['maximum_area_change_fraction']:reason='object_area_changed'
            elif iou<policy['minimum_mask_iou']:reason='object_silhouette_changed'
            previous_gap=gap
        if reason is not None:
            row['status']='prefix_stopped';row['reason']=reason;report['evidence'].append(row)
            report.update(stop_frame=t,stop_reason=reason)
            break
        anchored[t]=True;report['evidence'].append(row)
    if anchored.sum()<policy['minimum_observations']:
        anchored[:]=False;report['status']='insufficient_contiguous_evidence'
    else:report['status']='initial_rest_hypothesis_selected'
    report['anchored_frame_indices']=np.flatnonzero(anchored).tolist()
    return anchored,report


def initial_scale_anchor(initial_depth, crop, da3_depth, object_mask):
    """Only first observed source pixels establish the unposed depth-ratio anchor."""
    from PIL import Image
    x,y,w,h=map(int,crop)
    native=np.asarray(initial_depth,float)[y:y+h,x:x+w]
    if native.shape!=(h,w):
        raise ValueError('Declared first-image crop is outside predicted depth')
    aligned=np.asarray(Image.fromarray(native.astype(np.float32)).resize(
        (da3_depth.shape[1],da3_depth.shape[0]),Image.Resampling.BILINEAR))
    valid=np.isfinite(aligned)&(aligned>0)&np.isfinite(da3_depth)&(da3_depth>0)
    foreground=valid&object_mask
    selection=foreground if foreground.sum()>=12 else valid
    if selection.sum()<12:
        raise ValueError('Insufficient first-image predicted depth to anchor DA3 arbitrary scale')
    ratios=aligned[selection]/da3_depth[selection]
    scale=float(np.median(ratios))
    return scale,dict(method='median first-observed-image UniDepth/DA3 depth ratio',
        pixels=int(selection.sum()),region='object_mask' if foreground.sum()>=12 else 'valid_image',
        factor=scale,ratio_q25=float(np.quantile(ratios,.25)),ratio_q75=float(np.quantile(ratios,.75)),
        measured_metric_depth_used=False,interpretation='Alignment of two predictions, not a metric measurement')


def box_surface_samples(grid):
    line=np.linspace(-1,1,grid)
    a,b=np.meshgrid(line,line)
    points=[]
    for axis in range(3):
        for sign in [-1,1]:
            face=np.zeros((grid*grid,3));face[:,axis]=sign
            face[:,[j for j in range(3) if j!=axis]]=np.column_stack([a.ravel(),b.ravel()])
            points.append(face)
    return np.concatenate(points)


def box_sdf(points,half,rotation,translation):
    local=(points-translation)@rotation
    d=np.abs(local)-half
    return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)


def plane_clearance(half,poses,support):
    if support is None:
        return None
    n=np.asarray(support['normal_world'],float)
    projection=np.einsum('i,tij->tj',n,poses[:,:3,:3])
    return poses[:,:3,3]@n+float(support['offset_m'])-np.abs(projection)@half


def _outside_hull_distance(query,projected_corners):
    if not len(query):
        return np.empty(0)
    try:
        hull=ConvexHull(projected_corners)
        polygon=projected_corners[hull.vertices]
        inside=np.all(query@hull.equations[:,:2].T+hull.equations[:,2]<=1e-6,axis=1)
        a=polygon;edge=np.roll(polygon,-1,axis=0)-a
        delta=query[:,None]-a[None]
        u=np.sum(delta*edge[None],axis=-1)/np.maximum(np.sum(edge*edge,axis=-1),1e-12)
        nearest=a[None]+np.clip(u,0,1)[...,None]*edge[None]
        distance=np.linalg.norm(query[:,None]-nearest,axis=-1).min(axis=1)
        return np.where(inside,0,distance)
    except QhullError:
        return np.linalg.norm(query[:,None]-projected_corners[None],axis=-1).min(axis=1)


def silhouette_residuals(half,rotation,translation,scale,evidence,samples,corners):
    camera=evidence['c2w'].copy();camera[:3,3]*=scale
    def project(local):
        world=(local*half)@rotation.T+translation
        xyz=(world-camera[:3,3])@camera[:3,:3]
        image=xyz@evidence['K'].T
        return image[:,:2]/np.maximum(image[:,2,None],1e-6),xyz[:,2]
    uv,z=project(samples)
    h,w=evidence['shape']
    in_view=(z>0)&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    coords=np.vstack([np.clip(uv[:,1],0,h-1),np.clip(uv[:,0],0,w-1)])
    outside=map_coordinates(evidence['outside_distance'],coords,order=1,mode='nearest')
    hidden=map_coordinates(evidence['hand_mask'].astype(float),coords,order=0,mode='nearest')>.5
    outside=np.where(in_view&~hidden,outside,0) # outside image is also unobserved
    projected,_=project(corners)
    coverage=_outside_hull_distance(evidence['boundary_uv'],projected)
    return outside/np.sqrt(max(1,len(outside))),coverage/np.sqrt(max(1,len(coverage)))


def fit_shared_block(initial_scene,frames,timestamps,branch,anchor_scale,config):
    """Robust joint shape/pose fit with no table/contact projection or attraction."""
    validate_scene(initial_scene,require_metric_scale=True)
    initial=next(o for o in initial_scene['objects'] if o['id']==config['object_id'])
    if initial['geometry']['type']!='box' or len(initial_scene['timestamps'])!=1:
        raise ValueError('Need a supported first-image shared cuboid initializer')
    half0=np.asarray(initial['geometry']['half_extents_m'],float)
    pose0=np.asarray(initial['T_world_object'][0],float)
    count=len(timestamps)
    unposed=branch in {'source_unposed','helper_unposed'}
    if not unposed and branch!='helper_requested_cameras':
        raise ValueError('Unsupported DA3 forecast branch')
    scale0=anchor_scale if unposed else 1.
    nonpenetration=config['nonpenetration_weight']>0
    if nonpenetration and initial_scene['support'] is None:
        raise ValueError('Optional nonpenetration prior requires the first-image predicted support plane')
    rest_anchor,rest_report=initial_rest_regime(frames,timestamps,config)
    free_frames=np.flatnonzero((np.arange(count)>0)&~rest_anchor)
    pose_column={int(t):3+6*j for j,t in enumerate(free_frames)}
    supported=np.array([sum(len(v['points']) for v in frame if v['known'])>=config['minimum_object_points'] for frame in frames])
    centres=[]
    for frame in frames:
        clouds=[v['points'] for v in frame if v['known'] and len(v['points'])]
        centres.append(np.median(np.concatenate(clouds),axis=0)*scale0 if clouds else None)
    offset=pose0[:3,3]-centres[0] if centres[0] is not None else np.zeros(3)
    translations=np.tile(pose0[:3,3],(count,1))
    for t in range(1,count):
        translations[t]=centres[t]+offset if centres[t] is not None else translations[t-1]
    # Anchored poses have no optimization variables: this is not a post-fit snap.
    x0=np.r_[np.log(half0),np.column_stack([np.zeros((len(free_frames),3)),translations[free_frames]]).ravel(),
             np.full(count-1,np.log(scale0)) if unposed else np.empty(0)]
    shape_lo,shape_hi=config['shape_ratio_bounds']
    lo=np.r_[np.log(half0*shape_lo),np.column_stack([np.full((len(free_frames),3),-np.pi),translations[free_frames]-config['maximum_translation_m']]).ravel(),
             np.full(count-1,np.log(scale0*config['scale_ratio_bounds'][0])) if unposed else np.empty(0)]
    hi=np.r_[np.log(half0*shape_hi),np.column_stack([np.full((len(free_frames),3),np.pi),translations[free_frames]+config['maximum_translation_m']]).ravel(),
             np.full(count-1,np.log(scale0*config['scale_ratio_bounds'][1])) if unposed else np.empty(0)]
    samples=box_surface_samples(config['surface_grid'])
    corners=np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]],float)
    scale_start=3+6*len(free_frames)
    def decode(x):
        half=np.exp(x[:3]);pose=np.tile(pose0,(count,1,1))
        if len(free_frames):
            future=x[3:scale_start].reshape(len(free_frames),6)
            pose[free_frames,:3,:3]=Rotation.from_rotvec(future[:,:3]).as_matrix()@pose0[:3,:3]
            pose[free_frames,:3,3]=future[:,3:]
        scales=np.r_[scale0,np.exp(x[scale_start:])] if unposed else np.ones(count)
        return half,pose,scales
    frame_sizes=[]
    for frame in frames:
        frame_sizes.append(sum(len(v['points'])+len(samples)+len(v['boundary_uv']) for v in frame if v['known']))
    def residual(x):
        half,poses,scales=decode(x);values=[]
        for t,frame in enumerate(frames):
            for v in frame:
                if not v['known']:
                    continue
                surface=box_sdf(v['points']*scales[t],half,poses[t,:3,:3],poses[t,:3,3])
                values.append(surface/(config['depth_sigma_m']*np.sqrt(max(1,len(surface)))))
                outside,coverage=silhouette_residuals(half,poses[t,:3,:3],poses[t,:3,3],scales[t],v,samples,corners)
                values.extend([outside/config['silhouette_sigma_px'],coverage/config['silhouette_sigma_px']])
        values.append(np.log(half/half0)/config['shape_log_sigma'])
        if len(free_frames):
            future=x[3:scale_start].reshape(len(free_frames),6)
            values.extend([future[:,:3].ravel()/config['pose_rotation_prior_sigma_rad'],
                ((future[:,3:]-translations[free_frames])/config['pose_translation_prior_sigma_m']).ravel()])
        if unposed:
            values.append((x[scale_start:]-np.log(scale0))/config['scale_log_sigma'])
        if nonpenetration:
            clearances=plane_clearance(half,poses,initial_scene['support'])
            values.append(np.minimum(clearances,0)*np.sqrt(config['nonpenetration_weight'])/config['nonpenetration_sigma_m'])
        return np.concatenate(values)
    sparsity=lil_matrix((len(residual(x0)),len(x0)),dtype=int);cursor=0
    for t,size in enumerate(frame_sizes):
        sparsity[cursor:cursor+size,:3]=1
        if t in pose_column:
            col=pose_column[t];sparsity[cursor:cursor+size,col:col+6]=1
        if t and unposed:sparsity[cursor:cursor+size,scale_start+t-1]=1
        cursor+=size
    sparsity[cursor:cursor+3,:3]=1;cursor+=3
    for field in [0,3]:
        for t in free_frames:
            col=pose_column[t]+field;sparsity[cursor:cursor+3,col:col+3]=1;cursor+=3
    if unposed:
        for t in range(1,count):
            sparsity[cursor,scale_start+t-1]=1;cursor+=1
    if nonpenetration:
        for t in range(count):
            sparsity[cursor,:3]=1
            if t in pose_column:
                col=pose_column[t];sparsity[cursor,col:col+6]=1
            cursor+=1
    start=time.monotonic()
    result=least_squares(residual,x0,jac_sparsity=sparsity.tocsr(),bounds=(lo,hi),loss='soft_l1',f_scale=1.,
        max_nfev=config['max_nfev'],xtol=1e-5,ftol=1e-5,gtol=1e-5)
    half,poses,scales=decode(result.x)
    # Unsupported frames are placeholders, not a fit inferred from an empty mask.
    for t in range(1,count):
        if not supported[t]:
            poses[t]=poses[t-1];scales[t]=scales[t-1]
    supported[0]=False # the initial pose is observed-model initialization, not a future prediction
    clearance=plane_clearance(half,poses,initial_scene['support'])
    return dict(half_extents_m=half,poses=poses,scales=scales,prediction_supported=supported,
        report=dict(status='fitted',optimizer_success=bool(result.success),optimizer_message=str(result.message),
            nfev=int(result.nfev),initial_objective=float(np.sum(residual(x0)**2)),final_objective=float(np.sum(residual(result.x)**2)),
            seconds=time.monotonic()-start,first_pose_fixed=True,one_shared_shape=True,
            free_future_SE3=not bool(rest_anchor[1:].any()),free_pose_frame_indices=free_frames.tolist(),
            initial_rest_prior=rest_report,
            scale_mode='first-image anchor plus per-time rigidity/shape estimates' if unposed else 'inherited requested-camera model scale',
            nonpenetration_soft_prior_weight=config['nonpenetration_weight'],
            nonpenetration_hard_guarantee=False,plane_snap_used=False,
            estimated_plane_clearance_m=clearance.tolist() if clearance is not None else None,
            negative_clearance_frames=np.flatnonzero(clearance<0).tolist() if clearance is not None else None,
            first_pose_shape_caveat='Pose zero is fixed, but shared dimensions can change its initial clearance.',
            table_contact_constraints_used=False,forced_grasp_used=False,simulator_used=False,GT_used=False))


def merge_inputs(manifest_paths):
    merged=None;by_id={};inputs=[];root_hashes=set()
    for path in map(lambda p:Path(p).resolve(),manifest_paths):
        report=load_manifest(path)
        if report['condition']!='rgb_only' or report['task_mode']!='forecast':
            raise ValueError('All forecast reconstruction inputs require strict RGB-only forecast lineage')
        manifest=json.loads(path.read_text())
        if merged is None:
            merged=deepcopy(manifest)
        elif manifest['observed_until_s']!=merged['observed_until_s']:
            raise ValueError('Forecast cutoffs differ')
        active=set(report['method_artifact_ids'])
        for original in manifest['artifacts']:
            if original['id'] not in active:
                continue
            record=deepcopy(original)
            if 'path' in record:
                record['path']=str((path.parent/record['path']).resolve())
            if record['origin']=='observed_rgb':
                root_hashes.add(record['sha256'])
            previous=by_id.get(record['id'])
            if previous is not None and ({k:v for k,v in previous.items() if k!='path'} != {k:v for k,v in record.items() if k!='path'}):
                raise ValueError('Conflicting forecast artifact: '+record['id'])
            by_id[record['id']]=record
        inputs.extend(report['method_inputs'])
    if len(root_hashes)!=1:
        raise ValueError('Forecast inputs must share exactly the same first observed RGB')
    merged['artifacts']=list(by_id.values());merged['method_inputs']=list(dict.fromkeys(inputs))
    return merged


def export_scene(initial_scene,fit,timestamps,artifact_ids,object_id,task_text):
    initial=next(o for o in initial_scene['objects'] if o['id']==object_id)
    observed=np.zeros(len(timestamps),bool);observed[0]=bool(initial['observed'][0])
    if not observed[0]:
        raise ValueError('Forecast export needs a supported initial observed-image object')
    prediction_supported=np.asarray(fit['prediction_supported'],bool).copy();prediction_supported[0]=False
    provenance=dict(method='robust shared cuboid + forecast SE3/depth/silhouette fit',artifact_ids=artifact_ids,
        confidence_status='uncalibrated',hidden_surface_status='unknown',
        first_pose_status='fixed first-observed-image model initializer',
        future_pose_status='model forecast estimate, not recorded trajectory',
        rotation_status='estimated from generated evidence; cuboid symmetries unresolved',
        fallback_pose_status='unsupported carry-forward placeholder')
    obj=object_from_arrays(object_id=object_id,poses=fit['poses'],observed=observed,provenance=provenance,
        primitive=dict(type='box',half_extents_m=fit['half_extents_m'].tolist()))
    obj['prediction_supported']=prediction_supported.tolist()
    obj['initial_fit_supported']=bool(observed[0])
    obj['prediction_provenance']=dict(origin='model_prediction',generated=True,artifact_ids=artifact_ids,
        method='DA3 predicted depth/cameras and SAM3 masks on world-model-generated future RGB')
    scene=scene_from_arrays(timestamps=timestamps,coordinate_frame=initial_scene['coordinate_frame'],
        metric_scale=dict(status='estimated',provenance=dict(method=fit['report']['scale_mode'],artifact_ids=artifact_ids,
            confidence_status='uncalibrated')),
        actors=[],objects=[obj],support=deepcopy(initial_scene['support']),contacts=[],
        uncertainty=dict(status='estimated',provenance=provenance,
            interpretation='Shape, motion, depth scales and generated camera consistency are uncalibrated predictions; contact remains unknown.'),
        task=dict(description=task_text,target=None))
    scene['forecast']=dict(observed_until_s=float(timestamps[0]),observed_rgb_frames=1,
        future_recorded_RGB_used=False,generated_pose_support=prediction_supported.tolist(),
        contact_state='unknown; no contact intervals inferred',
        source_camera_assumption='fixed physical source camera; each DA3 group gauges view0 to identity',
        per_frame_depth_scale=fit['scales'].tolist(),support_policy=('first-image predicted plane copied; optional soft nonpenetration prior, no hard guarantee'
            if fit['report']['nonpenetration_soft_prior_weight']>0 else 'first-image predicted plane copied; not fitted/enforced across motion'))
    scene['forecast']['initial_rest_prior']=deepcopy(fit['report']['initial_rest_prior'])
    if 'initial_support_prior' in initial_scene:
        scene['initial_support_prior']=deepcopy(initial_scene['initial_support_prior'])
        scene['initial_support_prior_scope']='Initial-image model/task assumption only; no future support inferred'
    return scene


def render_and_measure(output,predictions,frames,fit):
    from PIL import Image,ImageDraw
    output=Path(output);(output/'overlays').mkdir();(output/'mask-overlays').mkdir()
    mesh,faces=geometry_arrays(dict(type='box',half_extents_m=fit['half_extents_m']))
    reports=[]
    for t,frame in enumerate(frames):
        pose=fit['poses'][t];world=mesh@pose[:3,:3].T+pose[:3,3]
        render_panels=[];mask_panels=[];views=[]
        for v,evidence in enumerate(frame):
            rgb=predictions['rgb'][t,v]
            c=evidence['c2w'].copy();c[:3,3]*=fit['scales'][t]
            depth=predictions['depth'][t,v]*fit['scales'][t]
            rendered=raster_depth(world,faces,evidence['K'],c,rgb.shape[:2])
            result=agreement(rendered,depth,evidence['object_mask'],evidence['known'],evidence['hand_mask'],np.isfinite(depth)&(depth>0))
            result.update(view=v,mask_status=evidence['status'],selected_candidate_row=evidence['selected_row'],
                          object_points=len(evidence['points']),hand_points=len(evidence['hand_points']))
            hands=evidence['hand_points']*fit['scales'][t]
            objects=evidence['points']*fit['scales'][t]
            result['hand_to_object_centroid_distance_m']=float(np.linalg.norm(np.median(hands,axis=0)-np.median(objects,axis=0))) if len(hands) and len(objects) else None
            result['hand_to_object_predicted_point_distance_q10_m']=float(np.quantile(cKDTree(objects).query(hands)[0],.1)) if len(hands) and len(objects) else None
            result['hand_to_box_min_signed_distance_m']=float(box_sdf(hands,fit['half_extents_m'],pose[:3,:3],pose[:3,3]).min()) if len(hands) else None
            result['hand_to_object_mask_gap_px']=float(distance_transform_edt(~evidence['object_mask'])[evidence['hand_mask']].min()) if evidence['object_mask'].any() and evidence['hand_mask'].any() else None
            result['contact_interpretation']='model-prediction proximity only; no touch/grasp label'
            overlay=rgb.copy();visible=(rendered>0)&~evidence['hand_mask']
            overlay[visible]=(.4*overlay[visible]+.6*np.array([240,170,50])).astype(np.uint8)
            marked=rgb.copy();mask=evidence['object_mask'];hand=evidence['hand_mask']
            marked[mask]=(.45*marked[mask]+.55*np.array([40,230,90])).astype(np.uint8)
            marked[hand]=(.45*marked[hand]+.55*np.array([190,60,230])).astype(np.uint8)
            render_panels.append(overlay);mask_panels.append(marked);views.append(result)
        label=f'frame {t} | '+('initial observed source; helper views generated' if t==0 else 'all views generated forecast')
        for directory,panels in [('overlays',render_panels),('mask-overlays',mask_panels)]:
            image=Image.fromarray(np.concatenate(panels,axis=1));draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,image.width,15),fill=(15,15,15));draw.text((3,2),label,fill=(255,255,255))
            image.save(output/directory/f'{t:03d}.png')
        reports.append(dict(frame=t,timestamp_s=float(predictions['timestamps'][t]),
            prediction_supported=bool(fit['prediction_supported'][t]),depth_scale=float(fit['scales'][t]),views=views))
    return reports


def run(*,condition,helper,predictions,masks,output,config_path=None):
    condition,helper,predictions,masks,output=map(lambda p:Path(p).resolve(),(condition,helper,predictions,masks,output))
    if output.exists():
        raise FileExistsError(output)
    config=json.loads(Path(config_path).read_text()) if config_path else default_config()
    if set(config)-{'initial_rest_prior'}!=set(default_config()):
        raise ValueError('Use a complete saved forecast geometry configuration')
    if (config['max_nfev']<1 or config['minimum_object_points']<3 or config['surface_grid']<2
            or config['nonpenetration_weight']<0 or config['nonpenetration_sigma_m']<=0):
        raise ValueError('Invalid fitting budget/evidence configuration')
    start=time.monotonic()
    job=read_inputs(helper,condition)
    if job['mode']!='forecast':
        raise ValueError('This adapter only accepts strict first-image forecast inputs')
    prediction_report=load_manifest(predictions/'input-manifest.json',
        [('depth',predictions/'predictions.npz'),('cameras',predictions/'predictions.npz')])
    mask_report=load_manifest(masks/'input-manifest.json',[('masks',masks/'masks.npz')])
    receipt=json.loads((predictions/'execution.json').read_text())
    branch=receipt['condition']
    if receipt.get('status')!='complete' or receipt.get('task_mode')!='forecast' or receipt.get('output_sha256')!=file_sha256(predictions/'predictions.npz'):
        raise ValueError('DA3 forecast predictions are incomplete or changed')
    if prediction_report['task_mode']!='forecast' or mask_report['task_mode']!='forecast':
        raise ValueError('DA3/masks must retain strict forecast ancestry')
    with np.load(predictions/'predictions.npz',allow_pickle=False) as z:
        values={k:z[k].copy() for k in z.files}
    expected=job['rgb'][:,:1] if branch=='source_unposed' else job['rgb']
    if not np.array_equal(values['rgb'],expected) or not np.array_equal(values['timestamps'],job['timestamps']):
        raise ValueError('DA3 predictions differ from the selected generated RGB stream')
    if not np.array_equal(values['source_frame_indices'],job['ids']):
        raise ValueError('Forecast future source indices must remain -1')
    with np.load(masks/'masks.npz',allow_pickle=False) as z:
        mask_values={k:z[k].copy() for k in z.files}
    prep=json.loads((condition/'preparation.json').read_text())
    initial_scene=load_scene(prep['scene_path'],require_metric_scale=True)
    if config['object_id']!=prep['object_id']:
        raise ValueError('Fitter object ID differs from the first-image camera target')
    with np.load(Path(prep['geometry_manifest']).parent/'geometry.npz',allow_pickle=False) as z:
        depth_name='depth_m' if 'depth_m' in z else 'depth'
        initial_depth=z[depth_name][0,0].copy()
    manifest=merge_inputs([predictions/'input-manifest.json',masks/'input-manifest.json',
                           prep['geometry_manifest'],prep['scene_manifest']])
    if branch=='source_unposed':
        active=validate_manifest(manifest,output,verify_files=True)['method_artifact_ids']
        if any(a['id'] in active and a.get('view_role')=='helper' for a in manifest['artifacts']):
            raise ValueError('Source-only branch requires separately exported source-only masks, with no generated-helper ancestry')
    if branch=='helper_requested_cameras' and prediction_report['metric_scale']['status']!='estimated':
        raise ValueError('Requested-camera branch lacks its inherited estimated metric-scale declaration')
    frames=prepare_evidence(values,mask_values,config)
    scale,anchor=initial_scale_anchor(initial_depth,prep['source_crop_xywh'],values['depth'][0,0],frames[0][0]['object_mask'])
    fit=fit_shared_block(initial_scene,frames,values['timestamps'],branch,scale,config)
    source_ids=sorted({a['id'] for a in manifest['artifacts'] if a['id'] in set(prediction_report['method_artifact_ids']+mask_report['method_artifact_ids'])
        and (a['kind'] in {'depth','cameras','masks'} or a.get('generated') is True)})
    scene=export_scene(initial_scene,fit,values['timestamps'],source_ids,config['object_id'],prep['task_text'])
    output.mkdir(parents=True)
    write_json(output/'scene-sequence.json',scene);write_json(output/'fit-config.json',config)
    scaled_c=values['c2w'].copy();scaled_c[:,:,:3,3]*=fit['scales'][:,None,None]
    object_masks=np.asarray([[v['object_mask'] for v in frame] for frame in frames])
    hand_masks=np.asarray([[v['hand_mask'] for v in frame] for frame in frames])
    known=np.asarray([[v['known'] for v in frame] for frame in frames])
    scaled_depth=values['depth']*fit['scales'][:,None,None,None]
    np.savez_compressed(output/'geometry.npz',rgb=values['rgb'],depth_m=scaled_depth,
        valid=np.isfinite(scaled_depth)&(scaled_depth>0),K=values['K'],c2w=scaled_c,
        timestamps=values['timestamps'],source_frame_indices=values['source_frame_indices'],
        observed_mask=job['observed_mask'][:,:scaled_depth.shape[1]],object_masks=object_masks,
        object_mask_known=known,occlusion_mask=hand_masks,estimated_depth_scale=fit['scales'],
        prediction_supported=fit['prediction_supported'])
    parents=manifest['method_inputs'].copy()
    for aid,kind,name in [('forecast_scaled_depth','depth','geometry.npz'),('forecast_scaled_cameras','cameras','geometry.npz'),
                          ('forecast_block_scene','object_geometry','scene-sequence.json')]:
        manifest['artifacts'].append(dict(id=aid,kind=kind,origin='model_prediction',parent_ids=parents,
            path=name,sha256=file_sha256(output/name),description='Shared forecast shape/SE3 model fit; not observed future geometry'))
    manifest['method_inputs']=['forecast_scaled_depth','forecast_scaled_cameras','forecast_block_scene']
    manifest['metric_scale']=dict(status='estimated',artifact_ids=['forecast_scaled_depth'],description=fit['report']['scale_mode'])
    manifest['robot_registration']=dict(status='unresolved',artifact_ids=[],description='No target-robot registration inferred from forecast pixels')
    validation=validate_manifest(manifest,output,verify_files=True)
    write_json(output/'input-manifest.json',manifest)
    per_frame=render_and_measure(output,values,frames,fit)
    report=dict(status='complete',method='numerical shared-cuboid forecast initializer, not an agent result',
        branch=branch,fit=fit['report'],scale_anchor=anchor,per_frame=per_frame,
        agent_calls=0,source_observed_frames=1,forecast_prediction_frames=len(values['timestamps'])-1,
        observed_frame_indices=[0],prediction_supported=fit['prediction_supported'].tolist(),
        half_extents_m=fit['half_extents_m'].tolist(),estimated_depth_scales=fit['scales'].tolist(),
        scene_sha256=file_sha256(output/'scene-sequence.json'),producer_sha256=file_sha256(__file__),
        input_protocol=validation,total_seconds=time.monotonic()-start,
        GT_geometry_accuracy=None,contact_accuracy=None,execution_success=None,
        limitations=['Generated-view agreement is not ground-truth reconstruction or forecast accuracy.',
            'Unposed per-time scale is inferred using shared shape and first-image priors; it is not directly observed.',
            'A cuboid cannot recover articulated hand geometry or small contact affordances.',
            'Cuboid rotation symmetry and generated camera/motion consistency remain unresolved.',
            'First pose/support inherit first-image errors; optional initial-rest anchoring is an RGB-derived hypothesis, not verified contact.'])
    write_json(output/'execution.json',report)
    write_json(output/'diagnostics.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    config=commands.add_parser('config');config.add_argument('--output',type=Path,required=True)
    config.add_argument('--nonpenetration-weight',type=float,default=0.)
    config.add_argument('--initial-rest',action='store_true',help='Add the optional source-RGB initial-rest prefix hypothesis')
    fit=commands.add_parser('run')
    for name in ['condition','helper','predictions','masks','output']:
        fit.add_argument('--'+name,type=Path,required=True)
    fit.add_argument('--config-path',type=Path)
    args=vars(parser.parse_args());mode=args.pop('command')
    if mode=='config':
        if args['output'].exists():raise FileExistsError(args['output'])
        settings=default_config();settings['nonpenetration_weight']=args['nonpenetration_weight']
        if settings['nonpenetration_weight']<0:raise ValueError('Nonpenetration weight must be nonnegative')
        if args['initial_rest']:
            settings['initial_rest_prior']=dict(default_initial_rest_prior(),enabled=True)
        write_json(args['output'],settings)
    else:
        print(json.dumps(run(**args),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
