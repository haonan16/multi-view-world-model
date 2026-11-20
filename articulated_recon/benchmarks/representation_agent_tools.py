"""Small offline scene interface copied into each frozen agent workspace.

Only source/model prediction consistency is measured. Unknown/occluded pixels
are excluded, never treated as free space. Rendering assumes pinhole K and is
approximate when the frontend reports a nonzero dense-ray pinhole residual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

try:
    from scene_sequence import validate_scene
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from artirecon.scene_sequence import validate_scene


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(root):
    root = Path(root).resolve()
    frozen = json.loads((root / 'frozen-inputs.json').read_text())
    for name, expected in frozen['files'].items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Frozen input is not a local ordinary file: ' + name)
        if digest(path) != expected:
            raise ValueError('Frozen input changed: ' + name)
    allowed = set(frozen['files']) | {'frozen-inputs.json','execution-receipt.json'}
    for path in root.rglob('*'):
        name = str(path.relative_to(root))
        if path.is_symlink() or (path.is_file() and name not in allowed and not name.startswith('output/')):
            raise ValueError('Unlisted/nonlocal packet file: ' + name)
    packet = json.loads((root / 'packet.json').read_text())
    with np.load(root / 'observations.npz', allow_pickle=False) as z:
        obs = {k: z[k].copy() for k in z.files}
    return packet, obs


def validate_export(scene, packet):
    counts = validate_scene(scene)
    if not np.array_equal(scene['timestamps'], packet['timestamps']):
        raise ValueError('Export timestamps must equal selected packet timestamps exactly')
    if scene['coordinate_frame'] != packet['coordinate_frame']:
        raise ValueError('Export must retain the packet coordinate frame')
    if scene['metric_scale']['status'] != packet['metric_scale_status']:
        raise ValueError('Export must retain the predicted/unresolved metric-scale status')
    if not scene['objects'] or packet['object_id'] not in {o['id'] for o in scene['objects']}:
        raise ValueError('Export must include the packet task object')
    for obj in scene['objects']:
        if obj['geometry'].get('uri') is not None:
            raise ValueError('Export meshes must use inline vertices_m/faces, not external URIs')
    # Contact is a source hypothesis, never verified contact truth in this task.
    if any(c['status'] == 'observed' for c in scene['contacts']):
        raise ValueError('Source-only contact must remain unknown or explicitly hypothesized')
    if packet.get('task_mode')=='forecast':
        cutoff=packet['observed_until_s'];times=np.asarray(packet['timestamps'])
        if scene.get('forecast',{}).get('observed_until_s')!=cutoff:
            raise ValueError('Forecast export requires the unchanged observed cutoff')
        for obj in scene['objects']:
            if np.asarray(obj['observed'])[times>cutoff].any():
                raise ValueError('Generated future poses cannot be labeled observed')
            predicted=np.asarray(obj.get('prediction_supported'))
            if predicted.shape!=times.shape or predicted.dtype.kind!='b' or predicted[0]:
                raise ValueError('Forecast needs boolean prediction_supported[T], false at initial observed frame')
            provenance=obj.get('prediction_provenance',{})
            if provenance.get('origin')!='model_prediction' or provenance.get('generated') is not True:
                raise ValueError('Forecast prediction provenance must declare generated model predictions')
            ids=provenance.get('artifact_ids')
            if not isinstance(ids,list) or not ids or not set(ids).issubset(packet['prediction_artifact_ids']) or not provenance.get('method'):
                raise ValueError('Forecast provenance requires a method and actual packet prediction artifact IDs')
        candidate_scales(scene,packet)
    return counts


def candidate_scales(scene,packet):
    count=len(packet['timestamps'])
    default=packet.get('forecast',{}).get('common_depth_scale',1.)
    values=np.asarray(scene.get('forecast',{}).get('per_frame_depth_scale',np.full(count,default)),float)
    if values.shape!=(count,) or not np.isfinite(values).all() or (values<=0).any():
        raise ValueError('per_frame_depth_scale must contain positive finite raw-DA3 scale factors')
    return values


def contact_sheet(images,labels,path,columns=4):
    from PIL import ImageDraw
    if not images:return
    h,w=images[0].shape[:2];rows=(len(images)+columns-1)//columns
    sheet=Image.new('RGB',(columns*w,rows*(h+18)),(24,29,37));draw=ImageDraw.Draw(sheet)
    for index,(picture,label) in enumerate(zip(images,labels)):
        x=(index%columns)*w;y=(index//columns)*(h+18)
        sheet.paste(Image.fromarray(picture),(x,y+18));draw.text((x+3,y+2),label,fill=(245,245,245))
    sheet.save(path)


def geometry_arrays(geometry):
    kind = geometry['type']
    if kind == 'mesh':
        v, f = np.asarray(geometry['vertices_m'], float), np.asarray(geometry['faces'], int)
        return v @ np.asarray(geometry.get('local_linear', np.eye(3))).T, f
    if kind == 'box':
        v = np.asarray([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], float)
        f = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
             [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
        return v * geometry['half_extents_m'], np.asarray(f)
    if kind == 'cylinder':
        n, radius, height = 32, geometry['radius_m'], geometry['height_m']
        theta = np.arange(n) * 2*np.pi/n
        ring = np.column_stack([radius*np.cos(theta), radius*np.sin(theta)])
        v = np.concatenate([np.column_stack([ring, np.full(n, -height/2)]),
                            np.column_stack([ring, np.full(n, height/2)]), [[0,0,-height/2],[0,0,height/2]]])
        f = []
        for i in range(n):
            j = (i+1) % n
            f.extend([[i,j,n+j], [i,n+j,n+i], [2*n,j,i], [2*n+1,n+i,n+j]])
        return v, np.asarray(f)
    if kind == 'sphere':
        n, rings = 24, 12
        theta, phi = np.meshgrid(np.arange(n)*2*np.pi/n, np.linspace(0, np.pi, rings+1))
        v = geometry['radius_m'] * np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], axis=-1).reshape(-1,3)
        f = []
        for r in range(rings):
            for i in range(n):
                a, b = r*n+i, r*n+(i+1)%n
                f.extend([[a,b,b+n],[a,b+n,a+n]])
        return v, np.asarray(f)
    raise ValueError('Unsupported geometry')


def raster_depth(vertices, faces, k, c2w, size):
    """Deterministic triangle Z buffer; pixel-centre samples, no hidden fill."""
    height, width = size
    camera = (np.asarray(vertices) - c2w[:3, 3]) @ c2w[:3, :3]
    projected = camera @ k.T
    uv = projected[:, :2] / np.where(projected[:, 2:] != 0, projected[:, 2:], 1)
    zbuffer = np.full((height, width), np.inf)
    for face in faces:
        xyz, tri = camera[face], uv[face]
        if (xyz[:, 2] <= 1e-6).any():
            continue
        x0, y0 = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(tri.max(axis=0)).astype(int), [width-1,height-1])
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = tri
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-12:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        u = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1]))/denominator
        v = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1]))/denominator
        w = 1-u-v
        inside = (u >= -1e-8) & (v >= -1e-8) & (w >= -1e-8)
        inverse = u/xyz[0,2] + v/xyz[1,2] + w/xyz[2,2]
        depth = np.divide(1., inverse, out=np.full_like(inverse,np.inf), where=inverse > 0)
        patch = zbuffer[y0:y1+1, x0:x1+1]
        np.minimum(patch, np.where(inside, depth, np.inf), out=patch)
    return np.where(np.isfinite(zbuffer), zbuffer, 0).astype(np.float32)


def agreement(rendered, reference_depth, reference_mask, mask_known, occlusion, valid):
    """Score only evidence that is available; null != empty/zero/error-free."""
    if not mask_known:
        return dict(status='unknown_object_mask', visible_mask_iou=None,
                    predicted_depth_agreement_median_m=None, depth_overlap_pixels=0)
    known_space = ~occlusion
    reference = reference_mask & known_space
    prediction = (rendered > 0) & known_space
    if not reference.any():
        return dict(status='no_visible_object_reference', visible_mask_iou=None,
                    predicted_depth_agreement_median_m=None, depth_overlap_pixels=0)
    overlap = prediction & reference & valid
    union = prediction | reference
    return dict(status='measured_source_prediction_agreement',
        visible_mask_iou=float((prediction & reference).sum()/union.sum()),
        predicted_depth_agreement_median_m=float(np.median(np.abs(rendered[overlap]-reference_depth[overlap]))) if overlap.any() else None,
        depth_overlap_pixels=int(overlap.sum()), visible_reference_pixels=int(reference.sum()),
        excluded_occluded_pixels=int(occlusion.sum()))


def check_scene(root, scene_path, output):
    root, scene_path, output = map(lambda p: Path(p).resolve(), (root, scene_path, output))
    packet, obs = load_inputs(root)
    scene = json.loads(scene_path.read_text())
    counts = validate_export(scene, packet)
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(root / 'output'):
        raise ValueError('Source tool outputs must be under workspace/output')
    output.mkdir(parents=True)
    target = packet['object_id']
    obj=next(o for o in scene['objects'] if o['id']==target)
    vertices,faces=geometry_arrays(obj['geometry']);rows=[]
    scales=candidate_scales(scene,packet)
    sheets=[[] for _ in range(obs['rgb'].shape[1])]
    for t in range(len(obs['timestamps'])):
        for v in range(obs['rgb'].shape[1]):
            rgb=obs['rgb'][t,v];k,c=obs['K'][t,v],obs['c2w'][t,v]
            pose = np.asarray(obj['T_world_object'][t])
            world = vertices @ pose[:3, :3].T + pose[:3, 3]
            depth=(raster_depth(world,faces,k,c,rgb.shape[:2]) if obs['camera_projection_valid'][t,v] else np.zeros(rgb.shape[:2],np.float32))
            row=agreement(depth,obs['depth_m'][t,v],obs['object_masks'][t,v],bool(obs['object_mask_known'][t,v]),
                          obs['occlusion_mask'][t,v],obs['depth_valid'][t,v])
            if not obs['camera_projection_valid'][t,v]:
                row.update(status='camera_projection_unavailable',visible_mask_iou=None,predicted_depth_agreement_median_m=None,depth_overlap_pixels=0)
            row.update(selected_frame=t,view=v,native_frame=int(obs['native_frame_ids'][t]),
                timestamp_s=float(obs['timestamps'][t]),exported_observed=obj['observed'][t],
                reference_scope='common frozen model depth/cameras; same for both agent conditions')
            if 'raw_depth' in obs:
                candidate_c=obs['raw_c2w'][t,v].copy();candidate_c[:3,3]*=scales[t]
                rendered=raster_depth(world,faces,k,candidate_c,rgb.shape[:2])
                diagnostic=agreement(rendered,obs['raw_depth'][t,v]*scales[t],obs['object_masks'][t,v],
                    bool(obs['object_mask_known'][t,v]),obs['occlusion_mask'][t,v],obs['depth_valid'][t,v])
                diagnostic.update(raw_depth_scale=float(scales[t]),interpretation='candidate-conditioned scale/camera diagnostic; not the common-reference primary comparison')
                row['candidate_scale_diagnostics']=diagnostic
                row['source_frame_index']=int(obs['source_frame_indices'][t])
                row['rgb_observed']=bool(obs['observed_mask'][t,v])
            if 'K_pinhole_rmse_px' in obs:
                residual=float(obs['K_pinhole_rmse_px'][t,v]);row['predicted_camera_pinhole_rmse_px']=residual if np.isfinite(residual) else None
            rows.append(row)
            overlay=rgb.copy();visible=(depth>0)&~obs['occlusion_mask'][t,v]
            overlay[visible]=(.45*rgb[visible]+.55*np.array([240,160,50])).astype(np.uint8)
            unknown=obs['occlusion_mask'][t,v];overlay[unknown]=(.5*rgb[unknown]+.5*np.array([130,50,180])).astype(np.uint8)
            stem=f'{t:03d}' if len(sheets)==1 else f'{t:03d}-view{v}'
            Image.fromarray(np.concatenate([rgb,overlay],axis=1)).save(output/f'{stem}-rgb-overlay.png')
            np.save(output/f'{stem}-rendered-depth.npy',depth);sheets[v].append(overlay)
    for v,pictures in enumerate(sheets):
        contact_sheet(pictures,[f'frame {t} | t={obs["timestamps"][t]:.3f}s | view {v}' for t in range(len(pictures))],output/f'contact-sheet-view-{v}.png')
    ious = [x['visible_mask_iou'] for x in rows if x['visible_mask_iou'] is not None]
    depths = [x['predicted_depth_agreement_median_m'] for x in rows if x['predicted_depth_agreement_median_m'] is not None]
    report = dict(status='source_check_complete', scene_sha256=digest(scene_path), counts=counts,
        source_mask_iou_mean=float(np.mean(ious)) if ious else None,
        predicted_depth_agreement_median_m=float(np.median(depths)) if depths else None,
        gt_geometry_accuracy=None, contact_accuracy=None, execution_success=None,
        GT_used=False, simulator_used=False, per_frame=rows,
        primary_reference='common frozen model evidence; candidate scales affect only the separately named diagnostic',
        interpretation='Agreement with source model predictions, not GT. Occlusion is unknown. '
                       'Pinhole projection may approximate frontend dense rays. No actor mesh/physical check is performed.')
    (output/'source-check.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def forecast_seed(root,output):
    from copy import deepcopy
    root=Path(root).resolve();output=Path(output).resolve();packet,obs=load_inputs(root)
    if packet.get('task_mode')!='forecast':raise ValueError('Seed tool requires a forecast packet')
    if output.exists() or not output.is_relative_to(root/'output'):raise ValueError('Choose a new scene file under output/')
    scene=json.loads((root/'first-image-scene.json').read_text());count=len(packet['timestamps'])
    scene['timestamps']=packet['timestamps'];scene['actors']=[];scene['contacts']=[]
    for obj in scene['objects']:
        obj['T_world_object']=[deepcopy(obj['T_world_object'][0]) for _ in range(count)]
        obj['observed']=[True]+[False]*(count-1);obj['prediction_supported']=[False]*count
        obj['prediction_provenance']=dict(origin='model_prediction',generated=True,
            artifact_ids=packet['prediction_artifact_ids'],method='Unestimated future placeholder; agent must refine and declare support')
    scene['forecast']=dict(observed_until_s=packet['observed_until_s'],per_frame_depth_scale=[packet['forecast']['common_depth_scale']]*count)
    validate_export(scene,packet);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(scene,indent=2)+'\n')
    return dict(status='initial_only_seed',scene=str(output),future_pose_support=False)


def forecast_optimize(root,output,config_path=None):
    root=Path(root).resolve();output=Path(output).resolve();packet,_=load_inputs(root)
    if packet.get('task_mode')!='forecast':raise ValueError('Optimizer requires forecast packet')
    if output.exists() or not output.is_relative_to(root/'output'):raise ValueError('Choose a new optimizer directory under output/')
    sys.path.insert(0,str(root/'optimizer/benchmarks'))
    import forecast_geometry as fg
    with np.load(root/'raw-predictions.npz',allow_pickle=False) as z:predictions={k:z[k].copy() for k in z.files}
    with np.load(root/'raw-masks.npz',allow_pickle=False) as z:masks={k:z[k].copy() for k in z.files}
    initial=json.loads((root/'first-image-scene.json').read_text())
    config=json.loads(Path(config_path or root/'optimizer-config.json').read_text())
    start=time.monotonic();frames=fg.prepare_evidence(predictions,masks,config)
    fitted=fg.fit_shared_block(initial,frames,predictions['timestamps'],packet['evidence_branch'],packet['forecast']['first_image_ratio'],config)
    scene=fg.export_scene(initial,fitted,predictions['timestamps'],packet['prediction_artifact_ids'],packet['object_id'],packet['task_text'])
    validate_export(scene,packet);output.mkdir(parents=True)
    (output/'scene-sequence.json').write_text(json.dumps(scene,indent=2,allow_nan=False)+'\n')
    report=dict(status='complete',elapsed_seconds=time.monotonic()-start,fit=fitted['report'],scene_sha256=digest(output/'scene-sequence.json'))
    (output/'optimizer-receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect')
    validate = sub.add_parser('validate')
    validate.add_argument('--scene', type=Path, required=True)
    check = sub.add_parser('check')
    check.add_argument('--scene', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    seed=sub.add_parser('seed');seed.add_argument('--output',type=Path,required=True)
    optimize=sub.add_parser('optimize');optimize.add_argument('--output',type=Path,required=True)
    optimize.add_argument('--config',type=Path)
    a = p.parse_args()
    started = time.monotonic()
    packet, obs = load_inputs(ROOT)
    if a.command == 'inspect':
        result = dict(packet=packet, arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in obs.items()})
    elif a.command == 'validate':
        result = validate_export(json.loads(a.scene.read_text()), packet)
    elif a.command=='seed':result=forecast_seed(ROOT,a.output)
    elif a.command=='optimize':result=forecast_optimize(ROOT,a.output,a.config)
    else:
        result = check_scene(ROOT, a.scene, a.output)
    log = ROOT/'output'/'source-tool-events.jsonl'
    log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        stream.write(json.dumps(dict(command=a.command, elapsed_seconds=time.monotonic()-started,
                                    scene_sha256=digest(a.scene) if hasattr(a,'scene') else None))+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
