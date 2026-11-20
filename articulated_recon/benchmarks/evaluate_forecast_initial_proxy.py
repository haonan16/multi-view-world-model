"""Sealed-agent initial geometry versus a recorded RGB-D reconstruction proxy.

No future trajectory, fitted alignment, scale correction or simulator is used.
The reference is an imperfect reconstruction, never described as exact GT.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from artirecon.input_protocol import file_sha256
from artirecon.scene_sequence import validate_scene
from independent_scene import transform_reference
from representation_agent_tools import geometry_arrays
from strict_forecast import _first_image_manifest

TERMINAL={'completed','stopped_at_budget','completed_over_reported_budget','failed','interrupted'}


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as out:json.dump(value,out,indent=2,allow_nan=False)
def relative(path,base):
    p=Path(path);return p if p.is_absolute() else base/p


def verify_seals(path):
    path=Path(path).resolve();seal=read(path)
    if seal.get('all_agents_sealed') is not True or len(seal.get('runs',[]))!=4:
        raise ValueError('All four agents must be sealed before reference access')
    if len({r['id'] for r in seal['runs']})!=4 or not seal.get('suite_sha256'):
        raise ValueError('Need four distinct runs and the frozen suite hash')
    bound=[]
    for r in seal['runs']:
        scene=relative(r['scene_path'],path.parent).resolve();status=relative(r['status_path'],path.parent).resolve()
        if file_sha256(scene)!=r['scene_sha256'] or file_sha256(status)!=r['status_sha256']:
            raise ValueError('Sealed scene/status changed: '+r['id'])
        receipt=read(status)
        if (receipt.get('status') not in TERMINAL or receipt.get('final_scene_valid') is not True
                or receipt.get('final_scene_sha256')!=r['scene_sha256']):
            raise ValueError('Every scored scene needs a valid terminal harness receipt')
        bound.append(dict(r,scene_path=str(scene),status_path=str(status)))
    return seal,bound


def package_seals(seal_path,output):
    seal,runs=verify_seals(seal_path);output=Path(output)
    output.mkdir(parents=True,exist_ok=False);copy=deepcopy(seal)
    for i,(run,record) in enumerate(zip(runs,copy['runs'])):
        for kind in ['scene','status']:
            filename=f'{i}-{kind}.json';data=Path(run[kind+'_path']).read_bytes()
            (output/filename).write_bytes(data);record[kind+'_path']=filename
    copy['original_seal_sha256']=file_sha256(seal_path)
    write(output/'sealed-suite.json',copy);verify_seals(output/'sealed-suite.json')
    verify_seals(seal_path)
    return dict(status='packaged',seal_sha256=file_sha256(output/'sealed-suite.json'),runs=4)


def first_archive_rgb(archive,source_view):
    if 'rgb' not in archive:raise ValueError('Camera archive has no RGB to establish exact source identity')
    rgb=archive['rgb']
    if rgb.dtype!=np.uint8:raise ValueError('Reference RGB must be original uint8 pixels')
    if rgb.ndim==5 and 0<=source_view<rgb.shape[1]:return rgb[0,source_view]
    if rgb.ndim==4 and source_view==0:return rgb[0]
    if rgb.ndim==3 and source_view==0:return rgb
    raise ValueError('Unsupported reference RGB/view layout')


def export_proxy(*,sealed_suite,observed_manifest,raw_reference,source_cameras,source_view,source_camera_id,output):
    # Crucially, neither evaluator file is opened before the actual seals pass.
    seal,_=verify_seals(sealed_suite)
    observed_manifest=Path(observed_manifest).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    _,_,observed,rgb,t0=_first_image_manifest(observed_manifest)
    if seal.get('first_image_rgb_sha256',observed['sha256'])!=observed['sha256']:
        raise ValueError('Observed source differs from sealed comparison')
    if type(source_view)is not int or source_view<0 or not source_camera_id:
        raise ValueError('Declare source view and camera identity')
    stable={str(Path(p).resolve()):file_sha256(p) for p in [sealed_suite,observed_manifest]}
    # Opening an evaluation RGB array is permitted only after all four runs end.
    stable[str(Path(source_cameras).resolve())]=file_sha256(source_cameras)
    with np.load(source_cameras,allow_pickle=False) as z:
        first=first_archive_rgb(z,source_view)
        if first.shape!=rgb[0].shape or not np.array_equal(first,rgb[0]):
            raise ValueError('Reference/source initial RGB differ; refuse geometry scoring, no approximate alignment')
        cameras=z['c2w']
        if cameras.shape==(4,4) and source_view==0:C=cameras.copy()
        elif cameras.ndim==3 and cameras.shape[1:]==(4,4) and source_view<len(cameras):C=cameras[source_view].copy()
        else:raise ValueError('Reference camera layout must be static [V,4,4] or [4,4]')
        verified_id=False
        if 'camera_ids' in z:
            ids=[str(x) for x in np.atleast_1d(z['camera_ids']).tolist()]
            if source_view>=len(ids) or ids[source_view]!=source_camera_id:raise ValueError('Source camera ID mismatch')
            verified_id=True
    stable[str(Path(raw_reference).resolve())]=file_sha256(raw_reference)
    raw=read(raw_reference)
    if len(raw.get('objects',[]))!=1:raise ValueError('Initial proxy requires one declared reference object')
    # Isolate t0 before coordinate conversion; no future shape/pose is scored.
    first_reference=dict(timestamps=[raw['timestamps'][0]],table=raw['table'],objects=[dict(raw['objects'][0],poses=[raw['objects'][0]['poses'][0]])])
    transformed=transform_reference(first_reference,C,np.eye(4))
    output.mkdir(parents=True)
    np.savez_compressed(output/'initial-proxy.npz',vertices=transformed['vertices'],faces=transformed['faces'],
        pose=transformed['poses'][0],support_normal=transformed['support_normal'],support_offset=transformed['support_offset'])
    audit=dict(status='complete',evaluation_only=True,reference_kind='recorded_RGB-D_reconstruction_proxy_NOT_exact_GT',
        suite_sha256=seal['suite_sha256'],sealed_suite_sha256=file_sha256(sealed_suite),
        scene_sha256_by_run={r['id']:r['scene_sha256']for r in seal['runs']},
        observed_manifest_sha256=file_sha256(observed_manifest),observed_rgb_bundle_sha256=observed['sha256'],
        initial_rgb_pixel_identity='exact native uint8 equality',image_size=list(first.shape),
        source_view=source_view,source_camera_id=source_camera_id,camera_id_verified_in_archive=verified_id,
        camera_identity_if_no_id_metadata='Caller-declared fixed source-view mapping; pixel identity verified',
        source_c2w=C.tolist(),coordinate_frame='source_camera',mapping='inverse(source_c2w) only',
        fitted_alignment=False,scale_fit=False,support_snap=False,future_motion_scored=False,
        units='metres, using the archived reconstruction producer convention',
        method_initial_time_s=float(t0),raw_reference_initial_time_s=float(raw['timestamps'][0]),
        proxy_sha256=file_sha256(output/'initial-proxy.npz'),inputs=stable,producer_sha256=file_sha256(__file__))
    verify_seals(sealed_suite)
    _first_image_manifest(observed_manifest)
    if any(file_sha256(p)!=h for p,h in stable.items()):raise ValueError('Frozen export inputs changed')
    write(output/'export-receipt.json',audit)
    return audit


def triangles(vertices,faces):
    v,f=np.asarray(vertices,float),np.asarray(faces)
    if (v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3
            or f.dtype.kind not in 'iu' or not len(f) or f.min()<0 or f.max()>=len(v)):
        raise ValueError('Invalid finite triangle mesh')
    tri=v[f];areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    keep=areas>1e-14
    if not keep.any():raise ValueError('Mesh has no nondegenerate surface')
    return tri[keep],areas[keep]


def surface_samples(vertices,faces,count=2048):
    tri,areas=triangles(vertices,faces);rng=np.random.default_rng(20261120)
    selected=rng.choice(len(tri),count,p=areas/areas.sum());u=np.sqrt(rng.random(count));v=rng.random(count)
    points=(1-u[:,None])*tri[selected,0]+(u*(1-v))[:,None]*tri[selected,1]+(u*v)[:,None]*tri[selected,2]
    centroid=np.average(tri.mean(1),axis=0,weights=areas)
    return points,centroid


def point_mesh_distances(points,vertices,faces):
    """Exact nearest-triangle distance at deterministic sampled query points."""
    tri,_=triangles(vertices,faces);a,b,c=tri[:,0],tri[:,1],tri[:,2]
    ab,ac=b-a,c-a;n=np.cross(ab,ac);nn=np.sum(n*n,axis=1)
    d00=np.sum(ab*ab,1);d01=np.sum(ab*ac,1);d11=np.sum(ac*ac,1);den=d00*d11-d01*d01
    output=[]
    for start in range(0,len(points),64):
        p=np.asarray(points[start:start+64],float)[:,None,:];ap=p-a
        plane=np.sum(ap*n,2);projected=ap-plane[:,:,None]/nn[None,:,None]*n
        d20=np.sum(projected*ab,2);d21=np.sum(projected*ac,2)
        u=(d11*d20-d01*d21)/den;v=(d00*d21-d01*d20)/den
        inside=(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)
        best=np.full(inside.shape,np.inf)
        for x,y in [(a,b),(b,c),(c,a)]:
            edge=y-x;fraction=np.clip(np.sum((p-x)*edge,2)/np.sum(edge*edge,1),0,1)
            best=np.minimum(best,np.sum((p-x-fraction[:,:,None]*edge)**2,2))
        best=np.where(inside,plane*plane/nn,best)
        output.append(np.sqrt(np.maximum(0,best.min(1))))
    return np.concatenate(output)


def compare_meshes(pred_vertices,pred_faces,proxy_vertices,proxy_faces,proxy_rotation,samples=2048):
    p,pc=surface_samples(pred_vertices,pred_faces,samples);r,rc=surface_samples(proxy_vertices,proxy_faces,samples)
    a=point_mesh_distances(p,proxy_vertices,proxy_faces);b=point_mesh_distances(r,pred_vertices,pred_faces)
    pe=np.ptp(np.asarray(pred_vertices)@proxy_rotation,axis=0);re=np.ptp(np.asarray(proxy_vertices)@proxy_rotation,axis=0)
    return dict(surface_centroid_error_mm=float(np.linalg.norm(pc-rc)*1000),predicted_surface_centroid_m=pc.tolist(),proxy_surface_centroid_m=rc.tolist(),
        proxy_axis_predicted_extents_mm=(pe*1000).tolist(),proxy_axis_reference_extents_mm=(re*1000).tolist(),
        proxy_axis_extent_MAE_mm=float(abs(pe-re).mean()*1000),
        predicted_to_proxy_surface_mean_mm=float(a.mean()*1000),proxy_to_predicted_surface_mean_mm=float(b.mean()*1000),
        symmetric_surface_mean_mm=float((a.mean()+b.mean())*500),symmetric_surface_p90_mm=float((np.quantile(a,.9)+np.quantile(b,.9))*500),
        surface_samples_per_mesh=samples,
        surface_method='Deterministic area-uniform query samples; exact nearest triangle per query; no alignment',
        extent_interpretation='Extents in proxy object axes; pose orientation error contributes, no fitted OBB alignment')


def score(*,sealed_suite,proxy_bundle,output,object_id='block'):
    seal,runs=verify_seals(sealed_suite)
    proxy_bundle=Path(proxy_bundle);receipt=read(proxy_bundle/'export-receipt.json')
    if (receipt['status']!='complete' or receipt['suite_sha256']!=seal['suite_sha256']
            or receipt['scene_sha256_by_run']!={r['id']:r['scene_sha256']for r in runs}
            or receipt['proxy_sha256']!=file_sha256(proxy_bundle/'initial-proxy.npz')):
        raise ValueError('Proxy export is not bound to the same four frozen scenes')
    stable={str(proxy_bundle/name):file_sha256(proxy_bundle/name)for name in ['export-receipt.json','initial-proxy.npz']}
    with np.load(proxy_bundle/'initial-proxy.npz',allow_pickle=False) as z:
        pv,pf,pose=[z[k].copy()for k in ['vertices','faces','pose']]
    world=pv@pose[:3,:3].T+pose[:3,3];rows=[]
    for run in runs:
        scene=read(run['scene_path']);validate_scene(scene,require_metric_scale=True)
        if scene['coordinate_frame']['name']!='source_camera' or not np.isclose(scene['timestamps'][0],receipt['method_initial_time_s'],atol=1e-9,rtol=0):
            raise ValueError('Scene initial timestamp/gauge differs from the source image')
        objects=[o for o in scene['objects']if o['id']==object_id]
        if len(objects)!=1 or not objects[0]['observed'][0]:raise ValueError('Initial task object must be explicitly supported')
        obj=objects[0]
        if obj['geometry'].get('uri') is not None:raise ValueError('External unsealed mesh URI is unsupported')
        v,f=geometry_arrays(obj['geometry']);T=np.asarray(obj['T_world_object'])[0]
        v=v@T[:3,:3].T+T[:3,3]
        rows.append(dict(run_id=run['id'],scene_sha256=run['scene_sha256'],**compare_meshes(v,f,world,pf,pose[:3,:3])))
    verify_seals(sealed_suite)
    if any(file_sha256(p)!=h for p,h in stable.items()):raise ValueError('Proxy changed during scoring')
    report=dict(status='complete',evaluation_only=True,reference_kind=receipt['reference_kind'],suite_sha256=seal['suite_sha256'],
        alignment_used=False,scale_fit=False,future_motion_scored=False,simulator_used=False,rows=rows,
        proxy_export_receipt_sha256=stable[str(proxy_bundle/'export-receipt.json')],producer_sha256=file_sha256(__file__),
        limitation='Recorded RGB-D reconstruction proxy and supplied camera can be inaccurate; these are not exact GT geometry or execution scores.')
    write(output,report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    package=sub.add_parser('package-seals');package.add_argument('--sealed-suite',type=Path,required=True);package.add_argument('--output',type=Path,required=True)
    export=sub.add_parser('export')
    for name in ['sealed-suite','observed-manifest','raw-reference','source-cameras','output']:export.add_argument('--'+name,type=Path,required=True)
    export.add_argument('--source-view',type=int,required=True);export.add_argument('--source-camera-id',required=True)
    metric=sub.add_parser('score')
    for name in ['sealed-suite','proxy-bundle','output']:metric.add_argument('--'+name,type=Path,required=True)
    metric.add_argument('--object-id',default='block')
    args=vars(parser.parse_args());command=args.pop('command')
    result=package_seals(args.pop('sealed_suite'),args.pop('output')) if command=='package-seals' else (export_proxy(**args)if command=='export'else score(**args))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
