"""One-frame mesh seed for camera design, not a recovered object motion track."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from artirecon.input_protocol import load_manifest,validate_manifest,file_sha256
from artirecon.scene_sequence import scene_from_arrays,object_from_arrays
from rgb_mano_initialize import merge_manifests


def prepare(geometry,asset,output,object_id='bottle'):
    import trimesh
    from PIL import Image
    geometry,asset,output=map(lambda p:Path(p).resolve(),[geometry,asset,output])
    if output.exists():raise FileExistsError(output)
    load_manifest(geometry/'input-manifest.json',[('depth',geometry/'geometry.npz'),('cameras',geometry/'geometry.npz')])
    load_manifest(asset/'input-manifest.json',[('object_geometry',asset/'object.ply'),
        ('object_geometry',asset/'object.json'),('masks',asset/'mask.png')])
    manifest=merge_manifests([geometry/'input-manifest.json',asset/'input-manifest.json'])
    roots=[a for a in manifest['artifacts'] if a['origin']=='observed_rgb']
    if len(roots)!=1:raise ValueError('Asset and geometry must use identical observed RGB')
    with np.load(geometry/'geometry.npz',allow_pickle=False) as z:
        d=z['depth_m'][0,0].copy();k=z['K'][0].copy();t=float(z['timestamps'][0]);rgb=z['rgb'][0,0].copy()
        if not np.allclose(z['c2w'][0],np.eye(4)):raise ValueError('Use source-camera gauge')
    if not np.array_equal(rgb,np.asarray(Image.open(asset/'source.png').convert('RGB'))):
        raise ValueError('SAM3D source image differs from geometry frame0')
    mask=np.asarray(Image.open(asset/'mask.png').convert('L'))>0
    if mask.shape!=d.shape:raise ValueError('Mask/depth grid mismatch')
    y,x=np.where(mask&np.isfinite(d)&(d>0))
    if len(x)<32:raise ValueError('Insufficient model depth on first-frame object')
    depth=float(np.median(d[y,x]));uv=np.median(np.c_[x,y],axis=0)
    target=np.r_[uv,1.]@np.linalg.inv(k).T*depth
    pose=json.loads((asset/'object.json').read_text());T=np.array(pose['T_obj2cam_opencv'],float)
    mesh=trimesh.load(asset/'object.ply',process=False)
    v=np.asarray(mesh.vertices)
    cam=v@T[:3,:3].T+T[:3,3]
    source_ratio=depth/float(np.median(cam[:,2]))
    if not np.isfinite(source_ratio) or not .2<source_ratio<5:raise ValueError('Implausible learned-depth/SAM3D scale ratio')
    T[:3,:3]*=source_ratio
    T[:3,3]=target-np.median(v@T[:3,:3].T,axis=0)
    provenance=dict(method='One RGB SAM3D mesh, source UniDepth foreground range and SAM3 mask centroid',
        artifact_ids=['sam3d_canonical_mesh','sam3d_pose','selected_object_mask','predicted_geometry','predicted_cameras'],
        confidence_status='uncalibrated')
    scene=scene_from_arrays(timestamps=[t],coordinate_frame=dict(name='source_camera',handedness='right',up_axis='unknown',pose_convention='T_world_local'),
        metric_scale=dict(status='estimated',provenance=provenance),actors=[],
        objects=[object_from_arrays(object_id=object_id,poses=T[None],observed=[True],provenance=provenance,mesh_uri=str(asset/'object.ply'))],
        support=None,contacts=[],uncertainty=dict(status='estimated',provenance=provenance,
            interpretation='Coarse foreground range/centroid seed; no known contact, support, hidden geometry or motion'),
        task=dict(description='Design helper-view cameras around the observed pump bottle',target=None))
    scene['motion_model']='single_initial_pose_only_no_track'
    output.mkdir(parents=True)
    (output/'scene-sequence.json').write_text(json.dumps(scene,indent=2,allow_nan=False))
    manifest['artifacts'].append(dict(id='source_mesh_seed',kind='object_geometry',origin='model_prediction',
        parent_ids=provenance['artifact_ids'],path='scene-sequence.json',sha256=file_sha256(output/'scene-sequence.json')))
    manifest['method_inputs']=['source_mesh_seed']
    manifest['metric_scale']=dict(status='estimated',artifact_ids=['predicted_geometry'],description='Learned UniDepth foreground range, no measured scale')
    manifest['robot_registration']=dict(status='unresolved',artifact_ids=[],description='Source camera not registered to robot')
    validation=validate_manifest(manifest,output,verify_files=True)
    (output/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    audit=dict(status='complete',observed_frames=1,source_sequence_has_future_track=False,
        source_depth_median_m=depth,source_mask_centroid_uv=uv.tolist(),target_source_camera_m=target.tolist(),
        scale_correction_to_model_pose=source_ratio,known_support_used=False,GT_geometry_used=False,
        completion_meaning='Coarse camera-design initialization, not object tracking or physical reconstruction accuracy',validation=validation)
    (output/'execution.json').write_text(json.dumps(audit,indent=2));return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['geometry','asset','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.geometry,a.asset,a.output),indent=2))
