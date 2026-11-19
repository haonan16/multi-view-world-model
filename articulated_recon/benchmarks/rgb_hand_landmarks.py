"""Fresh MediaPipe landmarks from isolated RGB and optional learned SAM3 crops.

Preserve all hand candidates and missing detections. MediaPipe world landmarks
are RGB model predictions, not triangulation/GT. Handedness scores are NOT joint
visibility or calibrated position confidence. No previous hand fits are read.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from rgb_frontend import load_source_bundle, sha256
from artirecon.input_protocol import load_manifest, validate_manifest


def native_landmarks(landmarks, world, inverse, size):
    """Undo a crop/rotation in image and relative metric landmark coordinates."""
    xy=np.array([[p.x*size[0],p.y*size[1]] for p in landmarks],np.float32)
    xyz=np.array([[p.x,p.y,p.z] for p in world],np.float32)
    if xy.shape!=(21,2) or xyz.shape!=(21,3): raise ValueError('Expected21 landmarks')
    xy=xy@inverse[:,:2].T+inverse[:,2]
    rotation=inverse[:,:2]/np.sqrt(abs(np.linalg.det(inverse[:,:2])))
    xyz[:,:2]=xyz[:,:2]@rotation.T
    xyz-=xyz[:1]
    if not np.isfinite(xy).all() or not np.isfinite(xyz).all():raise ValueError('Invalid detector landmarks')
    return xy,xyz


def deduplicate(rows, threshold_px=10.):
    """Keep the first (native before crops) duplicate; no ranking by side score."""
    kept=[]
    for row in rows:
        if not any(np.linalg.norm(row['xy']-x['xy'],axis=1).mean()<threshold_px for x in kept):
            kept.append(row)
    return kept


def detect(bundle, masks_folder, output, detector_model):
    import cv2
    import mediapipe as mp
    bundle, masks_folder, output=Path(bundle).resolve(),Path(masks_folder).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    manifest, source, rgb,timestamps=load_source_bundle(bundle)
    mask_path=masks_folder/'masks.npz'
    mask_report=load_manifest(masks_folder/'input-manifest.json',[('masks',mask_path)])
    if mask_report['condition']!='rgb_only':raise ValueError('SAM3 crop ancestry must be RGB-only')
    mask_manifest=json.loads((masks_folder/'input-manifest.json').read_text())
    active=set(mask_report['method_artifact_ids'])
    roots=[a for a in mask_manifest['artifacts'] if a['id'] in active and a['origin']=='observed_rgb']
    if len(roots)!=1 or roots[0]['sha256']!=source['sha256']:raise ValueError('Detector and SAM3 RGB differ')
    with np.load(mask_path,allow_pickle=False) as z:
        if not np.array_equal(z['timestamps'],timestamps):raise ValueError('SAM3 timing differs')
        prompt=list(z['prompts']).index('hand')
        rows=np.flatnonzero(z['p']==prompt)
        boxes,frames=z['box'][rows],z['t'][rows]
    opts=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(detector_model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2,
        min_hand_detection_confidence=.3,min_hand_presence_confidence=.3)
    output.mkdir(parents=True)
    receipt=dict(status='running',detector_version=mp.__version__,task_sha256=sha256(detector_model),
        task_path=str(Path(detector_model).resolve()),camera_calibration_used=False,depth_used=False,
        previous_hand_fit_used=False,world_landmarks='Learned relative3D from RGB, wrist centered; not GT/triangulation',
        score_meaning='Handedness classification score only, not joint visibility or localization confidence',
        candidate_identity='Independent per frame; no temporal hand identity imposed',
        crop_recipe='Native image, then all learned SAM3 hand boxes expanded1.5x and rotated0/90/180/270 at384px',
        code_sha256=sha256(__file__))
    all_rows=[];counts=[];seconds=[];started=time.monotonic()
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(opts) as detector:
            for t,image in enumerate(rgb):
                begin=time.monotonic();h,w=image.shape[:2]
                candidates=[(image,np.array([[1.,0,0],[0,1.,0]]),'native')]
                for b,box in enumerate(boxes[frames==t]):
                    centre=(box[:2]+box[2:])/2;radius=max(24.,float(np.max(box[2:]-box[:2]))*.75)
                    for angle in [0,90,180,270]:
                        M=cv2.getRotationMatrix2D(tuple(map(float,centre)),angle,384/(2*radius))
                        M[:,2]+=np.array([192,192])-centre
                        crop=cv2.warpAffine(image,M,(384,384),borderMode=cv2.BORDER_CONSTANT)
                        candidates.append((crop,cv2.invertAffineTransform(M),f'sam3_box{b}_rot{angle}'))
                found=[]
                for pixels,inverse,label in candidates:
                    result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(pixels)))
                    for lm,world,handed in zip(result.hand_landmarks,result.hand_world_landmarks,result.handedness):
                        xy,xyz=native_landmarks(lm,world,inverse,(pixels.shape[1],pixels.shape[0]))
                        inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                        if inside.sum()<12:continue
                        found.append(dict(xy=xy,world=xyz,side=handed[0].category_name,
                            handedness_score=float(handed[0].score),recipe=label,frame=t))
                unique=deduplicate(found);all_rows.extend(unique);counts.append(len(unique));seconds.append(time.monotonic()-begin)
                (output/'progress.json').write_text(json.dumps(dict(status='running',frames=len(rgb),processed_frames=t+1,last_candidates=len(unique))))
                print(f'RGB hand {output.name}: {t+1}/{len(rgb)}, candidates={len(unique)}',flush=True)
        if sha256(bundle/source['path'])!=source['sha256']:raise ValueError('RGB changed during detection')
        n=len(all_rows)
        np.savez_compressed(output/'landmarks.npz',xy=np.array([r['xy'] for r in all_rows],np.float32).reshape(n,21,2),
            world_relative=np.array([r['world'] for r in all_rows],np.float32).reshape(n,21,3),
            side=np.array([r['side'] for r in all_rows],dtype='U5'),
            handedness_score=np.array([r['handedness_score'] for r in all_rows],np.float32),
            frame=np.array([r['frame'] for r in all_rows],np.int64),recipe=np.array([r['recipe'] for r in all_rows],dtype='U40'),
            timestamps=timestamps,counts=np.array(counts),frame_seconds=np.array(seconds))
        manifest=copy.deepcopy(mask_manifest)
        for a in manifest['artifacts']:
            if 'path' in a:a['path']=str((masks_folder/a['path']).resolve())
        manifest['artifacts'].extend([
            dict(id='mediapipe_task',kind='model_prior',origin='prior',scope='global',prior_type='learned_model_weights',parent_ids=[],
                path=str(Path(detector_model).resolve()),sha256=sha256(detector_model),model_name='MediaPipe HandLandmarker',runtime=mp.__version__),
            dict(id='rgb_hand_landmarks',kind='hand_initialization',origin='model_prediction',parent_ids=['sam3_masks','mediapipe_task'],
                path='landmarks.npz',sha256=sha256(output/'landmarks.npz'))])
        manifest['method_inputs']=['rgb_hand_landmarks']
        receipt['validation']=validate_manifest(manifest,output,verify_files=True)
        (output/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
        receipt.update(status='complete',frames=len(rgb),detected_frames=int(np.count_nonzero(counts)),candidate_count=n,
            multi_candidate_frames=int(np.sum(np.array(counts)>1)),missing_frames=np.flatnonzero(np.array(counts)==0).tolist(),
            completion_meaning='Fresh RGB hand candidates, not fitted MANO or verified fingertip contact')
        (output/'progress.json').write_text(json.dumps(dict(status='complete',frames=len(rgb),processed_frames=len(rgb))))
    except Exception as error:
        receipt.update(status='failed',error=repr(error),processed_frames=len(counts));raise
    finally:
        receipt['seconds']=time.monotonic()-started
        (output/'execution.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',type=Path,required=True,help='List of bundle,masks,output records')
    p.add_argument('--detector-model',type=Path,required=True)
    a=p.parse_args()
    for job in json.loads(a.requests.read_text()):detect(job['bundle'],job['masks'],job['output'],a.detector_model)


if __name__=='__main__':main()
