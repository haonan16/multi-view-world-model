"""Test generated-view parallax against requested cameras using source estimates.

This is a consistency diagnostic, not GT camera/geometry accuracy. Mutual image
matches are filtered by a generic background homography, never by the camera
hypothesis being tested. All hand/object candidate masks are excluded.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artirecon.input_protocol import file_sha256, load_manifest


def requested_pixels(points, depth, k_source, k_target, source_c2w, target_c2w):
    """Lift source pixels with predicted z-depth, then apply requested SE(3)."""
    points, depth = np.asarray(points, float), np.asarray(depth, float)
    source = np.c_[points, np.ones(len(points))] @ np.linalg.inv(k_source).T * depth[:, None]
    transform = np.linalg.inv(target_c2w) @ source_c2w
    target = source @ transform[:3, :3].T + transform[:3, 3]
    uvw = target @ k_target.T
    valid = np.isfinite(target).all(axis=1) & (target[:, 2] > .01) & np.isfinite(depth) & (depth > 0)
    projected = np.full((len(points), 2), np.nan)
    projected[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return projected, valid


def compare_displacements(source, matched, expected):
    observed, requested = matched - source, expected - source
    identity = np.linalg.norm(observed, axis=1)
    residual = np.linalg.norm(matched - expected, axis=1)
    magnitude = np.linalg.norm(requested, axis=1)
    significant = magnitude > 2.
    coefficients = np.sum(observed[significant] * requested[significant], axis=1) / magnitude[significant] ** 2
    return dict(matches=len(source), observed_displacement_median_px=float(np.median(identity)),
                requested_displacement_median_px=float(np.median(magnitude)),
                requested_reprojection_error_median_px=float(np.median(residual)),
                identity_is_closer_fraction=float(np.mean(identity < residual)),
                requested_displacement_over_2px=int(significant.sum()),
                requested_flow_fraction_median=float(np.median(coefficients)) if len(coefficients) else None)


def feature_matches(source, target, allowed):
    detector = cv2.SIFT_create(nfeatures=1500, contrastThreshold=.015)
    k0, d0 = detector.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), allowed.astype(np.uint8)*255)
    k1, d1 = detector.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), None)
    if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), dict(source_features=len(k0), target_features=len(k1), mutual_matches=0, homography_inliers=0)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio_matches(a, b):
        return {row[0].queryIdx: row[0].trainIdx for row in matcher.knnMatch(a, b, k=2)
                if len(row) == 2 and row[0].distance < .75 * row[1].distance}
    forward, backward = ratio_matches(d0, d1), ratio_matches(d1, d0)
    pairs = [(a, b) for a, b in forward.items() if backward.get(b) == a]
    a = np.asarray([k0[i].pt for i, _ in pairs], float).reshape(-1, 2)
    b = np.asarray([k1[j].pt for _, j in pairs], float).reshape(-1, 2)
    if len(a) >= 8:
        cv2.setRNGSeed(0)
        _, inliers = cv2.findHomography(a, b, cv2.RANSAC, 3.)
        selected = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(a), bool)
    else:
        selected = np.zeros(len(a), bool)
    return a[selected], b[selected], dict(source_features=len(k0), target_features=len(k1),
        mutual_matches=len(a), homography_inliers=int(selected.sum()))


def transform_grid(array, crop, size, *, mask=False):
    x, y, w, h = crop
    return cv2.resize(array[y:y+h, x:x+w].astype(np.float32), size,
                      interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR)


def run(*, generated, condition, geometry, masks, output):
    generated, condition, geometry, masks, output = map(lambda p: Path(p).resolve(), (generated, condition, geometry, masks, output))
    if output.exists(): raise FileExistsError(output)
    load_manifest(generated/'input-manifest.json', [('rgb', generated/'observations.npz')])
    load_manifest(condition/'input-manifest.json', [('cameras', condition/'condition.npz')])
    load_manifest(geometry.parent/'input-manifest.json', [('depth', geometry), ('cameras', geometry)])
    prep = json.loads((condition/'preparation.json').read_text())
    if Path(prep['geometry_manifest']).resolve() != geometry.parent/'input-manifest.json':
        raise ValueError('Diagnostic geometry differs from the camera-design source')
    scene_manifest = json.loads(Path(prep['scene_manifest']).read_text())
    mask_artifacts = [a for a in scene_manifest['artifacts'] if a['kind']=='masks' and
                      (Path(prep['scene_manifest']).parent/a.get('path', '')).resolve()==masks]
    if len(mask_artifacts)!=1 or mask_artifacts[0]['sha256']!=file_sha256(masks):
        raise ValueError('Use the exact source SAM3 evidence bound to the initializer')
    with np.load(generated/'observations.npz', allow_pickle=False) as z:
        rgb, ids, times, k, c, known = [z[name].copy() for name in ['rgb','source_frame_indices','timestamps','K','c2w','observed_mask']]
    with np.load(generated/'generated.npz', allow_pickle=False) as z:
        raw = z['rgb'].copy()
    with np.load(condition/'condition.npz', allow_pickle=False) as z:
        if not all(np.array_equal(z[name], value) for name, value in [('rgb',rgb[:,0]),('K',k),('c2w',c),('timestamps',times),('source_frame_indices',ids)]):
            raise ValueError('Helper pixels/camera requests do not match this condition')
    if known.shape!=rgb.shape[:2] or not known[:,0].all() or known[:,1:].any() or not np.array_equal(raw[:,1:],rgb[:,1:]):
        raise ValueError('Observed/generated view identity differs from the reviewed contract')
    with np.load(geometry, allow_pickle=False) as z:
        depth, valid, native_rgb, native_k, native_times = [z[name].copy() for name in ['depth_m','valid','rgb','K','timestamps']]
    if not np.array_equal(times,native_times[ids]): raise ValueError('Source timestamps differ')
    crop=prep['source_crop_xywh']; size=(rgb.shape[3],rgb.shape[2])
    x,y,w,h=crop; transformed_k=native_k[0].copy()
    transformed_k[0,2]-=x; transformed_k[1,2]-=y
    transformed_k[0]*=size[0]/w; transformed_k[1]*=size[1]/h
    if not np.allclose(transformed_k,k[0],atol=1e-6): raise ValueError('Predicted K crop/resize mismatch')
    with np.load(masks,allow_pickle=False) as z:
        shape=tuple(z['shape']); mask_t=z['t'].copy(); mask_p=z['p'].copy(); packed=z['mask'].copy(); prompts=z['prompts'].tolist()
        if not np.array_equal(z['timestamps'],native_times): raise ValueError('SAM3/source timestamps differ')
    dynamic_ids=[i for i,p in enumerate(prompts) if p=='hand' or 'block' in p.split() or p in {'object','cube'}]
    if 'hand' not in prompts or len(dynamic_ids)<2: raise ValueError('Hand/object masks are required for static-background exclusion')
    output.mkdir(parents=True)
    rows, pooled = [], {v:[] for v in range(4)}
    for frame,source_id in enumerate(ids):
        expected_source=cv2.resize(native_rgb[source_id,0,y:y+h,x:x+w],size,interpolation=cv2.INTER_AREA)
        if not np.array_equal(rgb[frame,0],expected_source): raise ValueError('View0 pixels differ from the original isolated RGB')
        dynamic=np.zeros(shape,bool)
        for row in np.flatnonzero((mask_t==source_id)&np.isin(mask_p,dynamic_ids)):
            dynamic|=np.unpackbits(packed[row])[:np.prod(shape)].reshape(shape).astype(bool)
        excluded=transform_grid(dynamic,crop,size,mask=True)>0
        excluded=cv2.dilate(excluded.astype(np.uint8),np.ones((9,9),np.uint8))>0
        d=transform_grid(depth[source_id,0],crop,size)
        allowed=(transform_grid(valid[source_id,0],crop,size,mask=True)>0)&~excluded&np.isfinite(d)&(d>0)
        panels=[]
        for view in range(4):
            target=raw[frame,view] if view==0 else rgb[frame,view]
            a,b,counts=feature_matches(rgb[frame,0],target,allowed)
            result=dict(frame=frame,source_frame=int(source_id),time_s=float(times[frame]),view=view,
                        role='raw decoded source VAE control' if view==0 else 'generated helper',**counts)
            display=cv2.resize(target,(640,320),interpolation=cv2.INTER_NEAREST)
            if len(a)>=8:
                sampled=cv2.remap(d,a[:,0].astype(np.float32)[None],a[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0]
                expected,good=requested_pixels(a,sampled,k[0],k[view],c[0],c[view])
                a,b,expected=a[good],b[good],expected[good]
                if len(a)>=8:
                    result.update(status='measured',**compare_displacements(a,b,expected))
                    inside=(expected[:,0]>=0)&(expected[:,0]<size[0])&(expected[:,1]>=0)&(expected[:,1]<size[1])
                    result['requested_outside_frame_fraction']=float(np.mean(~inside))
                    pooled[view].append((a,b,expected))
                    for observed,predicted in zip(b[::max(1,len(b)//70)],expected[::max(1,len(b)//70)]):
                        if np.max(abs(predicted))<2000:
                            cv2.line(display,tuple(np.rint(2*observed).astype(int)),tuple(np.rint(2*predicted).astype(int)),(230,90,190),1,cv2.LINE_AA)
                            cv2.circle(display,tuple(np.rint(2*observed).astype(int)),3,(80,240,100),-1)
            if 'status' not in result: result['status']='insufficient_static_matches'
            display[:55]=20
            label=f'VAE source control' if view==0 else f'Generated helper {view}'
            cv2.putText(display,f'{label} | t={times[frame]:.1f}s',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1,cv2.LINE_AA)
            text=(f"shift {result['observed_displacement_median_px']:.1f}px; request error {result['requested_reprojection_error_median_px']:.1f}px"
                  if result['status']=='measured' else 'Insufficient static matches')
            cv2.putText(display,text,(8,47),cv2.FONT_HERSHEY_SIMPLEX,.52,(245,245,245),1,cv2.LINE_AA)
            panels.append(display);rows.append(result)
        if frame in [0,len(ids)//2,len(ids)-1]:
            grid=np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0)
            cv2.imwrite(str(output/f'correspondences-{frame:03d}.png'),cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
    summaries={}
    for view,samples in pooled.items():
        if samples:
            a,b,p=[np.concatenate([item[i] for item in samples]) for i in range(3)]
            summaries[str(view)]=dict(measured_frames=len(samples),**compare_displacements(a,b,p))
    report=dict(schema='requested-camera-consistency-v1',method='Mutual SIFT ratio .75; generic homography RANSAC 3px; source hand/object masks excluded with four-pixel margin',
        generated=str(generated),source_condition_sha256=file_sha256(condition/'condition.npz'),
        generated_sha256=file_sha256(generated/'generated.npz'),geometry_sha256=file_sha256(geometry),masks_sha256=file_sha256(masks),
        producer_sha256=file_sha256(__file__),source_crop_xywh=crop,summaries=summaries,per_frame=rows,
        GT_used=False,measured_depth_used=False,simulation_used=False,
        reading='Flow fraction near 0 favors copying/source-like camera; near 1 follows requested parallax. Identity residual equals observed displacement. Magenta segments join measured green matches to their requested-camera projection.',
        limitations=['Estimated depth/K can be wrong; this measures internal consistency, not true camera accuracy.',
                     'Homography inliers favor planar static background; matches are correlated and may cover only part of the image.',
                     'Missing/generated features, occlusions and uncertain segmentation can bias correspondences.',
                     'No GT pose is estimated, and appearance similarity alone is not reconstruction quality.'])
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(summaries,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['generated','condition','geometry','masks','output']:parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
