"""Amani object fitting with frozen MANO states, full budgets and explicit arms.

assisted: prior masks + primitive; sam3: actual SAM3 + primitive;
sam3d: actual SAM3 + SAM3D competing with primitive;
sam3d_forced: same, diagnostic forcing the generated mesh.
Default grasp adapter is frozen for causal comparisons. --grasp upstream adds
Amani's original state machine with MANO aperture/frame replacing Panda's.
"""
import argparse
import copy
import json
import os
import pickle
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


class AssistedMasks:
    def __init__(self,m,prompts):
        self.m=m;self.prompts=prompts
        self.score=np.ones(m.shape[0]*m.shape[1])
        self.box=np.zeros((len(self.score),4))
    def mask(self,i,raw=False):return self.m.reshape(-1,*self.m.shape[-2:])[i]
    def ids(self,p,t,v):return [t*self.m.shape[1]+v] if self.m[t,v].any() else []


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--case',choices=['button','block','bottle'],required=True)
    ap.add_argument('--arm',choices=['assisted','sam3','sam3d','sam3d_forced'],required=True)
    ap.add_argument('--grasp',choices=['legacy','upstream'],default='legacy')
    ap.add_argument('--tracked',action='store_true',help='Merge actual SAM3 video propagation masks after preliminary fitting')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--gate',type=float,help='Explicit diagnostic override of the object re-detection radius in metres')
    ap.add_argument('--track-only',action='store_true',help='Inspect source mask association without fitting geometry')
    args=ap.parse_args();run=args.run.resolve()
    os.environ['AMANI_DEVICE']=args.device;os.environ['AMANI_GRASP']=args.grasp
    os.environ['AMANI_FORCE_MESH']='1' if args.arm=='sam3d_forced' else '0'
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    sys.path.insert(0,str(run/'runtime'))
    from amani_comparison.objects_v2 import ObjectsV2,Sam3Masks
    case=run/'inputs'/args.case
    protocol=json.loads((case/'protocol.json').read_text());ids=np.array(protocol['keyframe_ids'])
    ann=json.loads((case/'ann.json').read_text());ann=copy.deepcopy(ann)
    z=np.load(case/'obs.npz');D=z['depth_m'][ids]
    C=z['c2w']
    if (case/'fit-calibration.npz').exists():
        calibration=np.load(case/'fit-calibration.npz')
        D=D*calibration['total_scale'][ids,:,None,None].astype(np.float32);C=calibration['c2w']
    obs=SimpleNamespace(depth=D,K=z['K'],c2w=C,T=len(ids),V=D.shape[1])
    assert obs.V==len(protocol['train_cameras'])
    output=run/'results'/(f'{args.arm}-{args.grasp}'+('-tracked' if args.tracked else ''))/args.case
    if args.gate is not None:output=output.parent.parent/(output.parent.name+f'-gate{args.gate:g}')/args.case
    output.mkdir(parents=True,exist_ok=True)
    ob=ann['objects'][0]
    if args.gate is not None:ob['gate']=args.gate
    full_t=ob['hint']['t']
    if full_t not in ids:raise ValueError('Hint must be an evaluated keyframe')
    ob['hint']['t']=int(np.flatnonzero(ids==full_t)[0])
    if ob['hint']['uv'] is None:raise ValueError('Unreviewed annotation click')
    if args.arm=='assisted':
        ctrl=np.load(run/'controls'/args.case/'assisted-masks.npz')
        assert np.array_equal(ctrl['frame_ids'],ids)
        masks=AssistedMasks(ctrl['mask'],ann['prompts'])
    else:
        execution=json.loads((case/'sam3-execution.json').read_text())
        assert execution['status']=='complete','SAM3 did not complete'
        s=np.load(case/'sam3.npz');keep=np.isin(s['t'],ids)
        remap={int(t):i for i,t in enumerate(ids)}
        payload={k:s[k][keep] for k in ['p','t','v','score','box','mask']}
        payload['t']=np.array([remap[int(t)] for t in payload['t']],dtype=int)
        payload.update(prompts=s['prompts'],shape=s['shape'])
        np.savez_compressed(output/'sam3-keyframes.npz',**payload)
        mask_paths=[output/'sam3-keyframes.npz']
        if args.tracked:
            source=run/'results'/f'{args.arm}-{args.grasp}'/args.case
            assert json.loads((source/'sam3-track-execution.json').read_text())['status']=='complete'
            mask_paths.append(source/'sam3_track.npz')
        masks=Sam3Masks(mask_paths,*D.shape[-2:])
    if args.arm.startswith('sam3d'):
        assert json.loads((case/'sam3d-execution.json').read_text())['status']=='complete'
        ob.update(shape='mesh',mesh=str(case/'sam3d'/f"{ob['name']}_20k.ply"),mesh_json=str(case/'sam3d'/f"{ob['name']}.json"))
        assert Path(ob['mesh']).is_file() and Path(ob['mesh_json']).is_file()
    else:ob['shape']=ob['alt_shape']
    h=np.load(case/'frozen-hand.npz')
    rob={k:h[k][ids] for k in ['q','base','hand_frames','pinch_points','pinch_width']}
    if 'visible' in h:rob['visible']=h['visible'][ids]
    class Hand:
        def surface(self,q,base):
            return (torch.as_tensor(h['points'][ids],device=q.device),torch.as_tensor(h['normals'][ids],device=q.device))
    table=protocol['table'];prep=dict(n=table['normal'],d=table['offset'])
    t0=time.monotonic();log=[]
    def logger(s):log.append(s);print(s,flush=True)
    fitter=ObjectsV2(obs,prep,Hand(),rob,masks,ann,log=logger)
    if args.track_only:
        tracks=[]
        for spec in ann['objects']:
            selected,centres,held=fitter.track(spec)
            tracks.append(dict(name=spec['name'],views_per_frame=[len(s) for s in selected],
                               depth_views_per_frame=[sum(c['c'] is not None for c in s.values()) for s in selected],
                               centres=centres.tolist(),held=held.tolist(),gate=spec.get('gate',.05)))
        (output/'association-diagnostic.json').write_text(json.dumps(tracks,indent=2))
        print([(x['name'],x['views_per_frame']) for x in tracks],flush=True)
        return
    initial_candidates=[]
    original_shape=fitter.shape
    def record_shape(spec,sel,cen,held):
        result=original_shape(spec,sel,cen,held)
        pose,half,mesh,rest=result
        initial_candidates.append(dict(name=spec['name'],shape=spec.get('shape','box'),
                    pose=np.asarray(pose).tolist(),half_extents=np.asarray(half).tolist(),
                    vertices=np.asarray(mesh.vertices).tolist(),faces=np.asarray(mesh.faces).tolist(),
                    rest_keyframes=[int(x) for x in rest],source_frame=int(ids[rest[0]])))
        return result
    fitter.shape=record_shape
    fit=fitter.run()
    (output/'initial-candidates.json').write_text(json.dumps(initial_candidates))
    if len(fit)!=len(ann['objects']):raise RuntimeError('Missing object; refusing to mark successful')
    with open(output/'objects-fit.pkl','wb') as f:pickle.dump(fit,f)
    scene=dict(schema='mano-scene-0.1',source='Amani ObjectsV2 MANO adaptation',
               frame_ids=ids.tolist(),timestamps=(ids/protocol['fps']).tolist(),table=table,
               hand_fit=protocol['hand_fit_sha256'],objects=[])
    for o in fit:
        scene['objects'].append(dict(name=o['name'],type=o['type'],vertices=o['mesh'][0].tolist(),faces=o['mesh'][1].tolist(),
                                     poses=o['poses'].tolist(),held=o['held'].tolist(),half_extents=o['half'].tolist(),loss=o['loss'].tolist(),
                                     selected_shape=o['selected_shape'],candidate_losses=o['candidate_losses']))
    (output/'scene.json').write_text(json.dumps(scene))
    (output/'fit.log').write_text('\n'.join(log))
    (output/'execution.json').write_text(json.dumps(dict(status='complete',arm=args.arm,grasp=args.grasp,tracked=args.tracked,
             seconds=time.monotonic()-t0,device=args.device,fit_cameras=protocol['train_cameras'],
             heldout_camera=protocol['heldout_camera'],budgets=dict(surface_samples=1500,shape_iters=250,pose_iters=150),
             annotation=ann,protocol=protocol),indent=2))
    print('Completed',output,flush=True)


if __name__=='__main__':main()
