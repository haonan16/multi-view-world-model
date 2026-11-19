"""Run actual SAM3/SAM3D through Amani's saved wrappers. No substitute masks.

SAM3 and SAM3D may use different Python environments. Run each stage using the
appropriate environment. The SAM3D root must contain Amani's sam3d_mesh.py and
repo/checkpoints/hf/pipeline.yaml. It is read, never modified.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_mask(case):
    ann=json.loads((case/'ann.json').read_text())
    ob=ann['objects'][0]
    t,v,uv=ob['hint']['t'],ob['hint']['v'],ob['hint']['uv']
    if uv is None:
        raise ValueError('RGB click has not been reviewed: '+str(case))
    z=np.load(case/'sam3.npz'); H,W=map(int,z['shape'])
    wanted=ob['prompt'] if isinstance(ob['prompt'],list) else [ob['prompt']]
    ids=np.flatnonzero(np.isin(z['p'],[list(z['prompts']).index(p) for p in wanted if p in z['prompts']])&(z['t']==t)&(z['v']==v))
    best=None
    for i in ids:
        m=np.unpackbits(z['mask'][i])[:H*W].reshape(H,W).astype(bool)
        if m[int(uv[1]),int(uv[0])] and (best is None or m.sum()<best[1].sum()):
            best=(int(i),m)
    if best is None:
        raise RuntimeError('SAM3 returned no instance under the reviewed RGB click: '+case.name)
    return ob,best


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--stage',choices=['sam3','sam3d'],required=True)
    ap.add_argument('--sam3d-root',type=Path)
    ap.add_argument('--case',choices=['all','button','block','bottle'],default='all')
    args=ap.parse_args(); run=args.run.resolve()
    failures=[]
    if args.stage=='sam3d':
        if args.sam3d_root is None:
            raise ValueError('--sam3d-root required')
        root=args.sam3d_root.resolve()
        assert (root/'repo/checkpoints/hf/pipeline.yaml').exists(), 'Missing genuine SAM3D checkpoint configuration'
        sys.path.insert(0,str(root))
        import sam3d_mesh
    for key in ['button','block','bottle']:
        if args.case not in ('all',key):continue
        case=run/'inputs'/key
        audit=case/(args.stage+'-execution.json')
        # A result is reusable only if the preceding execution record succeeded.
        if audit.exists() and json.loads(audit.read_text()).get('status')=='complete':
            print(key,args.stage,'already completed',flush=True);continue
        started=time.monotonic()
        record=dict(stage=args.stage,case=key,status='running',python=sys.executable,
                    input_sha256=sha(case/'obs.npz'),annotation_sha256=sha(case/'ann.json'))
        try:
            if args.stage=='sam3':
                script=run/'upstream/seg_sam3.py'
                record['wrapper_sha256']=sha(script)
                record['packages']={p:importlib.metadata.version(p) for p in ['torch','transformers','huggingface_hub']}
                prompts=json.loads((case/'ann.json').read_text())['prompts']
                subprocess.run([sys.executable,str(script),str(case),*prompts],check=True)
                from huggingface_hub import hf_hub_download
                cfg=Path(hf_hub_download('facebook/sam3','config.json',local_files_only=True))
                record['model_revision']=cfg.parent.name
                z=np.load(case/'sam3.npz')
                record['instances']=len(z['p'])
                record['mask_sha256']=sha(case/'sam3.npz')
                # Reject a successful script exit with no usable target mask.
                choose_mask(case)
            else:
                ob,(i,mask)=choose_mask(case)
                d=case/'sam3d';d.mkdir(exist_ok=True)
                stem=d/ob['name'];ip=stem.with_suffix('.png');mp=d/(ob['name']+'_mask.png')
                z=np.load(case/'obs.npz')
                Image.fromarray(z['rgb'][ob['hint']['t'],ob['hint']['v']]).save(ip)
                Image.fromarray(mask.astype(np.uint8)*255).save(mp)
                record.update(wrapper_sha256=sha(root/'sam3d_mesh.py'),pipeline_config_sha256=sha(root/'repo/checkpoints/hf/pipeline.yaml'),
                              selected_mask_index=i,mask_sha256=sha(case/'sam3.npz'),seed=0,
                              depth_input='Amani default: SAM3D internal MoGe; measured multi-view depth used in subsequent metric fit')
                sys.argv=['sam3d_mesh.py','--image',str(ip),'--mask',str(mp),'--out',str(stem.with_suffix('.ply')),'--seed','0','--no_preview']
                sam3d_mesh.main()
                import trimesh
                mesh=trimesh.load(stem.with_suffix('.ply'),force='mesh')
                record['original_faces']=len(mesh.faces)
                if len(mesh.faces)>20000:
                    mesh=mesh.simplify_quadric_decimation(face_count=20000)
                mesh.export(d/(ob['name']+'_20k.ply'))
                record.update(faces=len(mesh.faces),mesh_sha256=sha(d/(ob['name']+'_20k.ply')))
            record['status']='complete'
        except Exception as ex:
            record.update(status='failed',error=repr(ex));failures.append(key)
            print(key,args.stage,'FAILED',repr(ex),flush=True)
        record['seconds']=time.monotonic()-started
        audit.write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
    if failures:
        raise SystemExit('Failed '+args.stage+': '+', '.join(failures))


if __name__=='__main__':main()
