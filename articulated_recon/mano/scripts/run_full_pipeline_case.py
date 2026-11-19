"""Run recovered SAM3D -> direct MANO RobotFitter -> ObjectsV2 -> SAM3 tracker -> refit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',choices=['button','block','bottle'],required=True)
    p.add_argument('--models',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--through',choices=['assets','objects','complete'],default='complete',
                   help='assets only recovers the mesh; objects stops after preliminary fitting; neither claims tracker completion')
    a=p.parse_args();run=a.run.resolve();case=run/'inputs'/a.case
    import trimesh
    record=json.loads((case/'sam3d-execution.json').read_text())
    ann=json.loads((case/'ann.json').read_text());name=ann['objects'][0]['name']
    mesh_path=case/'sam3d'/(name+'_20k.ply')
    if record['status']!='complete' or not mesh_path.exists():
        if record['status']!='complete' and 'fast_simplification' not in record.get('error',''):
            raise RuntimeError('Unexpected SAM3D failure; cannot recover as simplification only')
        (case/'sam3d-execution-before-recovery.json').write_text(json.dumps(record,indent=2))
        stem=case/'sam3d'/name
        assert stem.with_suffix('.json').exists()
        mesh=trimesh.load(stem.with_suffix('.ply'),force='mesh')
        assert len(mesh.faces)==record['original_faces']
        start=time.monotonic()
        if len(mesh.faces)>20000:mesh=mesh.simplify_quadric_decimation(face_count=20000)
        path=case/'sam3d'/(name+'_20k.ply');mesh.export(path)
        record.update(status='complete',faces=len(mesh.faces),mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      recovery='Simplified existing successful SAM3D mesh using the installed decimation backend. No inference rerun.',
                      recovery_seconds=time.monotonic()-start)
        record.pop('error',None);(case/'sam3d-execution.json').write_text(json.dumps(record,indent=2))
        print('Recovered SAM3D mesh',a.case,flush=True)
    if a.through=='assets':return
    import run_robotfitter_mano
    sys.argv=['run_robotfitter_mano','--run',str(run),'--case',a.case,'--device',a.device]
    if a.models:sys.argv.extend(['--models',str(a.models)])
    if (run/'enable-self-calibration.json').exists():sys.argv.append('--calibrate')
    run_robotfitter_mano.main()
    import fit_sam_comparison
    sys.argv=['fit_sam_comparison','--run',str(run),'--case',a.case,'--arm','sam3d','--grasp','upstream','--device',a.device]
    pre=run/'results/sam3d-upstream'/a.case
    if not (pre/'execution.json').exists():fit_sam_comparison.main()
    if a.through=='objects':return
    import run_sam_comparison_tracking
    sys.argv=['run_sam_comparison_tracking','--run',str(run),'--case',a.case]
    if not (pre/'sam3-track-execution.json').exists():run_sam_comparison_tracking.main()
    sys.argv=['fit_sam_comparison','--run',str(run),'--case',a.case,'--arm','sam3d','--grasp','upstream','--device',a.device,'--tracked']
    fit_sam_comparison.main()
    (run/'results'/f'{a.case}-pipeline-complete.json').write_text(json.dumps(dict(status='complete',case=a.case)))


if __name__=='__main__':main()
