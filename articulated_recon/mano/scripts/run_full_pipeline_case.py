"""Run recovered SAM3D -> direct MANO RobotFitter -> ObjectsV2 -> SAM3 tracker -> refit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,help='Validated provenance for actual prepared inputs, including masks and mesh')
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
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from artirecon.input_protocol import load_manifest, unverified_report, ProtocolError
    input_protocol=unverified_report()
    protocol_record=run/'results'/f'{a.case}-input-protocol.json'
    if a.input_manifest:
        if record.get('status')!='complete' or not mesh_path.is_file():
            raise ProtocolError('Complete SAM3D asset recovery before hashing inputs for a verified pipeline run')
        if (run/'enable-self-calibration.json').exists():
            raise ProtocolError('Declare self-calibration as a separate input stage before a verified pipeline run')
        if (case/'fit-calibration.npz').exists():
            raise ProtocolError('Legacy fit-calibration.npz would silently alter object inputs; apply calibration in declared preparation instead')
        # A manifest never relabels a pre-existing unverified object/tracker cache.
        pre=run/'results/sam3d-upstream'/a.case
        if pre.exists() and not protocol_record.exists():
            raise ProtocolError('Existing object outputs lack input provenance; use a fresh run directory')
        required=[('rgb',case/'obs.npz'),('depth',case/'obs.npz'),('cameras',case/'obs.npz'),
                  ('hand_initialization',case/'hand-input.npz'),('support',case/'protocol.json'),
                  ('masks',case/'ann.json'),('masks',case/'sam3.npz'),('object_geometry',mesh_path),
                  ('object_geometry',case/'sam3d'/(name+'.json'))]
        input_protocol=load_manifest(a.input_manifest,required)
        if protocol_record.exists():
            cached=json.loads(protocol_record.read_text())
            if cached.get('manifest_sha256')!=input_protocol['manifest_sha256']:
                raise ProtocolError('Existing pipeline has different input provenance; use a fresh run directory')
        protocol_record.parent.mkdir(parents=True,exist_ok=True)
        protocol_record.write_text(json.dumps(input_protocol,indent=2))
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
    if a.input_manifest:sys.argv.extend(['--input-manifest',str(a.input_manifest.resolve())])
    if a.models:sys.argv.extend(['--models',str(a.models)])
    if (run/'enable-self-calibration.json').exists():sys.argv.append('--calibrate')
    run_robotfitter_mano.main()
    import fit_sam_comparison
    sys.argv=['fit_sam_comparison','--run',str(run),'--case',a.case,'--arm','sam3d','--grasp','upstream','--device',a.device]
    if a.input_manifest:sys.argv.extend(['--input-manifest',str(a.input_manifest.resolve())])
    pre=run/'results/sam3d-upstream'/a.case
    if not (pre/'execution.json').exists():fit_sam_comparison.main()
    if a.through=='objects':return
    import run_sam_comparison_tracking
    sys.argv=['run_sam_comparison_tracking','--run',str(run),'--case',a.case]
    if not (pre/'sam3-track-execution.json').exists():run_sam_comparison_tracking.main()
    sys.argv=['fit_sam_comparison','--run',str(run),'--case',a.case,'--arm','sam3d','--grasp','upstream','--device',a.device,'--tracked']
    if a.input_manifest:sys.argv.extend(['--input-manifest',str(a.input_manifest.resolve())])
    fit_sam_comparison.main()
    (run/'results'/f'{a.case}-pipeline-complete.json').write_text(json.dumps(dict(status='complete',case=a.case,input_protocol=input_protocol)))


if __name__=='__main__':main()
