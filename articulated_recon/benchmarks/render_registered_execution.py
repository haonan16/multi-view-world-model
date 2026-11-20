"""Render completed registered-plan execution logs without rerunning physics.

Every frame uses recorded robot qpos and object_pose, at the native 20 Hz.
The fixed evaluator support plane retains its actual tilt and offset. No fitted
trajectory, target animation, object reset or simulated interpolation is shown.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'mano')]
from artirecon.input_protocol import file_sha256
from benchmarks.execute_registered import _pose, load_evaluator
from benchmarks.freeze_submission import verify_submission


DEFAULT_URDF = Path('/n/lab_storage/ydu_lab/akiruga/.conda/envs/maniskill/lib/python3.10/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf')


def origin(node):
    value = np.eye(4)
    if node is not None:
        value[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        value[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return value


def read_robot(urdf):
    urdf = Path(urdf).resolve()
    xml = ET.parse(urdf).getroot()
    joints = {}
    for node in xml.findall('joint'):
        axis = node.find('axis')
        joints[node.find('child').get('link')] = dict(parent=node.find('parent').get('link'),
            name=node.get('name'), kind=node.get('type'), origin=origin(node.find('origin')),
            axis=np.fromstring(axis.get('xyz') if axis is not None else '0 0 1', sep=' '))
    return SimpleNamespace(path=urdf, xml=xml, joints=joints)


def validate_records(records, control_hz=20):
    if control_hz != 20 or not isinstance(records, list) or not records:
        raise ValueError('Require nonempty completed 20 Hz execution logs')
    times = np.asarray([r['time_s'] for r in records], float)
    expected = np.arange(1, len(records)+1)/control_hz
    if not np.isfinite(times).all() or not np.allclose(times, expected, atol=1e-7, rtol=0):
        raise ValueError('Native timestamps must contain every consecutive 20 Hz control step')
    for index, row in enumerate(records):
        if row['step'] != index:
            raise ValueError('Native execution log has missing or reordered steps')
        qpos = np.asarray(row['qpos'], float)
        if qpos.shape != (9,) or not np.isfinite(qpos).all():
            raise ValueError('Recorded Panda joint vector must be finite [9]')
        _pose(row['object_pose'], 'recorded object_pose')
        _pose(row['tcp'], 'recorded tcp')
        if not isinstance(row['phase'], str) or not row['phase']:
            raise ValueError('Recorded phase is missing')
    return times


def support_mesh(normal, offset, *, extent=1.2, cells=40):
    """Visualize the exact declared plane n·x+d=0, including its offset."""
    normal = np.asarray(normal, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset):
        raise ValueError('Invalid support plane')
    length = np.linalg.norm(normal)
    if length <= 1e-10:
        raise ValueError('Support normal is zero')
    normal, offset = normal/length, float(offset)/length
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = axis-normal*(normal@axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    line = np.linspace(-extent, extent, cells+1)
    xx, yy = np.meshgrid(line, line)
    vertices = -offset*normal + xx.ravel()[:, None]*u + yy.ravel()[:, None]*v
    faces = []
    for y in range(cells):
        for x in range(cells):
            a = y*(cells+1)+x
            faces.extend([[a,a+1,a+cells+2],[a,a+cells+2,a+cells+1]])
    return vertices, np.asarray(faces, np.int32)


def camera(view='overview'):
    eye, target, focal = np.array([.93,-.76,.62]), np.array([.42,0.,.12]), 560.
    if view == 'contact':
        eye, target, focal = np.array([.78,-.53,.46]), np.array([.45,0.,.07]), 650.
    forward = target-eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.,0.,1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c = np.eye(4); c[:3,:3] = np.stack([right,down,forward], axis=1); c[:3,3] = eye
    return np.array([[focal,0.,320.],[0.,focal,240.],[0.,0.,1.]]), c


def load_completed(*, submission, evaluation_receipt, execution, urdf):
    submission, evaluation_receipt, execution, urdf = map(lambda p:Path(p).resolve(),
        (submission,evaluation_receipt,execution,urdf))
    frozen = verify_submission(submission)
    config = json.loads(Path(frozen['files']['controller_config']['path']).read_text())
    evaluator = load_evaluator(submission,evaluation_receipt,config)
    summary = json.loads((execution/'summary.json').read_text())
    if (summary.get('status') != 'complete' or summary.get('physics_executed') is not True
            or summary.get('control_hz') != 20 or summary.get('submission_sha256') != file_sha256(submission)):
        raise ValueError('Only completed native execution of this exact seal can be replayed')
    for name, key in [('records.json','records_sha256'),('actions.npy','actions_sha256')]:
        if file_sha256(execution/name) != summary.get(key):
            raise ValueError('Native execution output changed: '+name)
    inputs = json.loads((execution/'execution-inputs.json').read_text())
    claim = inputs['claim']
    if (claim['submission'] != dict(path=str(submission),sha256=file_sha256(submission))
            or claim['evaluation_attachment'] != dict(path=str(evaluation_receipt),sha256=file_sha256(evaluation_receipt))
            or claim['output'] != str(execution)):
        raise ValueError('Execution claim does not bind this seal, evaluator and output')
    if json.loads(Path(frozen['execution_receipt_path']).read_text()) != claim:
        raise ValueError('Execution claim differs from the exclusive sealed receipt')
    if config != inputs['controller_config'] or file_sha256(urdf) != config['robot_urdf_sha256']:
        raise ValueError('Robot/config differs from the frozen native execution')
    records = json.loads((execution/'records.json').read_text())
    validate_records(records,summary['control_hz'])
    if summary['steps'] != len(records):
        raise ValueError('Execution record count differs from its summary')
    return frozen, config, evaluator, summary, records


def run(*, submission, evaluation_receipt, execution, output, urdf=DEFAULT_URDF, view='overview'):
    """One saved record per video frame; this function never constructs an env."""
    import cv2
    import imageio.v2 as imageio
    import trimesh
    from benchmarks.render_legacy_physics import link_poses, robot_meshes
    from mano_recon.render_fast import render
    output = Path(output).resolve()
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    frozen, config, evaluator, summary, records = load_completed(submission=submission,
        evaluation_receipt=evaluation_receipt,execution=execution,urdf=urdf)
    robot = read_robot(urdf)
    shapes = robot_meshes(robot,origin)
    base = np.asarray(evaluator['config']['robot']['base_pose'])
    all_poses = [link_poses(robot,row['qpos'],base) for row in records]
    fk_error = max(float(np.linalg.norm(poses['panda_hand_tcp'][:3,3]-np.asarray(row['tcp'])[:3,3]))
                   for poses,row in zip(all_poses,records))
    if fk_error > .002:
        raise ValueError('Offline robot FK disagrees with native logged TCP by more than 2 mm')
    collision_hashes = {node.get('filename'):file_sha256(robot.path.parent/node.get('filename'))
                        for node in robot.xml.findall('link/collision/geometry/mesh')}
    mesh = trimesh.load(evaluator['mesh'],process=False,force='mesh')
    plane = evaluator['config']['support']
    vertices, faces = support_mesh(plane['normal'],plane['offset'])
    k,c = camera(view)
    background,background_z = render(vertices,faces,k,c,size=(480,640),color=(185,181,171))
    opacity = .35 if view == 'contact' else 1.
    heights = np.asarray([np.asarray(row['object_pose'])[:3,3]@np.asarray(plane['normal']) for row in records])
    still_indices = {0,len(records)-1,int(np.argmax(heights))}
    output.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(output,fps=20,codec='libx264',quality=8,macro_block_size=16) as writer:
        for index,(row,poses) in enumerate(zip(records,all_poses)):
            canvas,depth = background.copy(),background_z.copy()
            def add(points,triangles,transform,color):
                image,z = render(points@transform[:3,:3].T+transform[:3,3],triangles,k,c,size=(480,640),color=color)
                visible = (z>0)&((depth<=0)|(z<depth))
                canvas[visible],depth[visible] = image[visible],z[visible]
            add(np.asarray(mesh.vertices),np.asarray(mesh.faces),np.asarray(row['object_pose']),(225,155,45))
            object_canvas,object_depth = canvas.copy(),depth.copy()
            for name,points,triangles in shapes:
                add(points,triangles,poses[name],(190,201,215))
            if opacity < 1:
                robot_pixels = (depth>0)&((object_depth<=0)|(depth<object_depth))
                canvas[robot_pixels] = (opacity*canvas[robot_pixels]+(1-opacity)*object_canvas[robot_pixels]).astype(np.uint8)
            canvas[:31] = 25; canvas[-30:] = 25
            title = 'Native execution' + (' | translucent robot' if opacity<1 else '')
            cv2.putText(canvas,title,(10,22),0,.58,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(canvas,f"{row['time_s']:.2f}s | {row['phase'].replace('_',' ')}",(10,470),0,.48,(245,245,245),1,cv2.LINE_AA)
            writer.append_data(canvas)
            if index in still_indices:
                imageio.imwrite(output.with_name(output.stem+f'-frame-{index:04d}.png'),canvas)
    # Recheck all frozen inputs and completed log hashes after rendering.
    load_completed(submission=submission,evaluation_receipt=evaluation_receipt,execution=execution,urdf=urdf)
    for name,digest in collision_hashes.items():
        if file_sha256(robot.path.parent/name) != digest:
            raise ValueError('Robot collision mesh changed during rendering')
    metadata = dict(status='complete',kind='offline_replay_of_logged_native_execution',
        physics_rerun=False,simulator_imported=False,recorded_states_changed=False,
        object_motion_source='Native records.json object_pose; no reconstructed or target trajectory is animated',
        fps=20,frames=len(records),first_time_s=records[0]['time_s'],last_time_s=records[-1]['time_s'],
        view=view,robot_opacity=opacity,camera_K=k.tolist(),camera_c2w=c.tolist(),
        support_plane=plane,support_replaced_by_z0=False,target_pose_drawn=False,
        source_submission_sha256=file_sha256(submission),source_summary_sha256=file_sha256(Path(execution)/'summary.json'),
        source_records_sha256=summary['records_sha256'],source_actions_sha256=summary['actions_sha256'],
        source_reference_mesh_sha256=file_sha256(evaluator['mesh']),urdf_sha256=file_sha256(urdf),
        render_time_robot_collision_mesh_sha256=collision_hashes,max_fk_translation_error_mm=1000*fk_error,
        script_sha256=file_sha256(__file__),raster_source_sha256=file_sha256(Path(__file__).parents[1]/'mano/mano_recon/raster_cpu.cpp'),
        video_sha256=file_sha256(output),limitation='Offline collision-mesh visualization of native physics with a recorded RGB-D proxy; not camera photorealism or exact GT assets')
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    return metadata


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['submission','evaluation-receipt','execution','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--urdf',type=Path,default=DEFAULT_URDF)
    parser.add_argument('--view',choices=['overview','contact'],default='overview')
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
