"""30-fps presentation of native forecast images and an actual 3D estimate.

RGB/depth evidence is held at its native timestamps. Only the separate 3D
display may interpolate supported rigid poses. No extra prediction frames,
MANO hand mesh, simulator states or evaluator geometry are manufactured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
from artirecon.input_protocol import file_sha256,load_manifest
from artirecon.scene_sequence import load_scene
from representation_agent_tools import geometry_arrays,raster_depth

SIGNS=np.asarray([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]],float)
EDGES=[(a,b) for a in range(8) for b in range(a+1,8) if np.sum(SIGNS[a]!=SIGNS[b])==1]
GOLD=(246,182,68)


def _font(size):
    for name in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/System/Library/Fonts/Supplemental/Arial.ttf','DejaVuSans.ttf']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()


def timeline(timestamps,fps=30,initial_hold_s=.5,final_hold_s=.5):
    times=np.asarray(timestamps,float)
    if len(times)<2 or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('Need explicit increasing native forecast timestamps')
    if fps<=0 or min(initial_hold_s,final_hold_s)<0:
        raise ValueError('Invalid display frame rate/hold duration')
    first=int(round(initial_hold_s*fps));last=max(1,int(round(final_hold_s*fps)))
    active=int(np.ceil((times[-1]-times[0])*fps-1e-9))
    queries=np.r_[np.full(first,times[0]),times[0]+np.arange(active)/fps,np.full(last,times[-1])]
    rows=np.clip(np.searchsorted(times,queries,side='right')-1,0,len(times)-1)
    phases=['initial hold']*first+['forecast']*active+['final hold']*last
    return queries,rows,phases


def supported_pose(poses,times,supported,query,interpolate=True):
    """Never smooth through a missing pose or treat a fallback as evidence."""
    left=int(np.clip(np.searchsorted(times,query,side='right')-1,0,len(times)-1))
    if not supported[left]:return None,'unsupported pose'
    if not interpolate or left==len(times)-1 or np.isclose(query,times[left],atol=1e-8,rtol=0):
        return poses[left].copy(),'native fitted pose'
    if not supported[left+1]:return None,'unsupported interval'
    alpha=(query-times[left])/(times[left+1]-times[left])
    pose=np.eye(4)
    pose[:3,:3]=Slerp(times[left:left+2],Rotation.from_matrix(poses[left:left+2,:3,:3]))([query]).as_matrix()[0]
    pose[:3,3]=(1-alpha)*poses[left,:3,3]+alpha*poses[left+1,:3,3]
    return pose,'interpolated 3D pose'


def project(points,k,camera):
    local=(np.asarray(points)-camera[:3,3])@camera[:3,:3]
    image=local@k.T
    return image[:,:2]/np.maximum(image[:,2,None],1e-8),local[:,2]


def wireframe(image,corners,k,camera,color=GOLD,width=2):
    result=image.copy();uv,z=project(corners,k,camera)
    if not np.isfinite(uv).all():return result
    for a,b in EDGES:
        if z[a]>.001 and z[b]>.001 and np.max(np.abs(uv[[a,b]]))<1e6:
            cv2.line(result,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),color,width,cv2.LINE_AA)
    return result


def look_at(eye,target,up):
    forward=target-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,up);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    camera=np.eye(4);camera[:3,:3]=np.column_stack([right,down,forward]);camera[:3,3]=eye
    return camera


def display_geometry(scene,obj):
    poses=np.asarray(obj['T_world_object']);half=np.asarray(obj['geometry']['half_extents_m'])
    supported=np.asarray(obj['observed'],bool)|np.asarray(obj.get('prediction_supported',np.zeros(len(poses))),bool)
    centres=poses[supported,:3,3]
    if not len(centres):raise ValueError('No supported observed/predicted object pose for display')
    target=np.mean(centres,axis=0)
    radius=max(.15,float(np.linalg.norm(centres-target,axis=1).max()+np.linalg.norm(half)*2.5))
    support=scene['support'];up=np.asarray(support['normal_world'] if support else [0,-1,0],float)
    up/=np.linalg.norm(up)
    front=-target+up*np.dot(target,up)
    if np.linalg.norm(front)<1e-6:front=np.cross(up,[1.,0,0])
    if np.linalg.norm(front)<1e-6:front=np.cross(up,[0.,0,1])
    front/=np.linalg.norm(front);side=np.cross(up,front)
    plane=None
    if support is not None:
        centre=target-(target@up+support['offset_m'])*up
        plane=dict(centre=centre,up=up,u=front,v=side,half_extent=radius*1.2)
    return dict(target=target,radius=radius,up=up,front=front,side=side,plane=plane,
                poses=poses,half=half,supported=supported)


def predicted_cloud(arrays,index,display,max_points=5000):
    depth=arrays['depth_m'][index];rgb=arrays['rgb'][index]
    parts=[];colors=[]
    for v in range(len(depth)):
        h,w=depth[v].shape;y,x=np.mgrid[0:h:3,0:w:3]
        z=depth[v,::3,::3]
        valid=np.isfinite(z)&(z>0)
        if not valid.any():continue
        k=arrays['K'][index,v];c=arrays['c2w'][index,v]
        camera=(np.stack([x,y,np.ones_like(x)],axis=-1)[valid]@np.linalg.inv(k).T)*z[valid,None]
        world=camera@c[:3,:3].T+c[:3,3]
        keep=np.linalg.norm(world-display['target'],axis=1)<display['radius']*1.5
        parts.append(world[keep]);colors.append(rgb[v,::3,::3][valid][keep])
    points=np.concatenate(parts) if parts else np.empty((0,3))
    color=np.concatenate(colors) if colors else np.empty((0,3),np.uint8)
    if len(points)>max_points:
        ids=np.unique(np.rint(np.linspace(0,len(points)-1,max_points)).astype(int));points=points[ids];color=color[ids]
    return points,color


def orbit_panel(display,points,colors,pose,progress,size,orbit_degrees=90):
    h,w=size;target=display['target'];radius=display['radius']
    angle=np.deg2rad((progress-.5)*orbit_degrees)
    direction=np.cos(angle)*display['front']+np.sin(angle)*display['side']+.6*display['up']
    direction/=np.linalg.norm(direction)
    camera=look_at(target+direction*radius*3.8,target,display['up'])
    k=np.array([[w*.95,0,w/2],[0,w*.95,h/2],[0,0,1.]])
    image=np.full((h,w,3),(26,32,42),np.uint8);zbuffer=np.full((h,w),np.inf)
    plane=display['plane']
    if plane is not None:
        extent=plane['half_extent'];centre=plane['centre'];u,v=plane['u'],plane['v']
        corners=np.array([centre+extent*(a*u+b*v) for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]])
        plane_depth=raster_depth(corners,np.array([[0,1,2],[0,2,3]]),k,camera,(h,w))
        drawn=plane_depth>0;image[drawn]=(42,51,62);zbuffer[drawn]=plane_depth[drawn]
        for axis,other in [(u,v),(v,u)]:
            for step in np.linspace(-extent,extent,9):
                line=np.array([centre+axis*step-other*extent,centre+axis*step+other*extent])
                uv,z=project(line,k,camera)
                if (z>0).all():cv2.line(image,tuple(np.rint(uv[0]).astype(int)),tuple(np.rint(uv[1]).astype(int)),(59,70,83),1,cv2.LINE_AA)
    if len(points):
        uv,z=project(points,k,camera);pixel=np.rint(uv).astype(int)
        valid=(z>0)&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
        pixel,z,color=pixel[valid],z[valid],colors[valid]
        order=np.argsort(z)[::-1];pixel,z,color=pixel[order],z[order],color[order]
        # Points within a small tolerance of the estimated plane stay visible.
        valid=z<zbuffer[pixel[:,1],pixel[:,0]]+.006
        pixel,z,color=pixel[valid],z[valid],color[valid]
        for dx,dy in [(0,0),(1,0),(0,1)]:
            p=pixel+[dx,dy];inside=(p[:,0]<w)&(p[:,1]<h)
            image[p[inside,1],p[inside,0]]=np.maximum(color[inside],42)
            zbuffer[p[inside,1],p[inside,0]]=z[inside]
    centres=display['poses'][:,:3,3]
    uv,z=project(centres,k,camera)
    for t in range(len(centres)-1):
        if display['supported'][t] and display['supported'][t+1] and min(z[t],z[t+1])>0:
            cv2.line(image,tuple(np.rint(uv[t]).astype(int)),tuple(np.rint(uv[t+1]).astype(int)),(90,190,220),2,cv2.LINE_AA)
    if pose is not None:
        vertices,faces=geometry_arrays(dict(type='box',half_extents_m=display['half']))
        world=vertices@pose[:3,:3].T+pose[:3,3]
        for face in range(0,len(faces),2):
            face_depth=raster_depth(world,faces[face:face+2],k,camera,(h,w))
            normal=np.cross(world[faces[face,1]]-world[faces[face,0]],world[faces[face,2]]-world[faces[face,0]])
            normal/=max(np.linalg.norm(normal),1e-12)
            shade=.55+.4*abs(normal@camera[:3,2])
            drawn=(face_depth>0)&(face_depth<zbuffer+.003)
            image[drawn]=np.asarray(GOLD)*shade;zbuffer[drawn]=face_depth[drawn]
        corners=(SIGNS*display['half'])@pose[:3,:3].T+pose[:3,3]
        image=wireframe(image,corners,k,camera,(255,206,113),1)
    return image


def load_inputs(helper,bundle):
    helper,bundle=Path(helper).resolve(),Path(bundle).resolve()
    hr=load_manifest(helper/'input-manifest.json',[('rgb',helper/'observations.npz')])
    br=load_manifest(bundle/'input-manifest.json',[('object_geometry',bundle/'scene-sequence.json'),
        ('depth',bundle/'geometry.npz'),('cameras',bundle/'geometry.npz')])
    if hr['task_mode']!='forecast' or br['task_mode']!='forecast':raise ValueError('Only strict forecast bundles are rendered')
    receipt=json.loads((bundle/'execution.json').read_text())
    if receipt.get('status')!='complete' or receipt.get('scene_sha256')!=file_sha256(bundle/'scene-sequence.json'):
        raise ValueError('Forecast scene is incomplete or changed after fitting')
    scene=load_scene(bundle/'scene-sequence.json')
    with np.load(helper/'observations.npz',allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    with np.load(bundle/'geometry.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    rgb,times=source['rgb'],source['timestamps']
    if rgb.ndim!=5 or rgb.shape[1]!=4 or not np.array_equal(times,arrays['timestamps']) or not np.array_equal(times,scene['timestamps']):
        raise ValueError('Need four generated views and exactly matched scene/geometry timestamps')
    views=arrays['rgb'].shape[1]
    if views not in {1,4} or not np.array_equal(arrays['rgb'],rgb[:,:views]):raise ValueError('Fitted RGB differs from generated pixels')
    expected=np.zeros(rgb.shape[:2],bool);expected[0,0]=True
    if not np.array_equal(source['observed_mask'],expected):raise ValueError('Only the initial source image may be observed')
    generated=[a for a in json.loads((bundle/'input-manifest.json').read_text())['artifacts']
        if a['id'] in br['method_artifact_ids'] and a.get('generated') is True and a.get('sha256')==file_sha256(helper/'observations.npz')]
    if not generated:raise ValueError('Provided generated views are not the fitted forecast ancestry')
    obj=next(o for o in scene['objects'] if o['geometry']['type']=='box')
    return source,arrays,scene,obj,receipt


def ffmpeg_executable():
    found=shutil.which('ffmpeg')
    if found:return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:raise RuntimeError('Install/use an existing ffmpeg or imageio-ffmpeg runtime; no download is attempted') from exc


def render(*,helper,bundle,output,fps=30,initial_hold_s=.5,final_hold_s=.5,
           interpolate_3d=True,panel_width=384,orbit_degrees=90.,max_points=5000):
    source,arrays,scene,obj,fit_receipt=load_inputs(helper,bundle)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if panel_width<160 or max_points<1:raise ValueError('Invalid display dimensions/point budget')
    times=np.asarray(scene['timestamps']);queries,rows,phases=timeline(times,fps,initial_hold_s,final_hold_s)
    display=display_geometry(scene,obj);rgb=source['rgb'];native_fps=1/float(np.median(np.diff(times)))
    image_h=round(panel_width*rgb.shape[2]/rgb.shape[3]);tile_h=image_h+26
    gap,margin=12,16;left_w=panel_width*2+gap;body_h=2*tile_h+gap;orbit_w=480
    width=margin*2+left_w+gap+orbit_w;header,footer=64,38;height=margin*2+header+body_h+footer
    width+=width%2;height+=height%2
    output.mkdir(parents=True)
    destination=output/'forecast-reconstruction.mp4'
    command=[ffmpeg_executable(),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart','-n',str(destination)]
    title_font,body_font,small_font=_font(26),_font(16),_font(13)
    clouds=[predicted_cloud(arrays,t,display,max_points) for t in range(len(times))]
    native_tiles=[]
    for t in range(len(times)):
        tiles=[];pose=display['poses'][t];corners=(SIGNS*display['half'])@pose[:3,:3].T+pose[:3,3]
        for v in range(4):
            picture=rgb[t,v].copy()
            has_camera=v<arrays['K'].shape[1]
            if display['supported'][t] and has_camera:
                marked=wireframe(picture,corners,arrays['K'][t,v],arrays['c2w'][t,v])
                if 'occlusion_mask' in arrays:marked[arrays['occlusion_mask'][t,v]]=picture[arrays['occlusion_mask'][t,v]]
                picture=marked
            image=Image.fromarray(picture).resize((panel_width,image_h),Image.Resampling.LANCZOS)
            tile=Image.new('RGB',(panel_width,tile_h),(30,37,47));tile.paste(image,(0,26));draw=ImageDraw.Draw(tile)
            label=f'View {v} · '+('observed initial image' if t==0 and v==0 else 'generated')
            if not has_camera:label+=' · no fitted camera'
            draw.text((8,5),label,font=small_font,fill=(225,232,240));tiles.append(tile)
        native_tiles.append(tiles)
    snapshots={0,len(queries)//2,len(queries)-1}
    with (output/'encoder-stderr.txt').open('wb') as errors:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=errors)
        try:
            for index,(query,native,phase) in enumerate(zip(queries,rows,phases)):
                canvas=Image.new('RGB',(width,height),(17,23,32));draw=ImageDraw.Draw(canvas)
                draw.text((margin,14),'Forecast reconstruction',font=title_font,fill=(244,247,252))
                draw.text((margin,45),'One observed image  ·  Generated multi-view RGB  ·  Estimated 3D',font=body_font,fill=(163,178,197))
                top=margin+header
                for v,tile in enumerate(native_tiles[native]):canvas.paste(tile,(margin+(v%2)*(panel_width+gap),top+(v//2)*(tile_h+gap)))
                pose,pose_status=supported_pose(display['poses'],times,display['supported'],query,interpolate_3d)
                orbit=orbit_panel(display,*clouds[native],pose,index/max(1,len(queries)-1),(body_h-26,orbit_w),orbit_degrees)
                orbit_x=margin+left_w+gap;canvas.paste(Image.fromarray(orbit),(orbit_x,top+26))
                draw.rectangle((orbit_x,top,orbit_x+orbit_w,top+26),fill=(30,37,47))
                draw.text((orbit_x+8,top+5),'3D estimate · display orbit',font=small_font,fill=(225,232,240))
                dimensions=' × '.join(f'{x:.1f}' for x in display['half']*200)
                draw.text((orbit_x+10,top+body_h-65),'Box estimate: '+dimensions+' cm',font=small_font,fill=(215,226,239))
                scene_label='Predicted points · shared box · '+('estimated plane' if display['plane'] is not None else 'metric track')
                draw.text((orbit_x+10,top+body_h-45),scene_label,font=small_font,fill=(215,226,239))
                draw.text((orbit_x+10,top+body_h-25),pose_status+'; cloud held at native frame',font=small_font,fill=GOLD)
                native_elapsed=times[native]-times[0]
                label=f'{native_fps:g} fps RGB held  →  {fps:g} fps display    |    native t={native_elapsed:.2f}s'
                if phase!='forecast':label+='    |    '+phase
                draw.text((margin,top+body_h+12),label,font=small_font,fill=(191,204,220))
                plane_label='Plane extent is illustrative' if display['plane'] is not None else 'No support plane supplied'
                draw.text((orbit_x,top+body_h+12),plane_label+' · no hand mesh',font=small_font,fill=(161,178,198))
                frame=np.asarray(canvas);proc.stdin.write(frame.tobytes())
                if index in snapshots:canvas.save(output/f'preview-{index:03d}.png')
            proc.stdin.close()
            code=proc.wait(timeout=60)
            if code:raise RuntimeError('Video encoder failed; inspect encoder-stderr.txt')
        except BaseException:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
            raise
    # A decodable-frame check verifies the delivered video rather than just file existence.
    reader=cv2.VideoCapture(str(destination));ok,decoded=reader.read()
    frame_count=int(reader.get(cv2.CAP_PROP_FRAME_COUNT));encoded_fps=float(reader.get(cv2.CAP_PROP_FPS));reader.release()
    if not ok or decoded.shape[:2]!=(height,width) or frame_count!=len(queries):
        raise RuntimeError('Encoded video did not decode with the expected dimensions/frame count')
    report=dict(status='complete',video=str(destination),video_sha256=file_sha256(destination),
        display_fps=encoded_fps,display_frames=frame_count,display_duration_s=frame_count/encoded_fps,
        prediction_horizon_s=float(times[-1]-times[0]),native_timestamps=times.tolist(),native_rgb_fps= native_fps,
        initial_hold_s=initial_hold_s,final_hold_s=final_hold_s,source_frame_for_display=rows.tolist(),
        query_time_for_display=queries.tolist(),rgb_interpolation=False,depth_interpolation=False,
        pose_interpolation='SLERP + linear translation only in separate 3D display, only across supported pairs' if interpolate_3d else 'none',
        camera_overlay='Fitted DA3 branch cameras only; absent helper projections omitted',
        display_orbit_degrees=orbit_degrees,point_cloud='Predicted depth lifted with fitted cameras; interaction-region display crop',
        point_count_per_native_frame=[len(x[0]) for x in clouds],
        table='Estimated infinite plane; finite patch drawn only for display' if display['plane'] is not None else 'No support plane supplied or drawn',
        object_dimensions_m=(2*display['half']).tolist(),hand_mesh_present=False,GT_used=False,simulator_used=False,
        branch=fit_receipt['branch'],scene_sha256=file_sha256(Path(bundle)/'scene-sequence.json'),
        helper_sha256=file_sha256(Path(helper)/'observations.npz'),producer_sha256=file_sha256(__file__))
    (output/'render-receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['helper','bundle','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fps',type=int,default=30);p.add_argument('--initial-hold-s',type=float,default=.5)
    p.add_argument('--final-hold-s',type=float,default=.5);p.add_argument('--no-interpolate-3d',action='store_true')
    p.add_argument('--panel-width',type=int,default=384);p.add_argument('--orbit-degrees',type=float,default=90.)
    p.add_argument('--max-points',type=int,default=5000)
    args=vars(p.parse_args());args['interpolate_3d']=not args.pop('no_interpolate_3d')
    print(json.dumps(render(**args),indent=2))


if __name__=='__main__':main()
