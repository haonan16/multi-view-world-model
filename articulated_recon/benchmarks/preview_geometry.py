"""Render predicted RGB-depth evidence; never present lifted points as a mesh."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def point_view(rgb, depth, k, camera, stride=2):
    h,w=depth.shape
    y,x=np.mgrid[0:h:stride,0:w:stride]
    z=depth[::stride,::stride]
    valid=np.isfinite(z)&(z>.05)&(z<5)
    rays=np.stack([x,y,np.ones_like(x)],axis=-1)[valid]@np.linalg.inv(k).T
    pts=rays*z[valid,None]
    pts=(pts-camera[:3,3])@camera[:3,:3]
    q=pts@k.T;uv=np.rint(q[:,:2]/np.maximum(q[:,2:],1e-8)).astype(int)
    keep=(pts[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    uv,pts,col=uv[keep],pts[keep],rgb[::stride,::stride][valid][keep]
    idx=uv[:,1]*w+uv[:,0];zbuf=np.full(h*w,np.inf)
    np.minimum.at(zbuf,idx,pts[:,2])
    front=pts[:,2]<=zbuf[idx]+1e-7
    out=np.full((h*w,3),24,np.uint8);out[idx[front]]=col[front]
    return out.reshape(h,w,3)


def render(bundle,output):
    from world_model_views import look_at_camera
    bundle=Path(bundle);output=Path(output)
    if output.exists():raise FileExistsError(output)
    with np.load(bundle/'geometry.npz',allow_pickle=False) as data:
        rgb,depth,rank=data['rgb'][:,0],data['depth_m'][:,0],data['confidence_rank'][:,0]
        k=data['K'][0];times=data['timestamps']
    vals=depth[np.isfinite(depth)&(depth>.05)&(depth<5)]
    near,far=np.quantile(vals,[.05,.95])
    if far<=near:far=near+.01
    h,w=depth.shape[1:];target=np.array([0.,0.,float(np.median(depth[0][np.isfinite(depth[0])&(depth[0]>.05)]))])
    camera=look_at_camera([target[2]*.25,0,0],target)
    output.mkdir(parents=True)
    fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 1
    writer=cv2.VideoWriter(str(output/'preview.mp4'),cv2.VideoWriter_fourcc(*'avc1'),fps,(2*w,2*h))
    if not writer.isOpened():
        import imageio.v2 as imageio
        writer=None;fallback=imageio.get_writer(output/'preview.mp4',fps=fps,codec='libx264',quality=8)
    else:fallback=None
    for i in range(len(rgb)):
        scaled=np.clip((depth[i]-near)/(far-near),0,1)
        scaled=np.nan_to_num(scaled)
        colored=cv2.cvtColor(cv2.applyColorMap((scaled*255).astype(np.uint8),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        confidence=np.repeat((rank[i].clip(0,1)*255).astype(np.uint8)[...,None],3,axis=2)
        lifted=point_view(rgb[i],depth[i],k,camera)
        tiles=[rgb[i].copy(),colored,confidence,lifted]
        names=['Recorded RGB',f'Predicted depth | {near:.2f} to {far:.2f} m','Confidence rank | not probability','Lifted visible points | not a mesh']
        for tile,name in zip(tiles,names):
            tile[:28]=24
            cv2.putText(tile,name,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(245,245,245),1,cv2.LINE_AA)
        frame=np.concatenate([np.concatenate(tiles[:2],axis=1),np.concatenate(tiles[2:],axis=1)],axis=0)
        if writer is not None:writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        else:fallback.append_data(frame)
        if i==0:cv2.imwrite(str(output/'first-frame.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    if writer is not None:writer.release()
    else:fallback.close()
    report={'frames':len(rgb),'fps':float(fps),'predicted_K':k.tolist(),'depth_color_limits_m':[float(near),float(far)],
            'render':'Observed RGB, predicted metric camera-Z depth, inverse-error rank and visible lifted points',
            'geometry_is_mesh':False,'ground_truth_geometry_compared':False,'invalid_or_beyond5m_points_hidden':True}
    (output/'preview.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();render(args.bundle,args.output)
