"""Compare estimated block fits; missing poses are never drawn as observations."""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np


SIGNS=np.array(list(itertools.product([-1.,1.],repeat=3)))
EDGES=[(a,b) for a in range(8) for b in range(a+1,8) if np.sum(SIGNS[a]!=SIGNS[b])==1]


def corners(scene, frame):
    obj=scene['objects'][0]
    pose=np.asarray(obj['T_world_object'][frame])
    return (SIGNS*np.asarray(obj['geometry']['half_extents_m']))@pose[:3,:3].T+pose[:3,3]


def draw_box(image,points,color):
    if not np.isfinite(points).all() or np.max(np.abs(points))>1e5:return
    points=np.rint(points).astype(int)
    for a,b in EDGES:cv2.line(image,tuple(points[a]),tuple(points[b]),color,2,cv2.LINE_AA)


def render(*, geometry_bundle, all_fit, rank_fit, output, left_label='All valid depth', right_label='Top 50% confidence rank'):
    import imageio.v2 as imageio
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    with np.load(Path(geometry_bundle)/'geometry.npz',allow_pickle=False) as data:
        rgb=data['rgb'][:,0].copy();k=data['K'][0];times=data['timestamps'].copy()
    scenes=[json.loads((Path(folder)/'scene-sequence.json').read_text()) for folder in [all_fit,rank_fit]]
    for scene in scenes:
        if not np.array_equal(scene['timestamps'],times):raise ValueError('Frame times differ')
    n=np.asarray(scenes[0]['support']['normal_world']);d=scenes[0]['support']['offset_m']
    p=np.asarray(scenes[0]['objects'][0]['T_world_object'])[:,:3,3]
    horizontal=p[-1]-p[0];horizontal-=n*(horizontal@n);horizontal/=np.linalg.norm(horizontal)
    origin=p[0]-n*(n@p[0]+d)
    all_points=np.concatenate([corners(scene,f) for scene in scenes for f in range(len(times)) if scene['objects'][0]['observed'][f]])
    projected=np.c_[(all_points-origin)@horizontal,(all_points-origin)@n]
    low=projected.min(0)-.04;high=projected.max(0)+.04
    h,w=rgb.shape[1:3];scale=min((w-80)/(high[0]-low[0]),(h-100)/(high[1]-low[1]))
    def side(points):
        xy=np.c_[(points-origin)@horizontal,(points-origin)@n]
        xy=(xy-low)*scale+[40,40];xy[:,1]=h-xy[:,1]
        return xy
    output.mkdir(parents=True)
    fps=(len(times)-1)/(times[-1]-times[0])
    with imageio.get_writer(output/'comparison.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=1) as writer:
        for f,time in enumerate(times):
            rows=[]
            for scene,name,color in zip(scenes,[left_label,right_label],[(80,220,140),(250,180,70)]):
                seen=scene['objects'][0]['observed'][f]
                source=rgb[f].copy();plot=np.full_like(source,25)
                n2=np.asarray(scene['support']['normal_world']);d2=scene['support']['offset_m']
                # Each estimated table is projected in the common ALL baseline side view.
                t=np.linspace(low[0],high[0],60)
                support=origin+t[:,None]*horizontal
                support-=((support@n2+d2)/(n@n2))[:,None]*n
                cv2.polylines(plot,[np.rint(side(support)).astype(int)],False,(140,140,140),2)
                if seen:
                    points=corners(scene,f);q=points@k.T
                    if np.all(q[:,2]>.01):draw_box(source,q[:,:2]/q[:,2:],color)
                    draw_box(plot,side(points),color)
                source[:32]=25;plot[:32]=25
                cv2.putText(source,f'{name} | {time:.1f}s',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(plot,'Estimated 3D + table' if seen else 'Insufficient depth: no supported pose',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1,cv2.LINE_AA)
                rows.append(np.concatenate([source,plot],axis=1))
            frame=np.concatenate(rows,axis=0);writer.append_data(frame)
            if f in [0,48,len(times)-1]:cv2.imwrite(str(output/f'frame-{f:03d}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    (output/'preview.json').write_text(json.dumps({'fps':fps,'frames':len(times),'source_fits':[str(all_fit),str(rank_fit)],
        'interpretation':'Predicted cuboid overlay and common orthographic side view; not ground truth geometry or execution. Missing poses omitted.',
        'support':'Separately predicted table planes, projected into the same side view.'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['geometry-bundle','all-fit','rank-fit','output']:parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--left-label',default='All valid depth');parser.add_argument('--right-label',default='Top 50% confidence rank')
    render(**vars(parser.parse_args()))
