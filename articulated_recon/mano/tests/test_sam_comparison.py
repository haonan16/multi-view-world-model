import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module(name):
    p=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_sam3d_selects_smallest_prompt_matching_instance_under_click(tmp_path):
    choose=module('run_sam_comparison_networks').choose_mask
    (tmp_path/'ann.json').write_text(json.dumps({'objects':[{'prompt':['block'],'hint':{'t':0,'v':0,'uv':[2,2]}}]}))
    masks=np.zeros((3,6,6),bool);masks[0]=True;masks[1,1:4,1:4]=True;masks[2,2,2]=True
    np.savez(tmp_path/'sam3.npz',prompts=['block','hand'],p=[0,0,1],t=[0,0,0],v=[0,0,0],
             shape=[6,6],mask=np.array([np.packbits(m) for m in masks]))
    _,(index,m)=choose(tmp_path)
    assert index==1 and m.sum()==9


def test_sam3d_refuses_no_detection_instead_of_using_assisted_mask(tmp_path):
    choose=module('run_sam_comparison_networks').choose_mask
    (tmp_path/'ann.json').write_text(json.dumps({'objects':[{'prompt':['block'],'hint':{'t':0,'v':0,'uv':[2,2]}}]}))
    np.savez(tmp_path/'sam3.npz',prompts=['block'],p=[0],t=[1],v=[0],shape=[6,6],mask=np.ones((1,5),np.uint8))
    with pytest.raises(RuntimeError,match='no instance'):choose(tmp_path)


def test_video_keeps_anisotropic_sam3d_transform_at_fitted_frames():
    from mano_recon.motion import interpolate
    from scipy.spatial.transform import Rotation
    poses=np.tile(np.eye(4),(3,1,1));poses[:,:3,:3]=Rotation.from_rotvec([[0,0,0],[0,0,.2],[0,0,.5]]).as_matrix()@np.diag([.8,1.2,1.1]);poses[:,0,3]=[0,.1,.2]
    result=interpolate(poses,np.array([0,4,9]),10)
    assert np.allclose(result[[0,4,9]],poses)
    assert np.allclose(np.linalg.svd(result[:,:3,:3],compute_uv=False),[1.2,1.1,.8])
