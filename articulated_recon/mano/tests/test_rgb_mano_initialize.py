import importlib.util
from pathlib import Path

import numpy as np


def module():
    p=Path(__file__).resolve().parents[2]/'benchmarks/rgb_mano_initialize.py'
    s=importlib.util.spec_from_file_location('rgb_mano_initialize',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_wrist_bootstrap_excludes_background_and_rejects_ambiguous_depth():
    m=module();depth=np.ones((40,40));mask=np.zeros_like(depth,bool);mask[17:24,17:24]=True
    depth[mask]=.6
    assert m.wrist_depth(depth,mask,[20,20])==.6
    mask[:]=False
    assert m.wrist_depth(depth,mask,[20,20]) is None
    mask[17:24,17:24]=True;depth[17:20,17:24]=1.
    assert m.wrist_depth(depth,mask,[20,20]) is None


def test_track_keeps_absent_frames_missing_and_uses_task_proximity():
    m=module();xy=np.zeros((4,21,2));xy[0]=[20,20];xy[1]=[180,180];xy[2]=[23,20];xy[3]=[180,180]
    selected=m.choose_track(xy,np.array([0,0,2,2]),np.array([[20,20],[20,20],[23,20]]),(200,200))
    assert selected.tolist()==[0,-1,2]


def test_side_ranking_keeps_consistent_rgb_evidence_instead_of_reprojection_alone():
    m=module();h=dict(left=dict(selection_score=9.,relative3d_rmse_mm=8.),
                     right=dict(selection_score=10.4,relative3d_rmse_mm=6.))
    ranked=m.score_hand_side(h,['Right']*10,[.99]*10)
    assert ranked['right']['total_score']<ranked['left']['total_score']
    old=m.score_hand_side(h,['Right']*10,[.99]*10,'reprojection_only')
    assert old['left']['total_score']<old['right']['total_score']
