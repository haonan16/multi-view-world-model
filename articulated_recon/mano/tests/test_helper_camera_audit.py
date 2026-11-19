from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.helper_camera_audit import requested_pixels,compare_displacements,transform_grid


def test_identity_camera_preserves_pixels_and_baseline_moves_them():
    k=np.array([[100,0,50],[0,100,40],[0,0,1.]])
    pixels=np.array([[30,30],[60,50.]])
    depth=np.array([1.,2.])
    projected,valid=requested_pixels(pixels,depth,k,k,np.eye(4),np.eye(4))
    assert valid.all() and np.allclose(projected,pixels)
    target=np.eye(4);target[0,3]=.1
    moved,valid=requested_pixels(pixels,depth,k,k,np.eye(4),target)
    assert valid.all() and np.allclose(moved,pixels-np.array([[10,0],[5,0]]))


def test_flow_fraction_distinguishes_copy_from_requested_projection():
    source=np.array([[10,10],[20,20],[30,40.]])
    expected=source+np.array([[10,0],[0,10],[-10,0]])
    copied=compare_displacements(source,source,expected)
    correct=compare_displacements(source,expected,expected)
    assert copied['requested_flow_fraction_median']==0 and copied['identity_is_closer_fraction']==1
    assert correct['requested_flow_fraction_median']==1 and correct['requested_reprojection_error_median_px']==0


def test_crop_changes_principal_point_without_changing_depth():
    image=np.ones((360,640),np.float32)*2.
    resized=transform_grid(image,[0,20,640,320],(320,160))
    assert resized.shape==(160,320) and np.all(resized==2)


def test_nonpositive_depth_is_excluded_from_camera_evidence():
    k=np.eye(3);pixels=np.ones((3,2));depth=np.array([1.,0.,np.nan])
    projected,valid=requested_pixels(pixels,depth,k,k,np.eye(4),np.eye(4))
    assert valid.tolist()==[True,False,False] and np.isnan(projected[1:]).all()
