"""Compiled equivalent of the existing CPU renderer, not a mesh simplification."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

_lib=None


def render(vertices,faces,K,c2w,size=160,color=(105,174,220)):
    global _lib
    if _lib is None:
        source=Path(__file__).with_name('raster_cpu.cpp')
        cache=source.parents[1]/'.runtime/raster';cache.mkdir(parents=True,exist_ok=True)
        target=cache/('raster-'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.so')
        if not target.exists():
            flags=[]
            if sys.platform=='darwin':
                sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
                flags=['-isysroot',sdk,'-isystem',str(Path(sdk)/'usr/include/c++/v1')]
            subprocess.run(['clang++','-std=c++17','-O3','-shared','-fPIC',*flags,str(source),'-o',str(target)],check=True)
        _lib=ctypes.CDLL(str(target))
        ptr=ctypes.c_void_p;integer=ctypes.c_int
        _lib.raster_cpu.argtypes=[ptr,integer,ptr,integer,ptr,ptr,integer,integer,ptr,ptr,ptr]
        _lib.raster_cpu.restype=None
    v=np.ascontiguousarray(vertices,np.float64);f=np.ascontiguousarray(faces,np.int32)
    k=np.ascontiguousarray(K,np.float64);c=np.ascontiguousarray(c2w,np.float64);col=np.ascontiguousarray(color,np.float64)
    H,W=(size,size) if np.isscalar(size) else size
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or k.shape!=(3,3) or c.shape!=(4,4):
        raise ValueError('Invalid renderer shapes')
    if len(f) and (f.min()<0 or f.max()>=len(v)):raise ValueError('Invalid face index')
    if not np.isfinite(v).all() or not np.isfinite(k).all() or not np.isfinite(c).all():raise ValueError('Nonfinite geometry/camera')
    image=np.empty((H,W,3),np.uint8);depth=np.empty((H,W),np.float32)
    _lib.raster_cpu(v.ctypes.data,len(v),f.ctypes.data,len(f),k.ctypes.data,c.ctypes.data,H,W,col.ctypes.data,image.ctypes.data,depth.ctypes.data)
    return image,depth
