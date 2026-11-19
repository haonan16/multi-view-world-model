from types import SimpleNamespace
import torch
def knn_points(a,b,K=1,**kwargs):
    if K!=1:raise ValueError('Only K=1 is supported')
    vals,inds=[],[]
    for chunk in a.split(256,dim=1):
        d=torch.cdist(chunk,b).square();v,i=d.min(-1);vals.append(v);inds.append(i)
    return SimpleNamespace(dists=torch.cat(vals,1)[...,None],idx=torch.cat(inds,1)[...,None])
