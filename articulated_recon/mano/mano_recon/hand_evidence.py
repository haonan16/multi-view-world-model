"""Human observation terms for the ported RobotFitter optimization stages.

Unlike a table-mounted robot, the hand may leave the scene and has too little
reliable sensor depth to determine all finger joints. RGB remains a constraint,
not merely an initialization. No GT joints or masks are used here.
"""
import torch
from torch.nn import functional as F
from mano_recon.model import translation_base


def attach_human_evidence(fitter, keypoints, valid_depth, visible):
    dev=fitter.dev
    fitter.human_keypoints=torch.as_tensor(keypoints,dtype=torch.float32,device=dev)
    fitter.human_depth_valid=torch.as_tensor(valid_depth,dtype=torch.bool,device=dev)
    fitter.human_visible=torch.as_tensor(visible,dtype=torch.bool,device=dev)


def human_loss(self,q,base,frames,w_depth=1.,s_ch=.02,model_extra=None):
    from .amani_objects.geom import chamfer_obs_to_model,rho
    fr=torch.as_tensor(frames,device=self.dev);seen=self.human_visible[fr]
    P,N=self.robot.surface(q,base)
    _,J=self.robot.hand(q[:,:-3],self.robot.betas,base@translation_base(q[:,-3:]))
    u,v,z=self.views.project(J)
    kp=self.human_keypoints[fr];weights=kp[...,2]*seen[:,None,None]
    residual=torch.stack([u,v],-1)-kp[...,:2]
    # Pixel-normalized robust reprojection; keep its scale independent of view count.
    err=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=3,reduction='none').sum(-1)
    per_view=(err*weights).sum(-1)
    if getattr(self,'finger_pair_hypotheses',False):
        # Some RGB views interchange index and middle finger identities. Compare
        # whole finger chains, never independently relabel individual joints.
        order=list(range(21));order[5:9],order[9:13]=order[9:13],order[5:9]
        alt=kp[:,:,order];alt_weights=alt[...,2]*seen[:,None,None]
        alt_err=F.smooth_l1_loss(torch.stack([u,v],-1)-alt[...,:2],torch.zeros_like(residual),beta=3,reduction='none').sum(-1)
        alternative=(alt_err*alt_weights).sum(-1)
        per_view=torch.minimum(per_view,alternative+.05*weights.sum(-1))
    reprojection=per_view.sum(1)/weights.sum((1,2)).clamp_min(1)/10
    behind=(F.relu(.01-z)*weights).sum((1,2))/weights.sum((1,2)).clamp_min(1)
    u,v,z=self.views.project(P);D,inside=self.views.sample(u,v,frames)
    B,V,S=u.shape
    ui=u.round().long().clamp(0,self.views.W-1);vi=v.round().long().clamp(0,self.views.H-1)
    vv=torch.arange(V,device=self.dev)[None,:,None]
    reliable=self.human_depth_valid[fr[:,None,None],vv,vi,ui]
    facing=((self.views.C[None,:,None]-P[:,None])*N[:,None]).sum(-1)>0
    valid=inside&reliable&facing&(z>.05)&(D>0)&seen[:,None,None]
    r=z-D
    cost=torch.where(r>.008,rho(torch.full_like(r,.008),.005),rho(r,.005))
    dl=(cost*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
    ch,_=chamfer_obs_to_model(self.F[fr],self.Fn[fr],P,s_ch)
    ch=ch*seen
    lim=(F.relu(self.lo-q)+F.relu(q-self.hi)).square().sum(1)*100
    below=F.relu(-(P@self.n+float(self.prep['d']))-.003).square().mean(1)/.01**2
    initial=self.robot.initial[fr]
    missing=(q[:,-3:]-initial[:,-3:]).square().sum(1)/.005**2+(q[:,:-3]-initial[:,:-3]).square().mean(1)
    trust=.1*(q[:,-3:]-initial[:,-3:]).square().sum(1)/.02**2
    prior=.01*self.robot.betas.square().mean()+.0001*q[:,3:-3].square().mean(1)
    total=ch+w_depth*dl+3*reprojection+behind+seen*(below+trust)+lim+prior+(~seen)*missing
    return total,ch,dl
