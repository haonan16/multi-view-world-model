"""Unit checks for the multiview backbone (algorithms/wan/modules/model_multiview.py), on one
GPU with a tiny randomly initialised Wan (real head_dim 128, so the RoPE / camera dim split
is the production one):
  - every cam_mode runs forward + backward with the right output shape
  - prope / gta are invariant to moving all cameras by one rigid transform (attention sees
    only relative cameras D_i D_j^-1)
  - every mode is equivariant to permuting the views
Usage: python scripts/rh20t/test_multiview_model.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from algorithms.wan.modules.model_multiview import CAM_MODES, MultiViewWanModel  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
B, V, C, F_, h, w = 2, 3, 16, 3, 8, 12


def rand_se3(n):
    q = torch.linalg.qr(torch.randn(n, 3, 3))[0]
    q = q * torch.sign(torch.linalg.det(q))[:, None, None]
    M = torch.eye(4).repeat(n, 1, 1)
    M[:, :3, :3], M[:, :3, 3] = q, torch.randn(n, 3)
    return M


def make(mode):
    m = MultiViewWanModel(model_type="t2v", patch_size=(1, 2, 2), text_len=16, in_dim=C, dim=256,
                          ffn_dim=512, freq_dim=256, text_dim=64, out_dim=C, num_heads=2,
                          num_layers=2, cam_mode=mode).to(dev).float()
    if m.extra_in:          # make the conditioning channels matter in the test
        torch.nn.init.normal_(m.patch_embedding.weight, std=0.05)
    torch.nn.init.normal_(m.head.head.weight, std=0.05)    # Wan zero-inits the output head
    if mode == "cat4d":
        torch.nn.init.normal_(m.frame_time_embedding[2].weight, std=0.02)
    return m


x = torch.randn(B, V, C, F_, h, w, device=dev)
cond = torch.randn(B, V, 6, h, w, device=dev)
t = torch.rand(B, V, F_, device=dev) * 1000
t[:, :, 0] = 0
ctx = [torch.randn(5, 64, device=dev) for _ in range(B)]
w2c = rand_se3(B * V).view(B, V, 4, 4).to(dev)
K = torch.eye(4, device=dev).repeat(B, V, 1, 1)
K[..., 0, 0], K[..., 1, 1] = 0.8, 1.1
mats = K @ w2c
hist = t == 0
ok = True
for mode in CAM_MODES:
    m = make(mode)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x, cond, t, ctx, cam_mats=mats, hist_mask=hist)
    assert y.shape == x.shape, (mode, y.shape)
    y.float().pow(2).mean().backward()
    g = sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None)
    msg = f"{mode:9s} out {tuple(y.shape)} grad-sum {float(g):.3e}"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        perm = torch.tensor([2, 0, 1])
        yp = m(x[:, perm], cond[:, perm], t[:, perm], ctx, cam_mats=mats[:, perm], hist_mask=hist[:, perm])
        e_perm = float((yp.float() - y.float()[:, perm]).abs().max() / y.float().abs().max())
        msg += f" | view-permutation err {e_perm:.2e}"
        if mode in ("prope", "gta"):
            G = rand_se3(1)[0].to(dev)
            y2 = m(x, cond, t, ctx, cam_mats=mats @ torch.linalg.inv(G), hist_mask=hist)
            e_rig = float((y2.float() - y.float()).abs().max() / y.float().abs().max())
            msg += f" | global-rigid-transform err {e_rig:.2e}"
            ok &= e_rig < 3e-2
            # and the cameras must actually matter
            y3 = m(x, cond, t, ctx, cam_mats=mats[:, torch.tensor([1, 2, 0])], hist_mask=hist)
            msg += f" | swap-cameras change {float((y3.float() - y.float()).abs().max() / y.float().abs().max()):.2e}"
        ok &= e_perm < 3e-2
    print(msg, flush=True)
print("ALL_OK" if ok else "CHECK_FAILED")
