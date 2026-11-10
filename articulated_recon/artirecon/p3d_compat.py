"""pytorch3d in test2 is built for sm_90 only. On other GPUs (e.g. A100 MIG, sm_80) replace knn_points with a
chunked torch implementation (same return fields: dists = squared distances, idx) and render on the CPU."""
from collections import namedtuple

import torch

KNN = namedtuple("KNN", "dists idx knn")
NATIVE = (not torch.cuda.is_available()) or torch.cuda.get_device_capability()[0] >= 9


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _nn_kernel(Q, R, L2, OD, OI, N, M, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
        pid = tl.program_id(0)
        b = tl.program_id(1)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mn = offs_n < N
        qb = Q + b * N * 3
        qx = tl.load(qb + offs_n * 3 + 0, mask=mn, other=0.0)
        qy = tl.load(qb + offs_n * 3 + 1, mask=mn, other=0.0)
        qz = tl.load(qb + offs_n * 3 + 2, mask=mn, other=0.0)
        Mb = tl.load(L2 + b)
        best = tl.full([BLOCK_N], float("inf"), tl.float32)
        bi = tl.zeros([BLOCK_N], tl.int32)
        rb = R + b * M * 3
        for m0 in range(0, M, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mm = offs_m < Mb
            rx = tl.load(rb + offs_m * 3 + 0, mask=mm, other=0.0)
            ry = tl.load(rb + offs_m * 3 + 1, mask=mm, other=0.0)
            rz = tl.load(rb + offs_m * 3 + 2, mask=mm, other=0.0)
            dx = qx[:, None] - rx[None, :]
            dy = qy[:, None] - ry[None, :]
            dz = qz[:, None] - rz[None, :]
            d = dx * dx + dy * dy + dz * dz
            d = tl.where(mm[None, :], d, float("inf"))
            dmin = tl.min(d, axis=1)
            amin = tl.argmin(d, axis=1).to(tl.int32)
            upd = dmin < best
            best = tl.where(upd, dmin, best)
            bi = tl.where(upd, amin + m0, bi)
        tl.store(OD + b * N + offs_n, best, mask=mn)
        tl.store(OI + b * N + offs_n, bi, mask=mn)

    def nn_triton(p1, p2, lengths2=None):
        B, N, _ = p1.shape
        M = p2.shape[1]
        q = p1.detach().float().contiguous()
        r = p2.detach().float().contiguous()
        L2 = (lengths2.to(torch.int32) if lengths2 is not None else torch.full((B,), M, dtype=torch.int32, device=p1.device)).contiguous()
        od = torch.empty(B, N, device=p1.device)
        oi = torch.empty(B, N, dtype=torch.int32, device=p1.device)
        grid = (triton.cdiv(N, 16), B)
        _nn_kernel[grid](q, r, L2, od, oi, N, M, BLOCK_N=16, BLOCK_M=256, num_warps=2)
        return oi.long()
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


def knn_points_torch(p1, p2, lengths1=None, lengths2=None, K=1, return_nn=False, **kw):
    B, N, _ = p1.shape
    M = p2.shape[1]
    if p1.device.type == "cpu":
        return _ORIG(p1, p2, lengths1=lengths1, lengths2=lengths2, K=K, return_nn=return_nn)
    if K == 1 and HAVE_TRITON:
        with torch.no_grad():
            idx = nn_triton(p1, p2, lengths2)[..., None]
        nn = torch.gather(p2, 1, idx.reshape(B, N, 1).expand(B, N, 3)).reshape(B, N, 1, 3)
        return KNN(((p1[:, :, None] - nn) ** 2).sum(-1), idx, nn if return_nn else None)
    with torch.no_grad():
        q = p2.detach()
        if lengths2 is not None:
            pad = torch.arange(M, device=p2.device)[None] >= lengths2[:, None]
        idx_all = []
        step = max(1, int(2.5e8 // max(1, B * M)))
        for s in range(0, N, step):
            d = torch.cdist(p1[:, s:s + step].detach(), q)
            if lengths2 is not None:
                d = d.masked_fill(pad[:, None], float("inf"))
            idx_all.append(d.topk(K, dim=-1, largest=False).indices)
        idx = torch.cat(idx_all, 1)                                            # B,N,K
    nn = torch.gather(p2, 1,
                      idx.reshape(B, N * K, 1).expand(B, N * K, 3)).reshape(B, N, K, 3)
    dists = ((p1[:, :, None] - nn) ** 2).sum(-1)
    return KNN(dists, idx, nn if return_nn else None)


_ORIG = None
if not NATIVE:
    import pytorch3d.ops
    _ORIG = pytorch3d.ops.knn_points
    pytorch3d.ops.knn_points = knn_points_torch
    import pytorch3d.ops.knn as _k
    _k.knn_points = knn_points_torch
