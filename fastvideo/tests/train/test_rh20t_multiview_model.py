# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the multi-view Wan transformer and the RH20T training helpers."""

import importlib.util
from pathlib import Path

import pytest
import torch

import fastvideo.envs as envs

envs.setdefault_external("MASTER_ADDR", "localhost")
envs.setdefault_external("MASTER_PORT", "29517")

from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.models.wan.config import WanVideoArchConfig, WanVideoConfig  # noqa: E402
from fastvideo.models.wan.multiview_transformer import MultiViewWanTransformer3DModel  # noqa: E402
from fastvideo.models.wan.transformer import WanTransformer3DModel  # noqa: E402
from fastvideo.train.methods.fine_tuning.multiview_flow import (future_frame_metrics, masked_flow_loss,  # noqa: E402
                                                                ssim)
from fastvideo.train.models.wan.multiview import InfiniteRandomSampler  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
Z_DIM, RAY = 4, 6
TEXT_DIM, TEXT_LEN = 32, 7


def _load_converter():
    spec = importlib.util.spec_from_file_location("make_multiview_checkpoint",
                                                  REPO / "scripts" / "rh20t" / "make_multiview_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(cls, in_channels: int):
    arch = WanVideoArchConfig(num_attention_heads=2,
                              attention_head_dim=12,
                              in_channels=in_channels,
                              out_channels=Z_DIM,
                              text_dim=TEXT_DIM,
                              freq_dim=16,
                              ffn_dim=32,
                              num_layers=2,
                              patch_size=(1, 2, 2),
                              rope_max_seq_len=64)
    config = WanVideoConfig(arch_config=arch)
    hf_config = {
        "num_attention_heads": arch.num_attention_heads,
        "attention_head_dim": arch.attention_head_dim,
        "in_channels": in_channels,
        "out_channels": Z_DIM,
        "text_dim": TEXT_DIM,
        "freq_dim": arch.freq_dim,
        "ffn_dim": arch.ffn_dim,
        "num_layers": arch.num_layers,
        "patch_size": arch.patch_size,
        "text_len": arch.text_len,
        "rope_max_seq_len": arch.rope_max_seq_len,
        "eps": 1e-6,
    }
    with envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"):
        model = cls(config=config, hf_config=hf_config)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim <= 1:
                p.fill_(1.0) if (name.endswith("weight") and "norm" in name) else p.normal_(0, 0.02)
            else:
                torch.nn.init.xavier_uniform_(p)
    return model.float().eval()


def _run(model, hidden, timestep, context):
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        return model(hidden_states=hidden, encoder_hidden_states=context, timestep=timestep)


@pytest.fixture
def mv_model(distributed_setup):
    return _build(MultiViewWanTransformer3DModel, Z_DIM + RAY)


def _inputs(B=2, V=3, F=3, H=4, W=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, V, Z_DIM + RAY, F, H, W, generator=g)
    ctx = torch.randn(B, TEXT_LEN, TEXT_DIM, generator=g)
    t = torch.rand(B, generator=g) * 1000
    return x, ctx, t


def test_single_view_matches_parent_forward(mv_model):
    x, ctx, t = _inputs(V=1)
    multi = _run(mv_model, x, t, ctx)
    single = _run(mv_model, x[:, 0], t, ctx)  # 5-D input -> stock WanTransformer3DModel.forward
    assert multi.shape == (2, 1, Z_DIM, 3, 4, 6)
    torch.testing.assert_close(multi[:, 0], single, rtol=1e-4, atol=1e-5)


def test_timestep_shapes_agree(mv_model):
    x, ctx, t = _inputs()
    scalar = _run(mv_model, x, t, ctx)
    per_frame = _run(mv_model, x, t[:, None].expand(-1, 3), ctx)
    per_view_frame = _run(mv_model, x, t[:, None, None].expand(-1, 3, 3), ctx)
    torch.testing.assert_close(scalar, per_frame)
    torch.testing.assert_close(scalar, per_view_frame)


def test_rejects_bad_timestep_shape(mv_model):
    x, ctx, t = _inputs()
    with pytest.raises(ValueError):
        _run(mv_model, x, t[:, None].expand(-1, 5), ctx)


def test_view_permutation_equivariance(mv_model):
    x, ctx, t = _inputs()
    perm = torch.tensor([2, 0, 1])
    out = _run(mv_model, x, t, ctx)
    out_perm = _run(mv_model, x[:, perm], t, ctx)
    torch.testing.assert_close(out_perm, out[:, perm], rtol=1e-4, atol=1e-5)


def test_views_attend_to_each_other(mv_model):
    x, ctx, t = _inputs()
    out = _run(mv_model, x, t, ctx)
    x2 = x.clone()
    x2[:, 1] += 1.0
    out2 = _run(mv_model, x2, t, ctx)
    assert (out2[:, 0] - out[:, 0]).abs().max() > 1e-4  # view 0 sees view 1


def test_clean_history_frame_changes_modulation(mv_model):
    x, ctx, t = _inputs()
    t_frames = t[:, None].expand(-1, 3).clone()
    hist = t_frames.clone()
    hist[:, 0] = 0.0
    assert not torch.allclose(_run(mv_model, x, t_frames, ctx), _run(mv_model, x, hist, ctx))


def test_zero_widened_patch_embedding_reproduces_base_model(distributed_setup):
    widen = _load_converter().widen_patch_embedding
    base = _build(WanTransformer3DModel, Z_DIM)
    multi = _build(MultiViewWanTransformer3DModel, Z_DIM + RAY)
    state = base.state_dict()
    key = next(k for k in state if k.startswith("patch_embedding") and k.endswith("weight"))
    state[key] = widen(state[key], RAY)
    multi.load_state_dict(state)
    x, ctx, t = _inputs(V=1)
    rays = torch.randn_like(x[:, :, Z_DIM:])
    x = torch.cat([x[:, :, :Z_DIM], rays], dim=2)
    torch.testing.assert_close(_run(multi, x, t, ctx)[:, 0], _run(base, x[:, 0, :Z_DIM], t, ctx), rtol=1e-4,
                               atol=1e-5)


def test_widen_patch_embedding_shape_and_zeros():
    widen = _load_converter().widen_patch_embedding
    w = torch.randn(8, 16, 1, 2, 2)
    out = widen(w, 6)
    assert out.shape == (8, 22, 1, 2, 2)
    torch.testing.assert_close(out[:, :16], w)
    assert torch.count_nonzero(out[:, 16:]) == 0


# ---- training helpers ----


def test_masked_flow_loss_ignores_history_frames():
    g = torch.Generator().manual_seed(0)
    lat, noise = torch.randn(2, 3, 4, 3, 2, 2, generator=g), torch.randn(2, 3, 4, 3, 2, 2, generator=g)
    target = noise - lat
    pred = target.clone()
    pred[:, :, :, 0] += 5.0  # wrong only on frame 0
    mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    assert masked_flow_loss(pred, lat, noise, mask).item() == pytest.approx(0.0, abs=1e-6)
    full = torch.ones(2, 3)
    assert masked_flow_loss(pred, lat, noise, full).item() == pytest.approx(25.0 / 3, rel=1e-5)


def test_masked_flow_loss_all_masked_is_finite():
    z = torch.zeros(1, 1, 1, 2, 1, 1)
    assert masked_flow_loss(z + 1, z, z, torch.zeros(1, 2)).item() == 0.0


def test_ssim_and_psnr_metrics():
    torch.manual_seed(0)
    img = torch.rand(4, 3, 16, 16)
    torch.testing.assert_close(ssim(img, img), torch.ones(4))
    assert torch.all(ssim(img, 1 - img) < 0.5)
    video = torch.rand(1, 2, 3, 3, 16, 16) * 2 - 1
    psnr_sum, ssim_sum = future_frame_metrics(video + 0.02, video)
    n = 1 * 2 * 2  # B * V * future frames
    assert psnr_sum.item() / n == pytest.approx(-10 * torch.log10(torch.tensor(0.01**2)).item(), rel=0.05)
    assert ssim_sum.item() / n > 0.9


def test_infinite_random_sampler_reshuffles_each_epoch():
    a, b = InfiniteRandomSampler(50, seed=3), InfiniteRandomSampler(50, seed=3)
    e0, e1 = list(a), list(a)
    assert sorted(e0) == list(range(50))
    assert e0 != e1
    assert list(b) == e0  # deterministic per seed
    assert list(InfiniteRandomSampler(50, seed=4)) != e0


def test_infinite_random_sampler_resumes_mid_epoch():
    a = InfiniteRandomSampler(20, seed=7)
    list(a)  # epoch 0
    it = iter(a)  # epoch 1
    head = [next(it) for _ in range(5)]
    state = a.state_dict()
    rest = list(it)
    b = InfiniteRandomSampler(20, seed=7)
    b.load_state_dict(state)
    assert list(b) == rest  # continues epoch 1 where it stopped
    assert sorted(head + rest) == list(range(20))


# ---- unseen views / point cloud ----


def test_masked_flow_loss_per_view_mask():
    g = torch.Generator().manual_seed(0)
    lat, noise = torch.randn(1, 2, 4, 2, 2, 2, generator=g), torch.randn(1, 2, 4, 2, 2, 2, generator=g)
    pred = (noise - lat).clone()
    pred[:, 0, :, 0] += 3.0  # wrong only on (view 0, frame 0)
    mask = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])  # view 0 frame 0 is clean history
    assert masked_flow_loss(pred, lat, noise, mask).item() == pytest.approx(0.0, abs=1e-6)
    assert masked_flow_loss(pred, lat, noise, torch.ones(1, 2, 2)).item() == pytest.approx(9.0 / 4, rel=1e-5)


def test_future_frame_metrics_respects_frame_mask():
    video = torch.zeros(1, 2, 2, 3, 16, 16)
    pred = video.clone()
    pred[:, 1] += 0.5  # only view 1 is wrong
    frames = torch.tensor([[[True, True], [False, False]]])
    psnr_sum, _ = future_frame_metrics(pred, video, frames)
    assert psnr_sum.item() / 2 == pytest.approx(100.0)  # clamped mse 1e-10
    none = torch.zeros(1, 2, 2, dtype=torch.bool)
    assert future_frame_metrics(pred, video, none)[0].item() == 0.0


class _StubSampler:
    """Just enough of WanMultiViewModel for `sample` (bound unbound)."""
    timestep_shift = 3.0
    num_train_timestep = 1000

    def __init__(self):
        self.seen_t = []

    def _null_text_dict(self, B):
        return {}

    def predict_flow(self, x, t, cond, text):
        self.seen_t.append(t.clone())
        return torch.ones_like(x)


def test_sample_keeps_only_context_history_clean():
    from fastvideo.train.models.wan.multiview import WanMultiViewModel
    stub = _StubSampler()
    lat = torch.randn(2, 3, 4, 3, 2, 2)
    ctx = torch.tensor([[True, False, True], [False, True, True]])
    out = WanMultiViewModel.sample(stub,
                                   lat,
                                   torch.zeros(2, 3, 6, 2, 2),
                                   n_hist=1,
                                   steps=4,
                                   generator=torch.Generator().manual_seed(0),
                                   context=ctx)
    hist = ctx[:, :, None] & (torch.arange(3) < 1)  # B, V, F
    out_f, lat_f = out.transpose(2, 3), lat.transpose(2, 3)  # B, V, F, C, h, w
    torch.testing.assert_close(out_f[hist], lat_f[hist])
    assert not torch.allclose(out_f[~hist], lat_f[~hist])
    for t in stub.seen_t:
        assert t.shape == (2, 3, 3)
        assert torch.all(t[hist] == 0) and torch.all(t[~hist] > 0)


class _StubPrepare:
    """Just enough of WanMultiViewModel for `prepare_batch`."""
    num_train_timestep = 1000
    n_hist_latent = 1
    design = "tokens"
    device = torch.device("cpu")

    def __init__(self, p_no_hist, point_cloud):
        self.p_no_hist, self.point_cloud = p_no_hist, point_cloud

    def ensure_null_context(self):
        pass

    def pack(self, raw):
        return torch.randn(2, 3, 4, 2, 2, 2), torch.ones(2, 3, 6 + 17, 2, 2, 2)

    def _sample_timesteps(self, B, device, generator):
        return torch.full((B, ), 500.0)

    def _null_text_dict(self, B):
        return {}


def _prepare(stub, monkeypatch):
    import fastvideo.train.models.wan.multiview as mv
    monkeypatch.setattr(mv, "get_sigmas", lambda sched, device, t, n_dim, dtype: (t / 1000)[:, None])
    stub.noise_scheduler = None
    stub.context_mask = lambda raw, n: mv.WanMultiViewModel.context_mask(stub, raw, n)
    raw = {"videos": torch.zeros(2, 3), "context": torch.tensor([[True, False, True], [False, True, False]])}
    return mv.WanMultiViewModel.prepare_batch(stub, raw, generator=torch.Generator().manual_seed(0)), raw


def test_prepare_batch_cleans_only_context_history(monkeypatch):
    tb, raw = _prepare(_StubPrepare(p_no_hist=0.0, point_cloud=True), monkeypatch)
    assert tb.sigmas.shape == (2, 3, 2) and tb.timesteps.shape == (2, 3, 2)
    hist = raw["context"][:, :, None] & (torch.arange(2) < 1)
    assert torch.all(tb.sigmas[hist] == 0) and torch.all(tb.sigmas[~hist] == 0.5)
    torch.testing.assert_close(tb.loss_mask, (~hist).float())
    torch.testing.assert_close(tb.noisy_model_input.transpose(2, 3)[hist], tb.latents.transpose(2, 3)[hist])
    assert torch.all(tb.raymaps == 1)  # point-cloud channels kept when history is given


def test_prepare_batch_no_history_drops_point_cloud(monkeypatch):
    tb, _ = _prepare(_StubPrepare(p_no_hist=1.0, point_cloud=True), monkeypatch)
    assert torch.all(tb.sigmas == 0.5) and torch.all(tb.loss_mask == 1)
    assert torch.all(tb.raymaps[:, :, :6] == 1) and torch.all(tb.raymaps[:, :, 6:] == 0)
