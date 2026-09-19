import threading
from collections import OrderedDict

import torch
import torch.nn.functional as F
import pytest

from distortion_library.mgtc_cascade_v1 import (
    mgtc_cascade_v1, MGTCCascadeV1,
    _cache_snapshot,
    _STRUCT_CACHE, _BLUR_CACHE, _CACHE_LOCK,
    _BLUR_MAX, _ALLOWED_KWARGS,
)


def _snapshot_and_clear(cache):
    with _CACHE_LOCK:
        saved = OrderedDict(cache)
        cache.clear()
    return saved


def _restore(cache, saved):
    with _CACHE_LOCK:
        cache.clear()
        cache.update(saved)


# ------------------------------------------------------------------ API signature
def test_wrapper_rejects_seed_kwarg():
    x = torch.rand(1, 1, 3, 16, 16)
    with pytest.raises(TypeError, match="unexpected keyword"):
        mgtc_cascade_v1(x, 0.5, seed=42)
    with pytest.raises(TypeError, match="unexpected keyword"):
        mgtc_cascade_v1(x, 0.5, generator=torch.Generator())


def test_wrapper_rejects_unknown_kwarg():
    x = torch.rand(1, 1, 3, 16, 16)
    with pytest.raises(TypeError, match="unexpected keyword"):
        mgtc_cascade_v1(x, 0.5, foobar=123)


def test_allowed_kwargs_match_init_signature():
    import inspect
    expected = frozenset(
        p for p in inspect.signature(MGTCCascadeV1.__init__).parameters
        if p != "self"
    )
    assert _ALLOWED_KWARGS == expected


def test_wrapper_accepts_all_documented_kwargs():
    """All documented kwargs accepted; calibration meaningfully changes output."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 3, 16, 16)

    base = dict(block_size=4, i_period=2, deblock_sigma=0.3,
                max_flow=4.0, debug=False, use_ste=True)

    y1, _ = mgtc_cascade_v1(x, 0.5, calibration={"qp": (10.0, 20.0, 1.0)}, **base)
    assert y1.shape == x.shape and torch.isfinite(y1).all()

    y2, _ = mgtc_cascade_v1(x, 0.5, calibration={"qp": (10.0, 45.0, 1.0)}, **base)
    assert not torch.allclose(y1, y2, atol=1e-6), \
        "calibration['qp'] must affect output"
def test_value_range_is_third_positional():
    x = (torch.rand(1, 1, 3, 16, 16) * 255).to(torch.uint8)
    y, _ = mgtc_cascade_v1(x, 0.5, (0, 255))
    assert y.shape == x.shape and y.dtype == torch.uint8


# ------------------------------------------------------------------ range / dtype
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(3, 32, 32), (2, 3, 32, 32), (2, 4, 3, 32, 32)])
def test_shape_dtype_range(shape, dtype):
    x = torch.rand(*shape, dtype=dtype)
    y, _ = mgtc_cascade_v1(x, 0.5)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.isfinite(y).all()
    assert y.min() >= -1e-5 and y.max() <= 1.0 + 1e-5


# ------------------------------------------------------------------ memory formats
def test_channels_last_preserved():
    x = torch.rand(2, 3, 32, 32).contiguous(memory_format=torch.channels_last)
    y, _ = mgtc_cascade_v1(x, 0.5)
    assert y.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch, "channels_last_3d"),
    reason="CUDA + channels_last_3d required",
)
def test_channels_last_3d_preserved():
    x = torch.rand(1, 2, 3, 32, 32, device="cuda").contiguous(
        memory_format=torch.channels_last_3d)
    y, _ = mgtc_cascade_v1(x, 0.5)
    assert y.is_contiguous(memory_format=torch.channels_last_3d)


# ------------------------------------------------------------------ fp16 / bf16
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_works_on_gpu():
    x = torch.rand(1, 2, 3, 32, 32, device="cuda", dtype=torch.float16)
    y, _ = mgtc_cascade_v1(x, 0.5)
    assert y.dtype == torch.float16 and torch.isfinite(y.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 requires CUDA")
def test_bf16_works_on_gpu():
    x = torch.rand(1, 2, 3, 32, 32, device="cuda", dtype=torch.bfloat16)
    y, _ = mgtc_cascade_v1(x, 0.5)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()


# ------------------------------------------------------------------ identity
def test_identity_at_minimum_severity():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    y, _ = mgtc_cascade_v1(x, 0.01)
    assert F.mse_loss(y, x).item() < 0.01


# ------------------------------------------------------------------ monotonicity
def test_monotonicity_of_mse():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 3, 64, 64)
    mses = [F.mse_loss(mgtc_cascade_v1(x, s)[0], x).item()
            for s in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
    for a, b in zip(mses, mses[1:]):
        assert b + 1e-3 >= a, mses


# ------------------------------------------------------------------ gradients
def test_differentiable_wrt_input():
    x = torch.rand(1, 2, 3, 32, 32, requires_grad=True)
    mgtc_cascade_v1(x, 0.6)[0].mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_differentiable_wrt_severity():
    x = torch.rand(1, 2, 3, 32, 32)
    s = torch.tensor([0.6], requires_grad=True)
    mgtc_cascade_v1(x, s)[0].mean().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


# ------------------------------------------------------------------ batch invariance
def test_batch_invariance_5d():
    torch.manual_seed(0)
    x = torch.rand(4, 2, 3, 32, 32)
    s = torch.tensor([0.2, 0.4, 0.6, 0.8])
    yb, _ = mgtc_cascade_v1(x, s)
    for i in range(4):
        yi, _ = mgtc_cascade_v1(x[i:i+1], s[i:i+1])
        assert torch.allclose(yb[i:i+1], yi, atol=1e-4)


def test_batch_invariance_4d():
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32)
    s = torch.tensor([0.2, 0.4, 0.6, 0.8])
    yb, _ = mgtc_cascade_v1(x, s)
    for i in range(4):
        yi, _ = mgtc_cascade_v1(x[i:i+1], s[i:i+1])
        assert torch.allclose(yb[i:i+1], yi, atol=1e-4)


# ------------------------------------------------------------------ determinism
def test_determinism():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 3, 32, 32)
    y1, _ = mgtc_cascade_v1(x, 0.5)
    y2, _ = mgtc_cascade_v1(x, 0.5)
    assert torch.equal(y1, y2)


# ------------------------------------------------------------------ labels
def test_label_shape_and_value():
    x = torch.rand(2, 3, 3, 32, 32)
    _, lab = mgtc_cascade_v1(x, 0.5)
    assert lab.shape == (2, 2) and lab.device == x.device
    assert (lab[:, 0] == MGTCCascadeV1.DISTORTION_ID).all()

    x3 = torch.rand(3, 32, 32)
    _, lab3 = mgtc_cascade_v1(x3, 0.7)
    assert lab3.shape == (2,) and lab3.device == x3.device
    assert lab3[0].item() == MGTCCascadeV1.DISTORTION_ID
    assert abs(lab3[1].item() - 0.7) < 1e-6


# ------------------------------------------------------------------ severity clamp
def test_severity_clamp():
    x = torch.rand(1, 1, 3, 32, 32)
    assert torch.allclose(mgtc_cascade_v1(x, -5.0)[0],
                          mgtc_cascade_v1(x, 0.01)[0], atol=1e-4)
    assert torch.allclose(mgtc_cascade_v1(x, 100.0)[0],
                          mgtc_cascade_v1(x, 1.0)[0], atol=1e-4)


# ------------------------------------------------------------------ gradcheck / gradgradcheck
@pytest.mark.parametrize("severity", [0.3, 0.7])
def test_gradcheck_fp64(severity):
    torch.manual_seed(0)
    module = MGTCCascadeV1(
        calibration={"N_eff": (1.0, 2.0, 1.0)},
        use_ste=False,
    )

    def f(xx, ss):
        return module(xx, ss)

    x = (0.4 + 0.2 * torch.rand(1, 1, 1, 16, 16, dtype=torch.float64)
         ).requires_grad_(True)
    s = torch.tensor([severity], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (x, s), eps=1e-6,
                                    atol=5e-3, rtol=5e-3)


def test_gradgradcheck_fp64():
    torch.manual_seed(0)
    module = MGTCCascadeV1(
        calibration={"N_eff": (1.0, 2.0, 1.0)},
        use_ste=False,
    )

    def f(xx, ss):
        return module(xx, ss)

    x = (0.45 + 0.1 * torch.rand(1, 1, 1, 8, 8, dtype=torch.float64)
         ).requires_grad_(True)
    s = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(f, (x, s), eps=1e-6,
                                        atol=5e-3, rtol=5e-3)


# ------------------------------------------------------------------ validation
def test_i_period_validation_class():
    with pytest.raises(AssertionError):
        MGTCCascadeV1(i_period=0)


def test_i_period_validation_wrapper():
    x = torch.rand(1, 1, 3, 16, 16)
    with pytest.raises(ValueError):
        mgtc_cascade_v1(x, 0.5, i_period=0)


def test_n_eff_lo_validated():
    with pytest.raises(AssertionError, match="N_eff.lo"):
        MGTCCascadeV1(calibration={"N_eff": (0.5, 3.0, 1.0)})


def test_n_eff_hi_validated():
    with pytest.raises(AssertionError, match="N_eff.hi"):
        MGTCCascadeV1(calibration={"N_eff": (1.0, 0.5, 1.0)})


def test_deblock_sigma_negative_class():
    with pytest.raises(AssertionError, match="deblock_sigma"):
        MGTCCascadeV1(deblock_sigma=-0.1)


def test_deblock_sigma_zero_is_valid():
    m = MGTCCascadeV1(deblock_sigma=0.0)
    assert m.deblock_sigma == 0.0
    m2 = MGTCCascadeV1(deblock_sigma=-0.0)
    assert m2.deblock_sigma == 0.0


def test_deblock_sigma_negative_wrapper():
    x = torch.rand(1, 1, 3, 16, 16)
    with pytest.raises(ValueError, match="deblock_sigma"):
        mgtc_cascade_v1(x, 0.5, deblock_sigma=-0.1)


# ------------------------------------------------------------------ forward strictness
def test_forward_rejects_batch_mismatch():
    module = MGTCCascadeV1()
    x = torch.rand(4, 1, 3, 16, 16)
    s_bad = torch.tensor([0.5, 0.7])
    with pytest.raises(ValueError, match="severity has 2 values"):
        module(x, s_bad)


def test_forward_accepts_scalar():
    module = MGTCCascadeV1()
    x = torch.rand(4, 1, 3, 16, 16)
    y = module(x, torch.tensor(0.5))
    assert y.shape == x.shape


def test_forward_accepts_single_element_tensor():
    module = MGTCCascadeV1()
    x = torch.rand(4, 1, 3, 16, 16)
    y = module(x, torch.tensor([0.5]))
    assert y.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_moves_severity_to_device():
    module = MGTCCascadeV1().cuda()
    x = torch.rand(2, 1, 3, 16, 16, device="cuda")
    s_cpu = torch.tensor([0.5, 0.7])
    y = module(x, s_cpu)
    assert y.shape == x.shape and y.device.type == "cuda"


def test_wrapper_rejects_batch_mismatch():
    x = torch.rand(4, 1, 3, 16, 16)
    with pytest.raises(ValueError, match="severity has 2 values"):
        mgtc_cascade_v1(x, torch.tensor([0.5, 0.7]))


# ------------------------------------------------------------------ warp internals
def test_warp_uses_border_padding_not_zeros():
    from distortion_library.mgtc_cascade_v1 import _warp
    x = torch.ones(1, 1, 8, 8)
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 100.0
    y = _warp(x, flow)
    assert y.min().item() > 0.9


def test_warp_small_flow():
    from distortion_library.mgtc_cascade_v1 import _warp
    x = torch.rand(1, 1, 8, 8)
    flow = torch.full((1, 2, 8, 8), 5.0)
    y = _warp(x, flow)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_warp_shape_check():
    from distortion_library.mgtc_cascade_v1 import _warp
    x = torch.zeros(1, 1, 8, 8)
    with pytest.raises(AssertionError):
        _warp(x, torch.zeros(1, 3, 8, 8))


# ------------------------------------------------------------------ flow internals
def test_flow_gradient_survives_saturation():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    ramp = torch.linspace(0, 1, 16).view(1, 1, 1, 16).expand(1, 1, 16, 16).clone()
    prev = ramp.clone().requires_grad_(True)
    curr = torch.roll(ramp, shifts=8, dims=-1).clone().requires_grad_(True)

    max_flow = 0.05
    flow = _estimate_flow(prev, curr, K=4, max_flow=max_flow)
    assert flow.shape == (1, 2, 16, 16)

    max_mag = flow.abs().max().item()
    assert max_mag >= 0.9 * max_flow

    flow.sum().backward()
    assert prev.grad is not None and torch.isfinite(prev.grad).all()
    assert curr.grad is not None and torch.isfinite(curr.grad).all()


def test_flow_returns_fp32_for_fp16_input():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    prev = torch.rand(1, 1, 16, 16, dtype=torch.float16, requires_grad=True)
    curr = torch.rand(1, 1, 16, 16, dtype=torch.float16)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0)
    assert flow.dtype == torch.float32
    flow.sum().backward()
    assert prev.grad is not None and torch.isfinite(prev.grad).all()


def test_flow_mixed_dtype_promotion():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    prev = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    curr = torch.rand(1, 1, 16, 16, dtype=torch.float16)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0)
    assert flow.dtype == torch.float32
    assert torch.isfinite(flow).all()


def test_flow_mixed_width_promotes_to_fp64():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow

    prev = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    curr = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0)
    assert flow.dtype == torch.float64
    assert torch.isfinite(flow).all()

    prev2 = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    curr2 = torch.rand(1, 1, 16, 16, dtype=torch.float32)
    flow2 = _estimate_flow(prev2, curr2, K=4, max_flow=2.0)
    assert flow2.dtype == torch.float64
    assert torch.isfinite(flow2).all()


def test_flow_both_fp64_returns_fp64():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    prev = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    curr = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0)
    assert flow.dtype == torch.float64


def test_flow_fp16_cpu():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    prev = torch.rand(1, 1, 16, 16, dtype=torch.float16)
    curr = torch.rand(1, 1, 16, 16, dtype=torch.float16)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0)
    assert flow.dtype == torch.float32
    assert torch.isfinite(flow).all()


def test_flow_bf16_cpu():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    prev = torch.rand(1, 1, 16, 16, dtype=torch.bfloat16)
    curr = torch.rand(1, 1, 16, 16, dtype=torch.bfloat16)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0)
    assert flow.dtype == torch.float32
    assert torch.isfinite(flow).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 requires CUDA")
def test_flow_fp16_no_overflow():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    prev = torch.rand(1, 1, 16, 16, device="cuda", dtype=torch.float16)
    curr = torch.rand(1, 1, 16, 16, device="cuda", dtype=torch.float16)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0)
    assert flow.dtype == torch.float32 and torch.isfinite(flow).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 requires CUDA")
def test_flow_bf16_no_overflow():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    prev = torch.rand(1, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
    curr = torch.rand(1, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0)
    assert flow.dtype == torch.float32 and torch.isfinite(flow).all()


def test_flow_user_eps_preserved():
    from distortion_library.mgtc_cascade_v1 import _estimate_flow
    prev = torch.rand(1, 1, 16, 16, dtype=torch.float16)
    curr = torch.rand(1, 1, 16, 16, dtype=torch.float16)
    flow = _estimate_flow(prev, curr, K=4, max_flow=2.0, eps=1e-4)
    assert torch.isfinite(flow).all()


# ------------------------------------------------------------------ block sizes
@pytest.mark.parametrize("K", [4, 6, 8, 16])
def test_block_size_variants(K):
    torch.manual_seed(0)
    x = torch.rand(1, 1, 3, 32, 32)
    y, _ = mgtc_cascade_v1(x, 0.6, block_size=K)
    assert y.shape == x.shape and torch.isfinite(y).all()


# ------------------------------------------------------------------ integer dtype
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16])
def test_integer_dtype_rounding(dtype):
    x = (torch.rand(1, 1, 3, 32, 32) * 255).to(dtype)
    y, _ = mgtc_cascade_v1(x, 0.5, value_range=(0, 255))
    assert y.dtype == dtype
    assert torch.isfinite(y.float()).all()


# ------------------------------------------------------------------ caches
def test_blur_kernel_cached():
    from distortion_library.mgtc_cascade_v1 import _gaussian_blur
    saved = _snapshot_and_clear(_BLUR_CACHE)
    try:
        sigma = 0.777
        x = torch.rand(1, 3, 32, 32)
        _gaussian_blur(x, sigma=sigma)
        with _CACHE_LOCK:
            keys = list(_BLUR_CACHE.keys())
        assert any(k[0] == "blur" and k[1] == round(sigma, 6) for k in keys)
    finally:
        _restore(_BLUR_CACHE, saved)


def test_blur_r_computed_once(monkeypatch):
    from distortion_library.mgtc_cascade_v1 import _gaussian_blur, _make_blur_kernel
    saved = _snapshot_and_clear(_BLUR_CACHE)
    try:
        sigma = 0.321
        calls = {"n": 0}
        orig = _make_blur_kernel
        def counted(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)
        monkeypatch.setattr("distortion_library.mgtc_cascade_v1._make_blur_kernel", counted)

        x = torch.rand(1, 3, 32, 32)
        _gaussian_blur(x, sigma=sigma)
        _gaussian_blur(x, sigma=sigma)
        assert calls["n"] == 1
    finally:
        _restore(_BLUR_CACHE, saved)


def test_cache_is_bounded(monkeypatch):
    import distortion_library.mgtc_cascade_v1 as mod
    from distortion_library.mgtc_cascade_v1 import _get_dct
    monkeypatch.setattr(mod, "_STRUCT_MAX", 8)
    saved = _snapshot_and_clear(_STRUCT_CACHE)
    try:
        for K in range(2, 2 + 8 + 2):
            _get_dct(K, torch.device("cpu"), torch.float32)
        with _CACHE_LOCK:
            assert len(_STRUCT_CACHE) <= 8
    finally:
        _restore(_STRUCT_CACHE, saved)


def test_cache_race_returns_same_object():
    from distortion_library.mgtc_cascade_v1 import _cache_get_impl
    saved = _snapshot_and_clear(_STRUCT_CACHE)
    try:
        results = []
        def worker():
            results.append(
                _cache_get_impl(_STRUCT_CACHE, ("race_test",),
                                lambda: object(), 256)
            )
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len({id(r) for r in results}) == 1
    finally:
        _restore(_STRUCT_CACHE, saved)


def test_cache_snapshot_threadsafe():
    _cache_snapshot()


def test_blur_cache_bound():
    assert _BLUR_MAX >= 32
