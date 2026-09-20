"""Test suite for hdr_mef_noise_mismatch_v1 (v1.1.7)."""
import warnings

import pytest
import torch

from distortion_library.hdr_mef_noise_mismatch_v1 import (
    hdr_mef_noise_mismatch_v1,
    HdrMefNoiseMismatchV1,
    DISTORTION_ID,
    DISTORTION_NAME,
    DISTORTION_VERSION,
    DISTORTION_REGISTRY,
    _KERNEL_CACHE,
    _get_gauss_kernel,
    _device_key,
    _NO_INDEX,
)


@pytest.fixture
def img4d():
    torch.manual_seed(0)
    return torch.rand(2, 3, 64, 64)


@pytest.fixture
def img3d():
    torch.manual_seed(0)
    return torch.rand(3, 64, 64)


@pytest.fixture
def vid5d():
    torch.manual_seed(0)
    return torch.rand(2, 4, 3, 48, 48)


# Registry / metadata
def test_registry_present():
    assert DISTORTION_NAME == "hdr_mef_noise_mismatch_v1"
    assert DISTORTION_NAME in DISTORTION_REGISTRY
    assert DISTORTION_REGISTRY[DISTORTION_NAME] is hdr_mef_noise_mismatch_v1


def test_function_attributes_exposed():
    assert hdr_mef_noise_mismatch_v1.DISTORTION_NAME == DISTORTION_NAME
    assert hdr_mef_noise_mismatch_v1.DISTORTION_VERSION == DISTORTION_VERSION


def test_metadata():
    assert DISTORTION_VERSION == "1.1.7"
    assert DISTORTION_ID == 1009
    as_f32 = torch.tensor([float(DISTORTION_ID)], dtype=torch.float32)
    assert int(as_f32.item()) == DISTORTION_ID


def test_module_attributes_exposed():
    assert HdrMefNoiseMismatchV1.DISTORTION_NAME == DISTORTION_NAME
    assert HdrMefNoiseMismatchV1.DISTORTION_VERSION == DISTORTION_VERSION
    assert HdrMefNoiseMismatchV1.DISTORTION_ID == DISTORTION_ID


def test_register_distortion_not_in_public_all():
    # `from distortion_library import X` returns the FUNCTION (shadowed by
    # __init__.py); use importlib to get the MODULE explicitly.
    import importlib
    mod = importlib.import_module("distortion_library.hdr_mef_noise_mismatch_v1")
    assert "register_distortion" not in mod.__all__
    assert hasattr(mod, "register_distortion")


# Device key normalization
def test_device_key_normalization():
    assert _device_key("cpu") == _device_key(torch.device("cpu"))
    assert _device_key("cpu") == ("cpu", _NO_INDEX)
    assert _device_key("cuda:0") == ("cuda", 0)
    assert _device_key("cuda:1") == ("cuda", 1)
    assert _device_key("cuda:1") != _device_key("cuda:0")
    expected = torch.cuda.current_device() if torch.cuda.is_available() else 0
    assert _device_key("cuda") == ("cuda", expected)
    assert _device_key("cuda") == _device_key(f"cuda:{expected}")


def test_device_key_matches_torch_semantics():
    assert _device_key("cuda:0") == _device_key(torch.device("cuda", 0))
    assert _device_key("cuda:1") != _device_key("cuda:0")
    assert _device_key("cpu") != _device_key("cuda")


def test_device_key_cuda_respects_current_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if torch.cuda.device_count() < 2:
        with torch.cuda.device(0):
            current = torch.cuda.current_device()
            assert _device_key("cuda") == ("cuda", current)
        return
    with torch.cuda.device(1):
        assert torch.cuda.current_device() == 1
        assert _device_key("cuda") == ("cuda", 1)
    with torch.cuda.device(0):
        assert _device_key("cuda") == ("cuda", 0)


def test_device_key_cuda_fallback_on_no_cuda(monkeypatch):
    def raise_no_cuda():
        raise RuntimeError("CUDA not available")
    monkeypatch.setattr(torch.cuda, "current_device", raise_no_cuda)
    assert _device_key("cuda") == ("cuda", 0)


# Shapes / labels
def test_shape_and_dtype(img4d):
    y, label = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=42)
    assert y.shape == img4d.shape
    assert y.dtype == img4d.dtype
    # 4D input with B=2 → label (B, 2) = (2, 2)
    assert label.shape == (2, 2)


def test_single_image(img3d):
    y, label = hdr_mef_noise_mismatch_v1(img3d, severity=0.5, seed=42)
    assert y.shape == img3d.shape
    assert label.shape == (2,)


def test_batch1_4d_label_is_b2(img4d):
    x = img4d[:1]
    y, label = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=42)
    assert y.shape == x.shape
    assert label.shape == (1, 2)


def test_video_shape(vid5d):
    y, label = hdr_mef_noise_mismatch_v1(vid5d, severity=0.5, seed=42)
    assert y.shape == vid5d.shape
    # 5D input with B=2 → label (B, 2) = (2, 2)
    assert label.shape == (2, 2)


def test_video_batch1_label_is_b2():
    x = torch.rand(1, 3, 3, 32, 32)
    _, label = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    assert label.shape == (1, 2)


# Range / identity / monotonicity
def test_range(img4d):
    for s in [0.01, 0.1, 0.5, 1.0]:
        y, _ = hdr_mef_noise_mismatch_v1(img4d, severity=s, seed=42)
        assert torch.isfinite(y).all()
        assert (y >= -1e-6).all() and (y <= 1.0 + 1e-6).all()


def test_range_neg1_1(img4d):
    x = img4d * 2 - 1
    y, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=42,
                                     value_range=(-1.0, 1.0))
    assert torch.isfinite(y).all()
    assert (y >= -1.0 - 1e-6).all() and (y <= 1.0 + 1e-6).all()


def test_value_range_inverted(img4d):
    x = img4d
    y_normal, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, value_range=(0.0, 1.0)
    )
    y_invert, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, value_range=(1.0, 0.0)
    )
    assert torch.equal(y_normal, y_invert)


def test_value_range_inverted_neg1_1(img4d):
    x = img4d * 2 - 1
    y_normal, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, value_range=(-1.0, 1.0)
    )
    y_invert, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, value_range=(1.0, -1.0)
    )
    assert torch.equal(y_normal, y_invert)


def test_identity_at_low_severity(img4d):
    y, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.01, seed=42)
    mse = ((y - img4d) ** 2).mean().item()
    assert mse < 0.01


def test_monotonicity(img4d):
    ms = []
    for s in [0.1, 0.3, 0.5, 0.7, 1.0]:
        y, _ = hdr_mef_noise_mismatch_v1(img4d, severity=s, seed=42)
        ms.append(((y - img4d) ** 2).mean().item())
    for a, b in zip(ms, ms[1:]):
        assert a <= b + 1e-5, f"non-monotonic MSE: {ms}"


# Determinism
def test_determinism(img4d):
    y1, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=123)
    y2, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=123)
    assert torch.equal(y1, y2)


def test_seed_changes_output(img4d):
    y1, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=1)
    y2, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=2)
    assert not torch.allclose(y1, y2, atol=1e-6)


def test_generator_reproducible(img4d):
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    y1, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, generator=g1)
    y2, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, generator=g2)
    assert torch.equal(y1, y2)


# Batch-position determinism
def test_sample0_reproducibility_across_batches(img4d):
    y_full, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=99)
    y0, _ = hdr_mef_noise_mismatch_v1(img4d[0:1], severity=0.5, seed=99)
    assert torch.allclose(y_full[0], y0[0], atol=1e-4, rtol=1e-4)


def test_sample1_position_keyed(img4d):
    y_full, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=99)
    y_solo, _ = hdr_mef_noise_mismatch_v1(img4d[1:2], severity=0.5, seed=99)
    assert not torch.allclose(y_full[1], y_solo[0], atol=1e-3)


def test_severity_per_sample():
    x = torch.rand(3, 3, 32, 32)
    s = torch.tensor([0.1, 0.5, 0.9])
    y, label = hdr_mef_noise_mismatch_v1(x, severity=s, seed=11)
    assert y.shape == x.shape
    assert label.shape == (3, 2)
    assert torch.allclose(label[:, 1], s, atol=1e-6)


def test_severity_accepts_4d_shape():
    x = torch.rand(3, 3, 32, 32)
    s = torch.tensor([0.1, 0.5, 0.9]).view(3, 1, 1, 1)
    y, label = hdr_mef_noise_mismatch_v1(x, severity=s, seed=11)
    assert y.shape == x.shape
    assert label.shape == (3, 2)
    assert torch.allclose(
        label[:, 1], torch.tensor([0.1, 0.5, 0.9]), atol=1e-6
    )


def test_severity_accepts_list():
    x = torch.rand(3, 3, 32, 32)
    y, label = hdr_mef_noise_mismatch_v1(x, severity=[0.1, 0.5, 0.9], seed=0)
    assert y.shape == x.shape
    assert label.shape == (3, 2)
    assert torch.allclose(
        label[:, 1], torch.tensor([0.1, 0.5, 0.9]), atol=1e-6
    )


def test_severity_accepts_tuple():
    x = torch.rand(3, 3, 32, 32)
    y, label = hdr_mef_noise_mismatch_v1(x, severity=(0.1, 0.5, 0.9), seed=0)
    assert y.shape == x.shape
    assert label.shape == (3, 2)
    assert torch.allclose(
        label[:, 1], torch.tensor([0.1, 0.5, 0.9]), atol=1e-6
    )


def test_severity_accepts_numpy_array():
    np = pytest.importorskip("numpy")
    x = torch.rand(3, 3, 32, 32)
    s = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    y, label = hdr_mef_noise_mismatch_v1(x, severity=s, seed=0)
    assert label.shape == (3, 2)
    assert torch.allclose(
        label[:, 1], torch.tensor([0.1, 0.5, 0.9]), atol=1e-6
    )


def test_severity_clamped_below_min():
    x = torch.rand(1, 3, 32, 32)
    _, label = hdr_mef_noise_mismatch_v1(x, severity=-5.0, seed=0)
    assert label[0, 1].item() >= 0.01 - 1e-6


def test_severity_clamped_above_max():
    x = torch.rand(1, 3, 32, 32)
    _, label = hdr_mef_noise_mismatch_v1(x, severity=99.0, seed=0)
    assert label[0, 1].item() <= 1.0 + 1e-6


# Labels
def test_label_id():
    x = torch.rand(2, 3, 32, 32)
    _, label = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    assert label.shape == (2, 2)
    assert torch.allclose(label[:, 0], torch.full((2,), float(DISTORTION_ID)))


def test_label_detached_cpu():
    x = torch.rand(3, 32, 32)
    _, label = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    assert not label.requires_grad
    assert label.device.type == "cpu"
    assert label.dtype == torch.float32


# Memory format / dtype
def test_channels_last_preserved():
    x = torch.rand(2, 3, 32, 32).to(memory_format=torch.channels_last)
    y, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    assert y.is_contiguous(memory_format=torch.channels_last)


def test_channels_last_3d_preserved():
    if not hasattr(torch, "channels_last_3d"):
        pytest.skip("channels_last_3d unavailable")
    x = torch.rand(1, 4, 3, 32, 32).to(memory_format=torch.channels_last_3d)
    y, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    assert y.shape == x.shape
    assert y.is_contiguous(memory_format=torch.channels_last_3d)


def test_dtype_preserved_fp16():
    x = torch.rand(1, 3, 32, 32, dtype=torch.float16)
    y, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()


def test_dtype_preserved_fp64():
    x = torch.rand(1, 3, 32, 32, dtype=torch.float64)
    y, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    assert y.dtype == torch.float64


# compute_dtype
def test_compute_dtype_explicit_fp64():
    x = torch.rand(1, 3, 32, 32, dtype=torch.float32)
    y, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, compute_dtype=torch.float64
    )
    assert y.dtype == torch.float32
    assert y.shape == x.shape


def test_compute_dtype_default_preserves_fp64():
    from distortion_library.hdr_mef_noise_mismatch_v1 import _resolve_compute_dtype
    x64 = torch.rand(1, 3, 32, 32, dtype=torch.float64)
    x32 = torch.rand(1, 3, 32, 32, dtype=torch.float32)
    x16 = torch.rand(1, 3, 32, 32, dtype=torch.float16)
    assert _resolve_compute_dtype(x64, None) == torch.float64
    assert _resolve_compute_dtype(x32, None) == torch.float32
    assert _resolve_compute_dtype(x16, None) == torch.float32


def test_compute_dtype_invalid_type():
    x = torch.rand(1, 3, 32, 32)
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(
            x, severity=0.5, seed=0, compute_dtype="float32"
        )


def test_compute_dtype_non_float_dtype_rejected():
    x = torch.rand(1, 3, 32, 32)
    for bad in (torch.int32, torch.int64, torch.bool, torch.uint8):
        with pytest.raises(ValueError):
            hdr_mef_noise_mismatch_v1(
                x, severity=0.5, seed=0, compute_dtype=bad
            )


def test_compute_dtype_in_allowed_kwargs():
    x = torch.rand(1, 3, 32, 32)
    y, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, compute_dtype=torch.float32
    )
    assert y.shape == x.shape


@pytest.mark.parametrize("cdt", [torch.float16, torch.bfloat16])
def test_compute_dtype_low_precision_paths(cdt):
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32, dtype=torch.float32)
    y, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, compute_dtype=cdt
    )
    assert y.dtype == torch.float32
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert (y >= 0.0).all() and (y <= 1.0).all()
    max_delta = (y - x).abs().max().item()
    assert max_delta > 1e-4, (
        f"compute_dtype={cdt} produced no visible noise: max|y-x|={max_delta:.3e}"
    )


def test_fp64_default_actually_uses_fp64_internally():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64, dtype=torch.float64)
    y_default, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    y_forced_fp32, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, compute_dtype=torch.float32
    )
    assert y_default.dtype == torch.float64
    assert y_forced_fp32.dtype == torch.float64
    diff = (y_default - y_forced_fp32).abs().max().item()
    assert diff > 1e-6, (
        f"compute_dtype=torch.float32 had no effect: pipeline is not "
        f"actually running in fp64 by default (diff={diff:.3e})"
    )


# Differentiability
def test_differentiable_input():
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    y, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.shape == x.shape


def test_differentiable_severity():
    x = torch.rand(2, 3, 32, 32)
    s = torch.tensor(0.5, requires_grad=True)
    y, _ = hdr_mef_noise_mismatch_v1(x, severity=s, seed=0)
    y.sum().backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()


def test_gradcheck_fp64_tight_tolerances():
    g = torch.Generator().manual_seed(1234)
    x = torch.rand(1, 2, 8, 8, generator=g, dtype=torch.float64).clamp(0.1, 0.9)
    x.requires_grad_(True)
    s = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def fn(xi, si):
        y, _ = hdr_mef_noise_mismatch_v1(xi, severity=si, seed=0)
        return y

    assert torch.autograd.gradcheck(fn, (x, s), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_fp64_forced_fp32():
    """fp64 input with fp32 internal compute.

    The internal pipeline runs at fp32 precision, so finite-difference
    perturbations must exceed fp32 epsilon (~1.2e-7) to survive the
    downcast. eps=1e-4 is large enough to be meaningful at fp32 while
    remaining small enough to stay in the linear regime of the pipeline.
    """
    x = torch.rand(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)
    s = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def fn(xi, si):
        y, _ = hdr_mef_noise_mismatch_v1(
            xi, severity=si, seed=0, compute_dtype=torch.float32
        )
        return y

    assert torch.autograd.gradcheck(fn, (x, s), eps=1e-4, atol=1e-3, rtol=1e-2)


def test_gradgradcheck_fp64():
    g = torch.Generator().manual_seed(1234)
    x = torch.rand(1, 2, 8, 8, generator=g, dtype=torch.float64).clamp(0.1, 0.9)
    x.requires_grad_(True)
    s = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def fn(xi, si):
        y, _ = hdr_mef_noise_mismatch_v1(xi, severity=si, seed=0)
        return y

    assert torch.autograd.gradgradcheck(fn, (x, s), eps=1e-6, atol=1e-2, rtol=1e-1)


# Strict kwargs & validation
def test_unknown_kwarg_rejected(img4d):
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, foobar=42)


def test_kwarg_wrong_type_rejected(img4d):
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, K="abc")
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, gamma="1.0")


def test_kwarg_bool_rejected(img4d):
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, K=True)
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, gamma=False)


def test_kwarg_numpy_scalar_accepted(img4d):
    np = pytest.importorskip("numpy")
    y1, _ = hdr_mef_noise_mismatch_v1(
        img4d, severity=0.5, seed=0, gamma=np.float64(1.5)
    )
    assert y1.shape == img4d.shape
    y2, _ = hdr_mef_noise_mismatch_v1(
        img4d, severity=0.5, seed=0, K=np.int64(5)
    )
    assert y2.shape == img4d.shape
    y3, _ = hdr_mef_noise_mismatch_v1(
        img4d, severity=0.5, seed=0, lam=np.float32(0.3)
    )
    assert y3.shape == img4d.shape
    assert not isinstance(np.float32(0.5), float)
    assert not isinstance(np.int64(3), int)


def test_kwarg_numpy_non_integer_K_rejected(img4d):
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, K=np.float64(3.7))


def test_K_float_integer_accepted(img4d):
    y, _ = hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, K=3.0)
    assert y.shape == img4d.shape


def test_K_float_noninteger_rejected(img4d):
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, K=3.7)
    with pytest.raises(TypeError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, K=2.999)


def test_rho_t_out_of_range_high(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, rho_t=1.5)


def test_rho_t_out_of_range_low(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, rho_t=-0.1)


def test_rho_f_out_of_range(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, rho_f=-0.1)


def test_lam_out_of_range(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, lam=1.5)


def test_tau_nonpositive(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, tau=0.0)


def test_L_max_nonpositive(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, L_max=0.0)


def test_K_below_one(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, K=0)


def test_gamma_nonpositive(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, gamma=0.0)
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, gamma=-1.0)


def test_kappa_negative(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, kappa=-0.1)


def test_sigma_negative(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, sigma_r=-1e-3)
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0, sigma_q=-1e-3)


def test_ev_min_greater_than_ev_max(img4d):
    with pytest.raises(ValueError):
        hdr_mef_noise_mismatch_v1(img4d, severity=0.5, seed=0,
                                  ev_min=2.0, ev_max=-2.0)


def test_invalid_ndim():
    for bad_shape in [(32,), (3, 32), (1, 1, 3, 32, 32, 32)]:
        x = torch.rand(*bad_shape)
        with pytest.raises(ValueError):
            hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0)


# Video / temporal correlation
def _adjacent_noise_corr(x: torch.Tensor, *, rho_t: float, seed: int = 5,
                         severity: float = 0.8, **kwargs) -> float:
    y, _ = hdr_mef_noise_mismatch_v1(
        x, severity=severity, seed=seed, rho_t=rho_t, **kwargs
    )
    noise = (y - x).reshape(x.shape[0], x.shape[1], -1)
    corrs = []
    for t in range(x.shape[1] - 1):
        a = noise[0, t] - noise[0, t].mean()
        b = noise[0, t + 1] - noise[0, t + 1].mean()
        corrs.append(((a * b).sum() / (a.norm() * b.norm() + 1e-8)).item())
    return sum(corrs) / len(corrs)


def test_video_temporal_correlation():
    x = torch.full((1, 6, 3, 48, 48), 0.5)
    quiet = dict(kappa=1e-4, sigma_r=0.0, sigma_q=0.0)

    corr_high = _adjacent_noise_corr(x, rho_t=0.9, **quiet)
    corr_zero = _adjacent_noise_corr(x, rho_t=0.0, **quiet)

    assert corr_high > 0.7, (
        f"AR(1) rho=0.9 gave adjacent noise-field correlation={corr_high:.3f}"
    )
    assert corr_high > corr_zero + 0.3, (
        f"rho=0.9 corr={corr_high:.3f} not clearly above rho=0 corr={corr_zero:.3f}"
    )


def test_video_rho_t_zero_uncorrelated():
    torch.manual_seed(3)
    x = torch.rand(1, 4, 3, 32, 32)
    y, _ = hdr_mef_noise_mismatch_v1(x, severity=0.8, seed=5, rho_t=0.0)
    assert torch.isfinite(y).all()


# Numerical stability
def test_no_nan_inf_extremes():
    for v in (0.0, 1.0):
        x = torch.full((1, 3, 32, 32), v)
        y, _ = hdr_mef_noise_mismatch_v1(x, severity=1.0, seed=0)
        assert torch.isfinite(y).all()
        assert not torch.allclose(y, torch.full_like(y, 0.5), atol=1e-6), (
            "output collapsed to fallback midpoint"
        )
        assert not torch.allclose(y, x, atol=1e-3), (
            "output identical to input at severity=1.0"
        )


def test_fallback_uses_midpoint_not_zero(monkeypatch):
    import importlib
    mod = importlib.import_module("distortion_library.hdr_mef_noise_mismatch_v1")

    orig_oetf = mod._srgb_from_linear

    def nan_oetf(L):
        return torch.full_like(orig_oetf(L), float("nan"))

    monkeypatch.setattr(mod, "_srgb_from_linear", nan_oetf)

    x = torch.zeros(1, 3, 32, 32)
    y, _ = mod.hdr_mef_noise_mismatch_v1(
        x, severity=1.0, seed=0, value_range=(0.0, 1.0),
    )
    assert torch.isfinite(y).all()
    assert torch.allclose(y, torch.full_like(y, 0.5), atol=1e-6), \
        f"fallback appears to be zero, not midpoint: sample={y.flatten()[:4]}"


def test_kernel_cache_bounded():
    for i in range(300):
        _get_gauss_kernel(0.1 + i * 0.05, torch.float32, torch.device("cpu"))
    assert len(_KERNEL_CACHE) <= _KERNEL_CACHE.max_size


def test_small_image_does_not_crash():
    for sz in (4, 8, 16):
        x = torch.rand(1, 3, sz, sz)
        y, _ = hdr_mef_noise_mismatch_v1(x, severity=0.8, seed=0, sigma_s=3.0)
        assert torch.isfinite(y).all()
        assert y.shape == x.shape


def test_sigma_s_warns_when_huge():
    x = torch.rand(1, 3, 32, 32)
    with pytest.warns(RuntimeWarning, match="conv2d cost grows"):
        hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0, sigma_s=20.0)


def test_sigma_s_no_warn_when_small():
    x = torch.rand(1, 3, 64, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", UserWarning)
        hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0, sigma_s=1.0)


# nn.Module wrapper
def test_module_wrapper(img4d):
    m = HdrMefNoiseMismatchV1(severity=0.5, seed=0)
    y, label = m(img4d)
    assert y.shape == img4d.shape
    # 4D input with B=2 → label (B, 2) = (2, 2)
    assert label.shape == (2, 2)


def test_module_wrapper_kwargs_forward():
    x = torch.rand(1, 3, 32, 32)
    m = HdrMefNoiseMismatchV1(severity=0.5, seed=0, K=5, sigma_s=2.0, rho_t=0.3)
    y_mod, _ = m(x)
    y_fn, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=0,
                                        K=5, sigma_s=2.0, rho_t=0.3)
    assert torch.allclose(y_mod, y_fn, atol=0, rtol=0)


def test_module_wrapper_compute_dtype():
    x = torch.rand(1, 3, 32, 32)
    m = HdrMefNoiseMismatchV1(severity=0.5, seed=0, compute_dtype=torch.float64)
    y_mod, _ = m(x)
    y_fn, _ = hdr_mef_noise_mismatch_v1(
        x, severity=0.5, seed=0, compute_dtype=torch.float64
    )
    assert torch.allclose(y_mod, y_fn, atol=0, rtol=0)


def test_module_wrapper_registered():
    assert DISTORTION_REGISTRY["hdr_mef_noise_mismatch_v1"] is hdr_mef_noise_mismatch_v1


# bf16 batch-position determinism
def test_batch_position_determinism_bf16():
    if not hasattr(torch, "bfloat16"):
        pytest.skip("bfloat16 unavailable")
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32).to(torch.bfloat16)
    y_full, _ = hdr_mef_noise_mismatch_v1(x, severity=0.5, seed=77)
    y0, _ = hdr_mef_noise_mismatch_v1(x[0:1], severity=0.5, seed=77)
    diff = (y_full[0].float() - y0[0].float()).abs().max().item()
    assert diff < 2e-2, f"bf16 batch-position diff={diff}"


# sRGB naming
def test_srgb_function_names():
    from distortion_library.hdr_mef_noise_mismatch_v1 import (
        _srgb_to_linear, _srgb_from_linear,
    )
    u = torch.linspace(0.0, 1.0, 64)
    u2 = _srgb_from_linear(_srgb_to_linear(u))
    assert torch.allclose(u, u2, atol=1e-6)


def test_srgb_legacy_names_absent():
    import importlib
    mod = importlib.import_module("distortion_library.hdr_mef_noise_mismatch_v1")
    assert not hasattr(mod, "_srgb_eotf_inv")
    assert not hasattr(mod, "_srgb_oetf")
    assert hasattr(mod, "_srgb_to_linear")
    assert hasattr(mod, "_srgb_from_linear")
