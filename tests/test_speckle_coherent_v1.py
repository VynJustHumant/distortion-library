"""Test suite for speckle_coherent_v1 (v1.0.8)."""
from __future__ import annotations

import inspect
import math
import re
import warnings

import pytest
import torch

from distortion_library import DISTORTION_REGISTRY
from distortion_library.speckle_coherent_v1 import (
    distortion,
    SpeckleCoherentV1,
    REGISTRY_ID_INT,
    REGISTRY_ID,
    VERSION,
    _severity_to_L,
    _sigma_from_L,
    _S_MIN,
)


@pytest.fixture
def img3():
    torch.manual_seed(0)
    return torch.rand(3, 64, 64)


@pytest.fixture
def img4():
    torch.manual_seed(0)
    return torch.rand(4, 3, 64, 64)


# Registry / package exports

def test_registry_contains():
    assert REGISTRY_ID in DISTORTION_REGISTRY
    assert DISTORTION_REGISTRY[REGISTRY_ID] is SpeckleCoherentV1


def test_registry_shares_atmospheric_dict():
    from distortion_library.blur_atmospheric_turbulence_v1 import (
        DISTORTION_REGISTRY as ATMO_REG,
    )
    assert DISTORTION_REGISTRY is ATMO_REG
    assert REGISTRY_ID in ATMO_REG


def test_registry_id_is_1008():
    assert REGISTRY_ID_INT == 1008
    assert SpeckleCoherentV1.DISTORTION_ID == 1008
    assert SpeckleCoherentV1.DISTORTION_NAME == REGISTRY_ID
    assert SpeckleCoherentV1.DISTORTION_VERSION == VERSION


def test_package_exports_callables():
    import distortion_library as dl
    for name in ["blur_atmospheric_turbulence_v1",
                 "blur_frc_optical_flow_v1",
                 "compression_recompression_v1",
                 "mgtc_cascade_v1",
                 "speckle_coherent_v1"]:
        obj = getattr(dl, name)
        assert callable(obj), f"{name} is not callable: {type(obj)}"


def test_package_all_complete():
    import distortion_library as dl
    expected = {
        "DISTORTION_REGISTRY",
        "blur_atmospheric_turbulence_v1",
        "blur_frc_optical_flow_v1",
        "compression_recompression_v1",
        "mgtc_cascade_v1",
        "speckle_coherent_v1",
    }
    assert set(dl.__all__) == expected


def test_package_version_is_semver():
    import distortion_library as dl
    assert re.match(r"^\d+\.\d+\.\d+$", dl.__version__), dl.__version__


def test_registry_class_attributes_set_by_decorator():
    assert SpeckleCoherentV1.DISTORTION_NAME == REGISTRY_ID
    assert SpeckleCoherentV1.DISTORTION_VERSION == VERSION
    assert "VERSION" not in vars(SpeckleCoherentV1)


def test_class_body_does_not_redefine_decorator_attributes():
    src = inspect.getsource(SpeckleCoherentV1)
    non_comment_lines = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    for attr in ("DISTORTION_NAME", "DISTORTION_VERSION"):
        pattern = rf"^\s*{attr}\s*[:=]"
        assert not re.search(pattern, non_comment_lines, re.MULTILINE), \
            f"{attr} should be set only by the decorator"


# Shape / dtype / device / memory format

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_shape_dtype_preserved(img3, dtype):
    x = img3.to(dtype)
    out, label = distortion(x, 0.5, seed=0)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert label.shape == (2,)
    assert label.dtype == torch.float32
    assert label.device.type == "cpu"


def test_batched_shape(img4):
    out, label = distortion(img4, torch.full((4,), 0.5), seed=0)
    assert out.shape == img4.shape
    assert label.shape == (4, 2)


def test_memory_format_channels_last(img4):
    x = img4.contiguous(memory_format=torch.channels_last)
    out, _ = distortion(x, 0.5, seed=0)
    assert out.is_contiguous(memory_format=torch.channels_last)


# compute_dtype overrides

def test_compute_dtype_override_on_fp64_keeps_output_dtype():
    x = torch.rand(1, 1, 32, 32, dtype=torch.float64)
    out, _ = distortion(x, 0.5, seed=0, compute_dtype=torch.float32)
    assert out.dtype == torch.float64


def test_compute_dtype_override_promotes_fp32():
    x = torch.rand(1, 1, 32, 32, dtype=torch.float32)
    out, _ = distortion(x, 0.5, seed=0, compute_dtype=torch.float64)
    assert out.dtype == torch.float32


def test_compute_dtype_override_actually_lowers_precision():
    x = torch.rand(1, 1, 64, 64, dtype=torch.float64)
    a, _ = distortion(x, 0.5, seed=0, compute_dtype=torch.float64)
    b, _ = distortion(x, 0.5, seed=0, compute_dtype=torch.float32)
    assert a.dtype == torch.float64 and b.dtype == torch.float64
    assert not torch.equal(a, b), "compute_dtype had no effect"


# Label

def test_label_id_is_1008(img4):
    _, label = distortion(img4, torch.tensor([0.1, 0.3, 0.6, 1.0]), seed=0)
    assert torch.all(label[:, 0] == 1008.0)


def test_label_detached(img4):
    x = img4.clone().requires_grad_(True)
    _, label = distortion(x, 0.5, seed=0)
    assert not label.requires_grad


# Range / identity / monotonicity

def test_output_in_range(img4):
    out, _ = distortion(img4, 1.0, seed=0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.isfinite(out).all()


def test_near_identity_low_severity(img4):
    out, _ = distortion(img4, 0.01, seed=0)
    assert (out - img4).pow(2).mean().item() < 0.01


def test_monotonic_mse_lognormal(img4):
    sevs = [0.01, 0.1, 0.3, 0.6, 1.0]
    mses = [(distortion(img4, s, seed=42)[0] - img4).pow(2).mean().item()
            for s in sevs]
    for a, b in zip(mses, mses[1:]):
        assert a <= b + 1e-6


def test_monotonic_mse_correlated(img4):
    sevs = [0.01, 0.1, 0.3, 0.6, 1.0]
    mses = [(distortion(img4, s, seed=42, rho=1.5)[0] - img4).pow(2).mean().item()
            for s in sevs]
    for a, b in zip(mses, mses[1:]):
        assert a <= b + 1e-6, mses


def test_mean_preserved_lognormal():
    torch.manual_seed(1234)
    x = 0.2 + 0.6 * torch.rand(4, 3, 32, 32)
    N = 512
    acc = torch.zeros_like(x)
    for i in range(N):
        acc += distortion(x, 0.5, seed=i)[0]
    err = (acc / N - x).abs().mean().item()
    assert err < 0.002, err


# Determinism / batch invariance

def test_seed_determinism(img4):
    a, _ = distortion(img4, 0.5, seed=7)
    b, _ = distortion(img4, 0.5, seed=7)
    assert torch.equal(a, b)


def test_generator_determinism(img4):
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    a, _ = distortion(img4, 0.5, generator=g1)
    b, _ = distortion(img4, 0.5, generator=g2)
    assert torch.equal(a, b)


def test_no_seed_uses_global_rng():
    x = torch.rand(1, 1, 32, 32)
    torch.manual_seed(42)
    a, _ = distortion(x, 0.5)
    torch.manual_seed(42)
    b, _ = distortion(x, 0.5)
    assert torch.equal(a, b)


def test_no_seed_differs_across_global_states():
    x = torch.rand(1, 1, 32, 32)
    torch.manual_seed(0)
    a, _ = distortion(x, 0.5)
    torch.manual_seed(1)
    b, _ = distortion(x, 0.5)
    assert not torch.equal(a, b)


def test_no_seed_fallback_shape_dtype():
    x = torch.rand(2, 3, 32, 32, dtype=torch.float32)
    out, label = distortion(x, 0.5)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert label.shape == (2, 2)


@pytest.mark.parametrize("kwargs", [
    dict(),
    dict(rho=1.5),
    dict(channel_independent=True),
    dict(exact_gamma=True),
])
def test_reproducibility_same_batch(img4, kwargs):
    """Same batch + same seed ⇒ bit-identical output (per-path)."""
    a, _ = distortion(img4, 0.5, seed=3, **kwargs)
    b, _ = distortion(img4, 0.5, seed=3, **kwargs)
    assert torch.equal(a, b)


@pytest.mark.parametrize("kwargs", [
    dict(),
    dict(rho=1.5),
    dict(channel_independent=True),
    dict(exact_gamma=True),
])
def test_batch_size_shifts_noise_stream(img4, kwargs):
    """
    Documented behaviour: per-sample RNG is keyed by (seed, batch_index).
    A sample extracted into a smaller batch sees a different stream.
    """
    full, _ = distortion(img4, 0.5, seed=3, **kwargs)
    single, _ = distortion(img4[2:3], 0.5, seed=3, **kwargs)
    assert not torch.equal(full[2:3], single)


def test_cross_sample_independence(img4):
    out, _ = distortion(img4, 0.5, seed=1)
    for i in range(4):
        for j in range(i + 1, 4):
            if not torch.equal(out[i], out[j]):
                return
    pytest.fail("all samples identical")


# Gradients

def test_grad_flows_to_image(img4):
    x = img4.clone().requires_grad_(True)
    out, _ = distortion(x, 0.5, seed=0)
    out.mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_grad_flows_to_severity(img4):
    s = torch.tensor(0.5, requires_grad=True)
    out, _ = distortion(img4, s, seed=0)
    out.mean().backward()
    assert s.grad is not None and s.grad.abs() > 0


def test_gradcheck_lognormal_fp64():
    x = torch.rand(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)
    s = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: distortion(a, b, seed=0, compute_dtype=torch.float64)[0],
        (x, s), eps=1e-6, atol=1e-4, rtol=1e-3,
    )


def test_gradcheck_correlated_fp64():
    x = torch.rand(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    s = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: distortion(
            a, b, seed=0, rho=1.0, compute_dtype=torch.float64
        )[0],
        (x, s), eps=1e-6, atol=1e-4, rtol=1e-3,
    )


def test_exact_gamma_blend_gradient():
    x = torch.rand(1, 1, 32, 32, requires_grad=True)
    s = torch.tensor(0.5, requires_grad=True)
    out, _ = distortion(x, s, seed=0, exact_gamma=True)
    out.mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert s.grad is not None and s.grad.abs() > 0


# exact_gamma path

def test_exact_gamma_runs():
    x = torch.rand(4, 1, 32, 32)
    out, _ = distortion(x, 0.5, seed=0, exact_gamma=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


def test_exact_gamma_mean_preserved():
    x = torch.full((1, 1, 64, 64), 0.3)
    N = 64
    acc = torch.zeros_like(x)
    for i in range(N):
        acc += distortion(x, 1.0, seed=i, exact_gamma=True)[0]
    mean = (acc / N).mean().item()
    assert abs(mean - 0.3) < 0.05, mean


# Correlated speckle

def test_correlated_speckle_mean_is_one():
    x = torch.full((1, 1, 64, 64), 0.3)
    N = 64
    acc = torch.zeros_like(x)
    for i in range(N):
        acc += distortion(x, 1.0, seed=i, rho=2.0)[0]
    mean = (acc / N).mean().item()
    assert abs(mean - 0.3) < 0.05, mean


def test_correlated_speckle_matches_lognormal_mean():
    x = torch.full((1, 1, 64, 64), 0.3)
    N = 64
    acc_iid = torch.zeros_like(x)
    acc_cor = torch.zeros_like(x)
    for i in range(N):
        acc_iid += distortion(x, 1.0, seed=i, rho=None)[0]
        acc_cor += distortion(x, 1.0, seed=i, rho=2.0)[0]
    m_iid = (acc_iid / N).mean().item()
    m_cor = (acc_cor / N).mean().item()
    assert abs(m_iid - 0.3) < 0.05, m_iid
    assert abs(m_cor - 0.3) < 0.05, m_cor


def test_correlated_scale_matches_one_at_s1_via_min():
    y_min = float("inf")
    for seed in range(8):
        o, _ = distortion(
            torch.full((4, 1, 128, 128), 0.5), 1.0, seed=seed, rho=2.0
        )
        y_min = min(y_min, o.min().item())
    assert y_min < 0.03, (
        f"y_min={y_min:.5f}; correct≈1e-4 (scale=1.0), buggy≈0.083 "
        f"(scale=sqrt(log2)≈0.833)"
    )


def test_correlated_variance_scales_with_severity():
    torch.manual_seed(0)
    # x=0.15: y = 0.15·m; clamping (y=1 ⇔ m=6.67) is far in the tail for
    # both Exp(1) and lognormal(σ²=log 2), so we measure the true Var[y].
    x_val = 0.15
    x = torch.full((1, 1, 128, 128), x_val)
    N = 32
    s_low, s_high = 0.05, 1.0
    L_max, gamma = 1000.0, 2.0

    def theoretical_var_y(s_val: float) -> float:
        s_t = torch.tensor([s_val], dtype=torch.float64)
        L_t = _severity_to_L(s_t, L_max=L_max, gamma=gamma).item()
        return (s_val ** 2) * (x_val ** 2) / L_t

    expected_ratio = theoretical_var_y(s_high) / theoretical_var_y(s_low)

    def emp_var(sev: float) -> float:
        acc = torch.zeros_like(x)
        acc_sq = torch.zeros_like(x)
        for i in range(N):
            o = distortion(x, sev, seed=i, rho=2.0)[0]
            acc += o
            acc_sq += o * o
        mean = acc / N
        return (acc_sq / N - mean * mean).mean().item()

    var_low = emp_var(s_low)
    var_high = emp_var(s_high)
    observed_ratio = var_high / max(var_low, 1e-12)
    assert observed_ratio > 0.5 * expected_ratio, (
        f"observed={observed_ratio:.3e}, expected≈{expected_ratio:.3e}"
    )


def test_correlated_variance_matches_lognormal_at_high_severity():
    torch.manual_seed(42)
    # x=0.15 avoids output clamping (y = x·m ≤ 1 requires m ≤ 6.67).
    # Expected Var[y] = x² · Var[m] = 0.15² · 1 = 0.0225.
    x = torch.full((1, 1, 128, 128), 0.15)
    N = 64

    def emp_var(kwargs) -> float:
        acc = torch.zeros_like(x)
        acc_sq = torch.zeros_like(x)
        for i in range(N):
            o = distortion(x, 1.0, seed=i, **kwargs)[0]
            acc += o
            acc_sq += o * o
        mean = acc / N
        return (acc_sq / N - mean * mean).mean().item()

    var_iid = emp_var(dict())
    var_cor = emp_var(dict(rho=2.0))
    expected = 0.15 ** 2   # = x² · Var[m] with Var[m]=1 at s=1
    assert abs(var_iid - expected) / expected < 0.25, var_iid
    assert abs(var_cor - expected) / expected < 0.25, var_cor
    assert abs(var_iid - var_cor) / max(var_iid, var_cor) < 0.25, \
        (var_iid, var_cor)


def test_correlated_variance_bounded_at_s1():
    torch.manual_seed(0)
    x = torch.full((1, 1, 128, 128), 0.5)
    N = 32
    acc = torch.zeros_like(x)
    acc_sq = torch.zeros_like(x)
    for i in range(N):
        o = distortion(x, 1.0, seed=i, rho=2.0)[0]
        acc += o
        acc_sq += o * o
    mean = acc / N
    var = (acc_sq / N - mean * mean).mean().item()
    assert 0.0 <= var <= 0.25, var


# rho upper-bound warning

def test_large_rho_emits_warning():
    x = torch.rand(1, 1, 32, 32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distortion(x, 0.5, seed=0, rho=10.0)
    msgs = [str(w.message) for w in caught]
    assert any("kernel" in m and "rho" in m for m in msgs), msgs


def test_small_rho_no_warning():
    x = torch.rand(1, 1, 64, 64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distortion(x, 0.5, seed=0, rho=1.0)
    msgs = [str(w.message) for w in caught]
    assert not any("kernel" in m and "rho=" in m for m in msgs), msgs


# Severity mapping hyperparameters

def test_L_max_1_collapses_L_to_1():
    for s_val in [0.01, 0.3, 0.5, 0.9, 1.0]:
        s = torch.tensor([s_val], dtype=torch.float64)
        L = _severity_to_L(s, L_max=1.0).item()
        assert abs(L - 1.0) < 1e-12, f"L({s_val}) = {L}, expected 1.0"


def test_L_max_1_variance_matches_at_all_severity():
    torch.manual_seed(0)
    x = torch.full((1, 1, 128, 128), 0.5)
    N = 64
    s_val = 0.5

    def emp_mse(L_max: float) -> float:
        acc = 0.0
        for i in range(N):
            o = distortion(x, s_val, seed=i, L_max=L_max)[0]
            acc += (o - x).pow(2).mean().item()
        return acc / N

    mse_strong = emp_mse(1.0)
    mse_mild = emp_mse(1000.0)
    assert mse_strong > 100 * mse_mild, (mse_strong, mse_mild)


def test_gamma_1_still_monotonic():
    x = torch.rand(1, 1, 32, 32)
    mses = [(distortion(x, s, seed=0, gamma=1.0)[0] - x).pow(2).mean().item()
            for s in (0.1, 0.3, 0.6, 1.0)]
    for a, b in zip(mses, mses[1:]):
        assert a <= b + 1e-6, mses


def test_gamma_effect_on_severity_mapping():
    x = torch.rand(1, 1, 64, 64)
    a, _ = distortion(x, 0.5, seed=0, gamma=0.5)
    b, _ = distortion(x, 0.5, seed=0, gamma=2.0)
    assert not torch.equal(a, b)


# channel_independent

def test_channel_independent_shape():
    x = torch.rand(2, 3, 32, 32)
    out, label = distortion(x, 0.5, seed=0, channel_independent=True)
    assert out.shape == x.shape
    assert label.shape == (2, 2)


def test_channel_independent_decorrelates_channels():
    x = torch.ones(1, 3, 64, 64)
    shared, _ = distortion(x, 0.5, seed=0, channel_independent=False)
    indep,  _ = distortion(x, 0.5, seed=0, channel_independent=True)
    assert torch.allclose(shared[0, 0], shared[0, 1], atol=1e-6)
    assert torch.allclose(shared[0, 0], shared[0, 2], atol=1e-6)
    assert not torch.allclose(indep[0, 0], indep[0, 1], atol=1e-3)
    assert not torch.allclose(indep[0, 1], indep[0, 2], atol=1e-3)


def test_channel_independent_determinism():
    x = torch.rand(2, 3, 32, 32)
    a, _ = distortion(x, 0.5, seed=9, channel_independent=True)
    b, _ = distortion(x, 0.5, seed=9, channel_independent=True)
    assert torch.equal(a, b)


# value_range

def test_value_range_uint8_scale():
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    out, _ = distortion(x, 0.5, seed=0, value_range=(0.0, 255.0))
    assert out.dtype == torch.uint8
    assert out.min() >= 0 and out.max() <= 255
    mse = ((out.float() - x.float()) / 255.0).pow(2).mean().item()
    assert mse < 0.01, mse


def test_value_range_asymmetric():
    x = torch.rand(1, 3, 32, 32) * 0.5 - 0.25
    out, _ = distortion(x, 0.5, seed=0, value_range=(-0.25, 0.25))
    assert out.min() >= -0.25 - 1e-5
    assert out.max() <= 0.25 + 1e-5
    assert (out - x).pow(2).mean().item() < 0.05


# Input validation

def test_rho_negative_raises(img3):
    with pytest.raises(ValueError):
        distortion(img3, 0.5, seed=0, rho=-0.5)


def test_rho_and_exact_gamma_raises(img3):
    with pytest.raises(ValueError):
        distortion(img3, 0.5, seed=0, rho=1.0, exact_gamma=True)


def test_rho_zero_canonicalizes_to_iid(img3):
    a, _ = distortion(img3, 0.5, seed=0, rho=0.0)
    b, _ = distortion(img3, 0.5, seed=0, rho=None)
    assert torch.equal(a, b)


def test_tiny_image_with_rho():
    x = torch.rand(1, 1, 4, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out, _ = distortion(x, 0.5, seed=0, rho=2.0)
    assert out.shape == x.shape


def test_unknown_kwarg_rejected(img3):
    with pytest.raises(TypeError):
        distortion(img3, 0.5, seed=0, bogus_kwarg=1)


def test_severity_accepts_list(img4):
    out, label = distortion(img4, [0.1, 0.3, 0.6, 0.9], seed=0)
    assert out.shape == img4.shape
    assert label.shape == (4, 2)


def test_severity_accepts_tuple(img4):
    out, label = distortion(img4, (0.1, 0.3, 0.6, 0.9), seed=0)
    assert out.shape == img4.shape
    assert label.shape == (4, 2)


def test_2d_input_rejected():
    with pytest.raises(ValueError, match=r"\(C,H,W\) or \(B,C,H,W\)"):
        distortion(torch.rand(32, 32), 0.5)


def test_1d_input_rejected():
    with pytest.raises(ValueError, match=r"\(C,H,W\) or \(B,C,H,W\)"):
        distortion(torch.rand(32), 0.5)


def test_severity_batch_mismatch_rejected(img4):
    with pytest.raises(ValueError, match="severity batch 2 != image batch 4"):
        distortion(img4, torch.tensor([0.1, 0.3]))


def test_severity_batch_mismatch_4d_rejected(img4):
    with pytest.raises(ValueError, match="severity batch 2 != image batch 4"):
        distortion(img4, torch.tensor([0.1, 0.3]).view(2, 1, 1, 1))


def test_severity_accepts_4d_shape(img4):
    sev = torch.tensor([0.1, 0.3, 0.6, 0.9]).view(4, 1, 1, 1)
    out, label = distortion(img4, sev, seed=0)
    assert out.shape == img4.shape
    assert label.shape == (4, 2)
    out_flat, _ = distortion(img4, torch.tensor([0.1, 0.3, 0.6, 0.9]), seed=0)
    assert torch.equal(out, out_flat)


# nn.Module wrapper

def test_module_wrapper_basic():
    m = SpeckleCoherentV1(rho=1.5)
    x = torch.rand(2, 3, 32, 32)
    out, label = m(x, 0.5, seed=0)
    assert out.shape == x.shape
    assert label.shape == (2, 2)


def test_module_wrapper_matches_functional():
    m = SpeckleCoherentV1(rho=1.5)
    x = torch.rand(2, 3, 32, 32)
    out1, _ = m(x, 0.5, seed=0)
    out2, _ = distortion(x, 0.5, seed=0, rho=1.5)
    assert torch.equal(out1, out2)


def test_module_wrapper_exact_gamma():
    m = SpeckleCoherentV1(exact_gamma=True)
    x = torch.rand(2, 1, 32, 32)
    out, _ = m(x, 0.5, seed=0)
    assert out.shape == x.shape


def test_module_wrapper_compute_dtype_override():
    m = SpeckleCoherentV1(compute_dtype=torch.float32)
    x = torch.rand(1, 1, 32, 32, dtype=torch.float64)
    a, _ = m(x, 0.5, seed=0)
    b, _ = distortion(x, 0.5, seed=0, compute_dtype=torch.float64)
    assert a.dtype == b.dtype == torch.float64
    assert not torch.equal(a, b)


def test_module_wrapper_value_range():
    m = SpeckleCoherentV1()
    x = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    out, _ = m(x, 0.5, seed=0, value_range=(0.0, 255.0))
    assert out.dtype == torch.uint8
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 255
    mse = ((out.float() - x.float()) / 255.0).pow(2).mean().item()
    assert mse < 0.01, mse


def test_module_rho_zero_canonicalized():
    m = SpeckleCoherentV1(rho=0.0)
    assert m.rho is None
    x = torch.rand(2, 3, 32, 32)
    a, _ = m(x, 0.5, seed=0)
    b, _ = distortion(x, 0.5, seed=0, rho=None)
    assert torch.equal(a, b)


def test_module_rejects_negative_rho():
    with pytest.raises(ValueError):
        SpeckleCoherentV1(rho=-0.5)


def test_module_rejects_rho_and_exact_gamma():
    with pytest.raises(ValueError):
        SpeckleCoherentV1(rho=1.0, exact_gamma=True)


# Warning behaviour

def test_exact_gamma_with_seed_emits_warning():
    x = torch.rand(1, 1, 8, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distortion(x, 0.5, seed=0, exact_gamma=True)
    messages = [str(w.message) for w in caught]
    assert any("torch._standard_gamma" in m for m in messages), messages


def test_exact_gamma_with_generator_emits_warning():
    x = torch.rand(1, 1, 8, 8)
    g = torch.Generator().manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distortion(x, 0.5, generator=g, exact_gamma=True)
    messages = [str(w.message) for w in caught]
    assert any(
        "torch._standard_gamma" in m and "seed=" in m and "generator=" in m
        for m in messages
    ), messages


def test_exact_gamma_warning_stacklevel_points_to_user_code():
    x = torch.rand(1, 1, 8, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distortion(x, 0.5, seed=0, exact_gamma=True)
    assert len(caught) >= 1, "no warning captured"
    w = caught[0]
    assert "test_speckle_coherent_v1.py" in w.filename, \
        f"warning attributed to {w.filename}:{w.lineno}"


def test_exact_gamma_public_path_no_warning():
    x = torch.rand(1, 1, 8, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distortion(x, 0.5, exact_gamma=True)
    messages = [str(w.message) for w in caught]
    assert not any("torch._standard_gamma" in m for m in messages), messages
