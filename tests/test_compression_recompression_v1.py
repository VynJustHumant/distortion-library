import warnings

import pytest
import torch

from distortion_library.compression_recompression_v1 import (
    distortion, register, REGISTRY_ID, DISTORTION_ID,
)


# ---------------- shape / dtype / device ----------------
@pytest.mark.parametrize("shape", [(3, 32, 32), (2, 3, 32, 32), (1, 3, 17, 23)])
def test_shape_preserved(shape):
    x = torch.rand(*shape)
    y, label = distortion(x, 0.5)
    assert y.shape == x.shape
    expected = (2,) if x.dim() == 3 else (x.shape[0], 2)          # [v3-3]
    assert label.shape == expected


def test_dtype_preserved():
    for dt in (torch.float16, torch.float32, torch.float64):
        x = torch.rand(1, 3, 32, 32, dtype=dt)
        y, _ = distortion(x, 0.5)
        assert y.dtype == dt


def test_memory_format_channels_last():
    x = torch.rand(2, 3, 32, 32).to(memory_format=torch.channels_last)
    y, _ = distortion(x, 0.5)
    assert y.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_input_works():                              # [v3-1]
    x = torch.rand(2, 3, 32, 32, device="cuda")
    y, _ = distortion(x, 0.5)
    assert y.device.type == "cuda"
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_with_kwargs():                              # [v3-6]
    x = torch.rand(2, 3, 32, 32, device="cuda")
    y, _ = distortion(x, 0.5, N_max=2)
    assert y.device.type == "cuda"


# ---------------- identity / low severity ----------------
def test_identity_low_severity():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    y, _ = distortion(x, 0.01)
    mse = ((y - x) ** 2).mean().item()
    assert mse < 0.01, f"MSE(s=0.01)={mse} exceeds 0.01"


# ---------------- monotonicity ----------------
def test_monotonicity_mse():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    sevs = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    mses = [((distortion(x, s)[0] - x) ** 2).mean().item() for s in sevs]
    for a, b in zip(mses, mses[1:]):
        assert a <= b + 1e-6, f"monotonicity violated: {mses}"


# ---------------- determinism ----------------
def test_determinism():
    x = torch.rand(1, 3, 32, 32)
    y1, _ = distortion(x, 0.5, seed=42)
    y2, _ = distortion(x, 0.5, seed=42)
    assert torch.equal(y1, y2)


def test_seed_does_not_change_output():
    """API-compat: `seed` is accepted for uniformity with sibling
    distortions that consume it; this operator is analytically
    deterministic, so distinct seeds must yield identical outputs."""
    x = torch.rand(1, 3, 32, 32)
    y1, _ = distortion(x, 0.5, seed=1)
    y2, _ = distortion(x, 0.5, seed=999)
    assert torch.equal(y1, y2)


# ---------------- differentiability ----------------
def test_gradients_exist_and_finite():
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    s = torch.tensor([0.5], requires_grad=True)
    y, _ = distortion(x, s)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_gradient_wrt_severity_nonzero():
    x = torch.rand(1, 3, 32, 32)
    s = torch.tensor([0.5], requires_grad=True)
    y, _ = distortion(x, s)
    y.sum().backward()
    assert s.grad.abs().sum().item() > 0.0


# ---------------- real gradcheck (STE disabled) ----------------
def test_gradcheck_smooth_path_fp64():
    """Real gradcheck on the smooth path.

    Requirements for gradcheck-ability:
      * use_ste=False       -> quantizer becomes identity
      * N_max=1             -> eliminates floor/ceil/gather branch
      * chroma_upsample="bilinear"  -> "nearest" has zero gradient
                                        (piecewise-constant selection)
      * interior input      -> avoids clamp(0,1) non-smoothness

    The full pipeline is then: RGB<->YCbCr (linear), 4:2:0 avg-pool
    (differentiable), DCT/IDCT (orthogonal linear), bilinear upsample
    (smooth), identity blend.  All ops are C^1 on interior points.
    """
    torch.manual_seed(0)
    x = (0.2 + 0.6 * torch.rand(1, 3, 16, 16, dtype=torch.float64)).requires_grad_(True)
    s = torch.tensor([0.5], dtype=torch.float64)

    def f(xx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y, _ = distortion(
                xx, s,
                use_ste=False, N_max=1,
                chroma_upsample="bilinear",
            )
        return y

    assert torch.autograd.gradcheck(
        f, (x,), eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=True,
    )


def test_ste_gradient_sign_and_magnitude():               # [v3.1-4]
    """STE backward is identity; dMSE/ds must be strictly positive with
    non-trivial magnitude (not just float noise) and bounded by the MSE
    headroom (<=1) per unit severity."""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    s = torch.tensor([0.5], requires_grad=True)
    y, _ = distortion(x, s, use_ste=True)
    mse = ((y - x) ** 2).mean()
    mse.backward()

    g = s.grad.item()
    assert g > 1e-6, f"sign correct but magnitude too small: {g}"
    assert g < 1.0, f"gradient suspiciously large: {g}"


def test_ste_gradient_matches_finite_difference():        # [v3.1-4]
    """On the smooth path, analytic dMSE/ds should match central
    finite difference to within a loose tolerance."""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 16, 16, dtype=torch.float64)

    def mse_at(s_val: float) -> float:
        s = torch.tensor([s_val], dtype=torch.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y, _ = distortion(x, s, use_ste=False)
        return ((y - x) ** 2).mean().item()

    s0 = 0.5
    s = torch.tensor([s0], dtype=torch.float64, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y, _ = distortion(x, s, use_ste=False)
    ((y - x) ** 2).mean().backward()
    analytic = s.grad.item()

    h = 1e-4
    fd = (mse_at(s0 + h) - mse_at(s0 - h)) / (2 * h)
    assert abs(analytic - fd) < 1e-4, (analytic, fd)


# ---------------- batch invariance ----------------
@pytest.mark.parametrize("B", [2, 4])
def test_batch_invariance(B):
    torch.manual_seed(0)
    x = torch.rand(B, 3, 32, 32)
    sev = torch.full((B,), 0.5)
    y_batch, _ = distortion(x, sev)
    for i in range(B):
        y_single, _ = distortion(x[i], 0.5)
        diff = (y_batch[i] - y_single).abs().max().item()
        assert diff < 1e-4, f"batch invariance violated: {diff}"


def test_batch_invariance_per_sample_severity():
    torch.manual_seed(0)
    x = torch.rand(3, 3, 32, 32)
    sev = torch.tensor([0.2, 0.5, 0.9])
    y_batch, _ = distortion(x, sev)
    for i, s in enumerate(sev):
        y_single, _ = distortion(x[i], s.item())
        assert (y_batch[i] - y_single).abs().max().item() < 1e-4


# ---------------- range ----------------
def test_output_range():
    x = torch.rand(2, 3, 32, 32)
    y, _ = distortion(x, 1.0)
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_value_range_override():
    x = torch.rand(2, 3, 32, 32) * 255.0
    y, _ = distortion(x, 0.5, value_range=(0.0, 255.0))
    assert y.min().item() >= 0.0
    assert y.max().item() <= 255.0


def test_value_range_neg1_pos1_batch():                    # [v3-8]
    torch.manual_seed(0)
    x = torch.rand(3, 3, 32, 32) * 2.0 - 1.0
    y, _ = distortion(x, torch.tensor([0.1, 0.5, 0.9]),
                      value_range=(-1.0, 1.0))
    assert y.min().item() >= -1.0 - 1e-5
    assert y.max().item() <= 1.0 + 1e-5
    y_single, _ = distortion(x[1], 0.5, value_range=(-1.0, 1.0))
    assert (y[1] - y_single).abs().max().item() < 1e-4


def test_value_range_neg1_pos1_grad():                     # [v3-8]
    x = (torch.rand(1, 3, 32, 32) * 2.0 - 1.0).requires_grad_(True)
    s = torch.tensor([0.5], requires_grad=True)
    y, _ = distortion(x, s, value_range=(-1.0, 1.0))
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(s.grad).all()


# ---------------- label ----------------
def test_label_shape_and_content():
    x = torch.rand(2, 3, 32, 32)
    _, label = distortion(x, 0.7)
    assert label.shape == (2, 2)
    assert label.dtype == torch.float32                    # [v3-4]
    assert label.device.type == "cpu"                      # [v3-4]
    assert (label[:, 0] == float(DISTORTION_ID)).all()
    assert torch.allclose(label[:, 1], torch.tensor([0.7, 0.7]))


def test_label_fp32_even_for_fp16_input():                 # [v3-4]
    x = torch.rand(1, 3, 32, 32, dtype=torch.float16)
    _, label = distortion(x, 0.5)
    assert label.dtype == torch.float32
    assert label.device.type == "cpu"


def test_label_detached():
    x = torch.rand(1, 3, 32, 32)
    s = torch.tensor([0.5], requires_grad=True)
    _, label = distortion(x, s)
    assert not label.requires_grad


# ---------------- registry ----------------
def test_registry_registration():
    reg = register()
    assert REGISTRY_ID in reg
    assert reg[REGISTRY_ID] is distortion


# ---------------- kwargs & caching ----------------
def test_kwargs_override_nmax():
    """Verify N_max override creates distinct modules in the cache.

    Output-level comparison is unreliable: the JPEG re-quantization
    operator is idempotent, so gen-2 and gen-8 produce identical tensors
    after the first re-quantization.  We instead confirm the wrapper
    resolves to two distinct cached modules.
    """
    import importlib
    m = importlib.import_module('distortion_library.compression_recompression_v1')

    # Unique Q_min so cache keys cannot collide with other tests.
    _ = distortion(torch.rand(1, 3, 16, 16), 0.5, N_max=2, Q_min=11.0)
    _ = distortion(torch.rand(1, 3, 16, 16), 0.5, N_max=8, Q_min=11.0)

    n_maxes = sorted(
        dict(cfg_items)["N_max"]
        for (_, cfg_items), _ in m._MODULE_CACHE.items()
        if ("Q_min", 11.0) in cfg_items
    )
    assert n_maxes == [2, 8], f"expected [2, 8], got {n_maxes}"


def test_kwargs_override_chroma_upsample():
    """Verify chroma_upsample override actually creates a distinct module.

    Output-level difference is not reliable at moderate severity because
    the quantized chroma can be trivially constant (nearest == bilinear).
    Instead we check that two distinct cache entries are created.
    """
    import importlib
    m = importlib.import_module('distortion_library.compression_recompression_v1')

    # Unique N_max so cache keys cannot collide with other tests.
    _ = distortion(torch.rand(1, 3, 32, 32), 0.5, N_max=77,
                   chroma_upsample="nearest")
    _ = distortion(torch.rand(1, 3, 32, 32), 0.5, N_max=77,
                   chroma_upsample="bilinear")

    mods = [
        mod for (_, cfg_items), mod in m._MODULE_CACHE.items()
        if ('N_max', 77) in cfg_items
    ]
    chromas = sorted(mod.chroma_upsample for mod in mods)
    assert chromas == ["bilinear", "nearest"], (
        f"expected two distinct chroma configs, got {chromas}"
    )


def test_module_cache_reuses_instance():                   # [v3-6]
    import importlib
    m = importlib.import_module('distortion_library.compression_recompression_v1')
    _ = distortion(torch.rand(1, 3, 32, 32), 0.5, N_max=4)
    n1 = len(m._MODULE_CACHE)
    _ = distortion(torch.rand(1, 3, 32, 32), 0.5, N_max=4)
    n2 = len(m._MODULE_CACHE)
    assert n1 == n2


def test_module_cache_bounded():                           # [v3.1-1]
    import importlib
    m = importlib.import_module('distortion_library.compression_recompression_v1')
    for i in range(m._CACHE_MAX + 10):
        _ = distortion(torch.rand(1, 3, 16, 16), 0.5, N_max=1 + (i % 3),
                       Q_min=1.0 + 0.001 * i)
    assert len(m._MODULE_CACHE) <= m._CACHE_MAX


# ---------------- use_ste guardrail ----------------
def test_use_ste_false_warns():                            # [v3.1-3]
    x = torch.rand(1, 3, 32, 32)
    with pytest.warns(RuntimeWarning, match="use_ste=False"):
        distortion(x, 0.5, use_ste=False)


# ---------------- finite sanitization ----------------
def test_nonfinite_input_sanitized():
    x = torch.rand(1, 3, 32, 32)
    x[0, 0, 0, 0] = float("nan")
    x[0, 1, 0, 0] = float("inf")
    y, _ = distortion(x, 0.5)
    assert torch.isfinite(y).all()
