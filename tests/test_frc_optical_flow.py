# tests/test_frc_optical_flow.py
"""
Pytest suite for blur_frc_optical_flow_v1.

Run:
    pytest tests/test_frc_optical_flow.py -q
    pytest tests/test_frc_optical_flow.py -q -m "not slow"    # fast CI
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from distortion_library.blur_frc_optical_flow_v1 import (
    blur_frc_optical_flow_v1 as frc,
    DISTORTION_ID,
    DISTORTION_NAME,
    DISTORTION_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def xb():
    return torch.rand(4, 3, 64, 64)


# ---------------------------------------------------------------------------
# Determinism & seeding
# ---------------------------------------------------------------------------
def test_determinism_seeded(xb):
    y1, _ = frc(xb, severity=0.5, seed=42)
    y2, _ = frc(xb, severity=0.5, seed=42)
    assert torch.equal(y1, y2)


def test_determinism_default_seed(xb):
    y1, _ = frc(xb, severity=0.5)
    y2, _ = frc(xb, severity=0.5)
    assert torch.equal(y1, y2)


def test_different_seeds_differ(xb):
    y1, _ = frc(xb, severity=0.5, seed=1)
    y2, _ = frc(xb, severity=0.5, seed=2)
    assert not torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# Batch invariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("severity", [0.3, 0.5, 0.9])
def test_batch_invariance_uniform(xb, severity):
    y_single, _ = frc(xb[0], severity=severity, seed=42)
    y_batch, _ = frc(xb, severity=severity, seed=42)
    assert (y_single - y_batch[0]).abs().max().item() < 1e-4


def test_batch_invariance_mixed_severity(xb):
    sev = torch.tensor([0.15, 0.35, 0.75, 0.95])
    y_single, _ = frc(xb[0], severity=sev[0], seed=42)
    y_batch, _ = frc(xb, severity=sev, seed=42)
    assert (y_single - y_batch[0]).abs().max().item() < 1e-4
    y3_single, _ = frc(xb[3], severity=sev[3], seed=42)
    assert (y3_single - y_batch[3]).abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# Range / identity / monotonicity
# ---------------------------------------------------------------------------
def test_output_range(xb):
    for s in [0.01, 0.3, 0.6, 1.0]:
        y, _ = frc(xb, severity=s, seed=7)
        assert y.min().item() >= 0.0 - 1e-6
        assert y.max().item() <= 1.0 + 1e-6


def test_identity_preservation(xb):
    y, _ = frc(xb, severity=0.01, seed=1)
    assert F.mse_loss(y, xb).item() < 0.01


def test_monotonicity(xb):
    prev = -1.0
    for s in [0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0]:
        y, _ = frc(xb, severity=s, seed=99)
        d = (y - xb).abs().mean().item()
        assert d >= prev - 1e-4, f"non-monotonic at s={s}: {prev} -> {d}"
        prev = d


# ---------------------------------------------------------------------------
# Shape / dtype / device
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [
    (3, 32, 32),           # single
    (2, 3, 32, 32),        # batch
    (2, 4, 3, 32, 32),     # video
])
def test_shape_preserved(shape):
    x = torch.rand(*shape)
    y, _ = frc(x, severity=0.5, seed=3)
    assert y.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_preserved(dtype, xb):
    x = xb.to(dtype)
    y, _ = frc(x, severity=0.5, seed=3)
    assert y.dtype == dtype


def test_channels_last_preserved(xb):
    x = xb.contiguous(memory_format=torch.channels_last)
    y, _ = frc(x, severity=0.5, seed=3)
    assert y.is_contiguous(memory_format=torch.channels_last)


def test_finite(xb):
    y, _ = frc(xb, severity=0.7, seed=3)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------
def test_backward_image(xb):
    x = xb.clone().requires_grad_(True)
    y, _ = frc(x, severity=0.5, seed=3)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_backward_severity(xb):
    s = torch.tensor([0.3, 0.5, 0.7, 0.9], requires_grad=True)
    y, _ = frc(xb, severity=s, seed=3)
    y.square().mean().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


@pytest.mark.slow
def test_gradcheck():
    """
    Numerical gradcheck in fp64.

    ``grid_sample`` bilinear backward is only C¹ at the sample points;
    for some N the finite-difference step can land on a kink and blow up
    the tolerance.  We use a small spatial size, N=3, and a looser
    atol/rtol (1e-3 / 1e-3) to keep the check meaningful without
    being flaky.  If it still fails on your platform, mark it
    xfail rather than removing it — it's a real diagnostic.
    """
    x = torch.rand(1, 3, 16, 16, dtype=torch.float64, requires_grad=True)
    s = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

    def fn(xx, ss):
        return frc(xx, severity=ss, seed=0, N=3, sigma_max=1.0)[0]

    assert torch.autograd.gradcheck(
        fn, (x, s), eps=1e-6, atol=1e-3, rtol=1e-3, nondet_tol=1e-8
    )


# ---------------------------------------------------------------------------
# Label & registry
# ---------------------------------------------------------------------------
def test_label_single(xb):
    _, lab = frc(xb[0], severity=0.4, seed=1)
    assert lab.shape == (2,)
    assert lab.dtype == torch.float32
    assert lab.device.type == "cpu"
    assert int(lab[0].item()) == DISTORTION_ID
    assert abs(lab[1].item() - 0.4) < 1e-6


def test_label_batch(xb):
    _, lab = frc(xb, severity=0.4, seed=1)
    assert lab.shape == (4, 2)
    assert lab.dtype == torch.float32
    assert lab.device.type == "cpu"
    assert (lab[:, 0] == DISTORTION_ID).all()
    assert torch.allclose(lab[:, 1], torch.full((4,), 0.4), atol=1e-6)


def test_registry():
    assert DISTORTION_REGISTRY.get(DISTORTION_NAME) is frc


# ---------------------------------------------------------------------------
# Severity validation
# ---------------------------------------------------------------------------
def test_severity_mismatch_raises(xb):
    with pytest.raises(ValueError):
        frc(xb, severity=torch.tensor([0.1, 0.2, 0.3]), seed=0)


@pytest.mark.parametrize("sev_in,expected", [
    (0.0,   0.01),
    (-1.0,  0.01),
    (2.0,   1.0),
    (0.5,   0.5),
])
def test_severity_clamped(xb, sev_in, expected):
    _, lab = frc(xb, severity=sev_in, seed=1)
    assert torch.allclose(lab[:, 1], torch.full((4,), expected), atol=1e-6)


# ---------------------------------------------------------------------------
# Calibration overrides
# ---------------------------------------------------------------------------
def test_override_changes_output(xb):
    yA, _ = frc(xb, severity=0.5, seed=5)
    yB, _ = frc(xb, severity=0.5, seed=5, sigma_max=2.0, N=3)
    assert not torch.equal(yA, yB)


# ---------------------------------------------------------------------------
# High-memory warning
# ---------------------------------------------------------------------------
def test_high_memory_warns():
    """
    Threshold check:
        _MEM_WARN_PIXELS = 512*512*8 = 2_097_152
        B*H*W = 3*1024*1024 = 3_145_728  >  2_097_152  → warning fires.
    Using B=2 gives exactly the threshold (not strictly greater) and
    would NOT fire, which was the previous bug.
    """
    x = torch.rand(3, 3, 1024, 1024)
    with pytest.warns(RuntimeWarning, match="high-memory"):
        frc(x, severity=0.5, seed=1, N=20)
