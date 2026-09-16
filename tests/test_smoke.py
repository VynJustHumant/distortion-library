"""Smoke tests untuk blur_atmospheric_turbulence_v1."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from distortion_library.blur_atmospheric_turbulence_v1 import (
    blur_atmospheric_turbulence_v1 as blur,
)


def test_import():
    """Modul bisa di-import."""
    assert blur is not None


def test_shape_preserved():
    """Input (1,3,64,64) → output (1,3,64,64)."""
    x = torch.rand(1, 3, 64, 64)
    y, lab = blur(x, 0.5, seed=42)
    assert y.shape == x.shape, f"shape mismatch: {y.shape}"
    assert lab.shape == (1, 2), f"label shape: {lab.shape}"


def test_single_3d():
    """Input (3,64,64) → output (3,64,64), label (2,)."""
    x = torch.rand(3, 64, 64)
    y, lab = blur(x, 0.5, seed=42)
    assert y.shape == x.shape
    assert lab.shape == (2,)


def test_output_range():
    """Output dalam [0,1], bebas NaN/Inf."""
    x = torch.rand(1, 3, 64, 64)
    y, _ = blur(x, 0.5, seed=42)
    assert y.min() >= 0.0
    assert y.max() <= 1.0
    assert torch.isfinite(y).all()


def test_identity():
    """severity 0.01 → output ≈ input."""
    x = torch.rand(1, 3, 64, 64)
    y, _ = blur(x, 0.01, seed=42)
    diff = (y - x).abs().mean().item()
    assert diff < 0.01, f"identity diff too large: {diff}"


def test_monotonicity():
    """MSE naik monoton terhadap severity."""
    x = torch.rand(1, 3, 64, 64)
    sevs = [0.1, 0.3, 0.5, 0.7, 1.0]
    mses = []
    for s in sevs:
        y, _ = blur(x, s, seed=42)
        mses.append(((y - x) ** 2).mean().item())
    for i in range(len(mses) - 1):
        assert mses[i] <= mses[i + 1] + 1e-4, f"not monotonic: {mses}"


def test_determinism():
    """Seed sama → output identik."""
    x = torch.rand(1, 3, 64, 64)
    y1, _ = blur(x, 0.5, seed=42)
    y2, _ = blur(x, 0.5, seed=42)
    assert torch.equal(y1, y2)


def test_batch_invariance():
    """Single sample == elemen dalam batch."""
    x1 = torch.rand(1, 3, 64, 64)
    xb = x1.repeat(4, 1, 1, 1)
    y1, _ = blur(x1, 0.6, seed=42)
    yb, _ = blur(xb, 0.6, seed=42)
    diff = (y1 - yb[:1]).abs().max().item()
    assert diff < 1e-4, f"batch invariance broken: {diff}"


def test_differentiability():
    """Gradien mengalir ke input."""
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    y, _ = blur(x, 0.5, seed=42)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_label_id():
    """Label ID == 1004."""
    x = torch.rand(1, 3, 64, 64)
    _, lab = blur(x, 0.5, seed=42)
    assert lab[0, 0].item() == 1004.0
