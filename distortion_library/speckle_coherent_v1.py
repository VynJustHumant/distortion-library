"""
Speckle Noise Distortion — Coherent Imaging (Laser / SAR / Ultrasound / Holography)
Registry ID: speckle_coherent_v1   VERSION: 1.0.8   DISTORTION_ID: 1008

Semantics
---------
Multiplicative speckle applied to *linear intensity* images (scene-referred):

    y = (1 - s) * x + s * x * m,   m > 0,  E[m] = 1  ⟹  E[y] = x.

For sRGB / gamma-encoded inputs, linearize before this module and re-encode after.

Mathematical model
------------------
Lognormal default (differentiable w.r.t. severity and image):
    m = exp(σ·n − ½σ²),   n ~ N(0,1),   σ² = log(1 + 1/L),
    L(s) = 1 + (L_max − 1) · ((1−s)/(1−s_min))^γ,   s_min = 0.01.
    E[m] = 1;  Var[m] = 1/L;  skewness = (e^σ²+2)·sqrt(e^σ²−1);
    excess kurtosis = e^{4σ²} + 2e^{3σ²} + 3e^{2σ²} − 6.

Exact-gamma path (opt-in, `exact_gamma=True`):
    m = u / L,   u ~ Gamma(L, 1)   ⟹  m ~ Gamma(L, L),  Var[m] = 1/L.
    L detached during sampling; blend still carries ∂/∂s.

Correlated path (opt-in, `rho>0`):
    m_raw = |G * (ε_r + i·ε_i)|² / (2·‖G‖₂²),   ε_r, ε_i ~ N(0,1),
    m     = 1 + scale(s) · (m_raw − 1),   scale(s) = sqrt(1 / L(s)).

Distributional note
-------------------
Correlated path matches lognormal path on E[m] and Var[m] at every severity,
but higher moments (skewness, excess kurtosis) differ. Interchangeable for
mean/variance-based calibration (MSE, PSNR); NOT for tail-sensitive metrics.

Model is a Gaussian-PSF approximation, not exact coherent propagation.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
try:
    from distortion_library.blur_atmospheric_turbulence_v1 import (  # type: ignore
        DISTORTION_REGISTRY,
    )
except Exception:
    try:
        from blur_atmospheric_turbulence_v1 import (  # type: ignore
            DISTORTION_REGISTRY,
        )
    except Exception:
        DISTORTION_REGISTRY: Dict[str, type] = {}


def register_distortion(name: str, version: str = "1.0.0"):
    """Register `cls` in the shared DISTORTION_REGISTRY (single source of truth)."""
    def deco(cls):
        DISTORTION_REGISTRY[name] = cls
        cls.DISTORTION_NAME = name
        cls.DISTORTION_VERSION = version
        return cls
    return deco


REGISTRY_ID: str = "speckle_coherent_v1"
VERSION: str = "1.0.8"
REGISTRY_ID_INT: int = 1008

_S_MIN: float = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det_hash(*values: int) -> int:
    """Deterministic BLAKE2b-64 hash (builtin `hash` is process-randomized)."""
    h = hashlib.blake2b(digest_size=8)
    for v in values:
        iv = int(v) & 0xFFFFFFFFFFFFFFFF
        h.update(iv.to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "little")


def _severity_to_L(
    s: torch.Tensor,
    L_max: float = 1000.0,
    gamma: float = 2.0,
    s_min: float = _S_MIN,
) -> torch.Tensor:
    """s ∈ [0.01, 1.0] → effective number of looks L ≥ 1."""
    ratio = ((1.0 - s) / (1.0 - s_min)).clamp(0.0, 1.0)
    return (1.0 + (L_max - 1.0) * ratio.pow(gamma)).clamp_min(1.0)


def _sigma_from_L(L: torch.Tensor) -> torch.Tensor:
    """Lognormal σ² = log(1 + 1/L)."""
    return torch.log1p(1.0 / L).clamp_min(0.0).sqrt()


def _make_gaussian_kernel(
    rho: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """2D Gaussian kernel (outer product of two 1D Gaussians), shape (1,1,k,k)."""
    r = max(1, int(math.ceil(3.0 * rho)))
    k = 2 * r + 1
    ax = torch.arange(k, device=device, dtype=dtype) - r
    g1 = torch.exp(-0.5 * (ax / max(rho, 1e-6)) ** 2)
    g1 = g1 / g1.sum()
    return (g1[:, None] * g1[None, :]).view(1, 1, k, k)


def _sample_normal(
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
    seed: Optional[int],
    stream_tag: int,
) -> torch.Tensor:
    """
    Batch-invariant N(0,1) sampling.

    Priority:
      1. explicit `generator`  → forward to torch.randn.
      2. explicit `seed`       → per-sample BLAKE2b-derived generators.
      3. neither               → torch.randn on the global RNG (fallback).
    """
    B = shape[0]
    if generator is not None:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    if seed is not None:
        gen_dev = torch.device(device).type
        rows = []
        for b in range(B):
            sb = _det_hash(int(seed), REGISTRY_ID_INT, b, stream_tag) % (2 ** 31)
            g = torch.Generator(device=gen_dev)
            g.manual_seed(sb)
            rows.append(
                torch.randn((1, *shape[1:]), generator=g, device=device, dtype=dtype)
            )
        return torch.cat(rows, dim=0)
    return torch.randn(shape, device=device, dtype=dtype)


def _standard_gamma(
    concentration: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample Gamma(concentration, 1). Public path preferred."""
    if generator is None:
        return torch.distributions.Gamma(concentration, 1.0).sample()
    warnings.warn(
        "speckle_coherent_v1: exact_gamma with an explicit `seed=`/`generator=` "
        "relies on torch._standard_gamma (private API). For portability, use "
        "the default lognormal path, or accept the private-API dependency.",
        RuntimeWarning,
        stacklevel=4,
    )
    return torch._standard_gamma(concentration, generator=generator)


def _sample_gamma_unit_mean(
    shape: Tuple[int, ...],
    L_exp: torch.Tensor,
    *,
    device: torch.device,
    generator: Optional[torch.Generator],
    seed: Optional[int],
    stream_tag: int,
) -> torch.Tensor:
    """u ~ Gamma(L, L)  ⟺  u = X / L,  X ~ Gamma(L, 1).  E[u]=1, Var[u]=1/L."""
    L_safe = L_exp.clamp_min(1.0)
    if generator is not None or seed is None:
        x = _standard_gamma(L_safe, generator=generator)
        return x / L_safe
    B = shape[0]
    gen_dev = torch.device(device).type
    rows = []
    for b in range(B):
        sb = _det_hash(int(seed), REGISTRY_ID_INT, b, stream_tag) % (2 ** 31)
        g = torch.Generator(device=gen_dev)
        g.manual_seed(sb)
        lb = L_safe[b : b + 1].contiguous()
        xb = _standard_gamma(lb, generator=g)
        rows.append(xb / lb)
    return torch.cat(rows, dim=0)


def _resolve_compute_dtype(
    image: torch.Tensor, compute_dtype: Optional[torch.dtype]
) -> torch.dtype:
    """fp64 inputs stay fp64 unless overridden."""
    if compute_dtype is not None:
        return compute_dtype
    return torch.float64 if image.dtype == torch.float64 else torch.float32


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------

def distortion(
    image: torch.Tensor,
    severity: Union[torch.Tensor, float, Sequence[float]],
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    value_range: Tuple[float, float] = (0.0, 1.0),
    L_max: float = 1000.0,
    gamma: float = 2.0,
    rho: Optional[float] = None,
    channel_independent: bool = False,
    exact_gamma: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply coherent-imaging speckle distortion."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("`image` must be a torch.Tensor.")
    if image.dim() not in (3, 4):
        raise ValueError("`image` must be (C,H,W) or (B,C,H,W).")

    orig_dtype = image.dtype
    squeezed = image.dim() == 3
    if image.dim() == 4:
        if image.is_contiguous(memory_format=torch.channels_last):
            orig_mem_fmt = torch.channels_last
        else:
            orig_mem_fmt = None
    else:
        orig_mem_fmt = None
    if squeezed:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    device = image.device
    cdt = _resolve_compute_dtype(image, compute_dtype)

    if rho is not None:
        rho = float(rho)
        if rho < 0.0:
            raise ValueError(f"rho must be >= 0.0, got {rho}.")
        if rho == 0.0:
            rho = None
        else:
            r_est = int(math.ceil(3.0 * rho))
            if r_est > 0.5 * min(H, W):
                warnings.warn(
                    f"rho={rho} produces a {2 * r_est + 1}² kernel against a "
                    f"{H}×{W} image; conv2d cost grows as O(rho²) and the "
                    f"kernel exceeds half the image extent, so most of it is "
                    f"padding. Consider rho ≤ {min(H, W) / 6:.2f}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    if rho is not None and exact_gamma:
        raise ValueError(
            "`rho` (correlated) and `exact_gamma=True` are mutually exclusive."
        )

    vmin, vmax = float(value_range[0]), float(value_range[1])
    span = (vmax - vmin) if vmax != vmin else 1.0

    x = image.to(cdt)
    x = (x - vmin) / span
    x = x.clamp(0.0, 1.0)
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    if isinstance(severity, torch.Tensor):
        sev = severity.to(device=device, dtype=cdt)
    else:
        sev = torch.as_tensor(severity, dtype=cdt, device=device)

    if sev.dim() == 0:
        sev = sev.expand(B)
    elif sev.dim() == 1:
        if sev.numel() == 1:
            sev = sev.expand(B)
        elif sev.numel() != B:
            raise ValueError(f"severity batch {sev.numel()} != image batch {B}.")
    elif sev.dim() == 4:
        sev = sev.reshape(-1)
        if sev.numel() == 1:
            sev = sev.expand(B)
        elif sev.numel() != B:
            raise ValueError(f"severity batch {sev.numel()} != image batch {B}.")
    else:
        raise ValueError(f"Unsupported severity shape: {tuple(sev.shape)}.")
    sev = sev.contiguous()

    s = sev.clamp(_S_MIN, 1.0)
    L = _severity_to_L(s, L_max=float(L_max), gamma=float(gamma))
    s_b = s.view(B, 1, 1, 1)

    C_out = C if channel_independent else 1
    noise_shape = (B, C_out, H, W)

    if rho is not None:
        eps_r = _sample_normal(
            noise_shape, device=device, dtype=cdt,
            generator=generator, seed=seed, stream_tag=10,
        )
        eps_i = _sample_normal(
            noise_shape, device=device, dtype=cdt,
            generator=generator, seed=seed, stream_tag=11,
        )
        kernel = _make_gaussian_kernel(rho, device, cdt)
        kH, kW = kernel.shape[-2], kernel.shape[-1]
        pad_h, pad_w = kH // 2, kW // 2

        if H > pad_h and W > pad_w:
            pad_mode = "reflect"
        elif H >= pad_h + 1 and W >= pad_w + 1:
            pad_mode = "replicate"
        else:
            pad_mode = "constant"
        eps_r = F.pad(eps_r, (pad_w, pad_w, pad_h, pad_h), mode=pad_mode)
        eps_i = F.pad(eps_i, (pad_w, pad_w, pad_h, pad_h), mode=pad_mode)

        k_exp = kernel.expand(C_out, 1, kH, kW)
        c_r = F.conv2d(eps_r, k_exp, groups=C_out)
        c_i = F.conv2d(eps_i, k_exp, groups=C_out)
        power = c_r * c_r + c_i * c_i
        norm = 2.0 * (kernel * kernel).sum().clamp_min(1e-12)
        m_raw = power / norm

        inv_L = 1.0 / L.clamp_min(1.0)
        scale_b = inv_L.sqrt().view(B, 1, 1, 1)
        m = 1.0 + scale_b * (m_raw - 1.0)
        m = m.clamp_min(0.0)

    elif exact_gamma:
        L_b = L.detach().clamp_min(1.0).view(B, 1, 1, 1)
        L_exp = L_b.expand(noise_shape).contiguous()
        m = _sample_gamma_unit_mean(
            noise_shape, L_exp, device=device,
            generator=generator, seed=seed, stream_tag=20,
        )

    else:
        sigma_b = _sigma_from_L(L).view(B, 1, 1, 1)
        n = _sample_normal(
            noise_shape, device=device, dtype=cdt,
            generator=generator, seed=seed, stream_tag=30,
        )
        m = torch.exp(sigma_b * n - 0.5 * sigma_b * sigma_b)

    y = (1.0 - s_b) * x + s_b * x * m
    y = y.clamp(0.0, 1.0)

    out = (y * span + vmin).to(orig_dtype)
    if orig_mem_fmt is not None:
        out = out.contiguous(memory_format=orig_mem_fmt)
    if squeezed:
        out = out.squeeze(0)

    sev_cpu = sev.detach().to("cpu", dtype=torch.float32)
    id_col = torch.full_like(sev_cpu, float(REGISTRY_ID_INT))
    if squeezed:
        label = torch.stack([id_col[0], sev_cpu[0]])
    else:
        label = torch.stack([id_col, sev_cpu], dim=1)

    return out, label


@register_distortion(REGISTRY_ID, version=VERSION)
class SpeckleCoherentV1(nn.Module):
    """Coherent-imaging speckle distortion (see module docstring)."""

    DISTORTION_ID: int = REGISTRY_ID_INT

    def __init__(
        self,
        L_max: float = 1000.0,
        gamma: float = 2.0,
        rho: Optional[float] = None,
        channel_independent: bool = False,
        exact_gamma: bool = False,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if rho is not None and rho < 0.0:
            raise ValueError(f"rho must be >= 0.0, got {rho}.")
        if rho is not None and exact_gamma:
            raise ValueError("rho and exact_gamma are mutually exclusive.")
        self.rho = None if (rho is None or float(rho) == 0.0) else float(rho)
        self.L_max = float(L_max)
        self.gamma = float(gamma)
        self.channel_independent = bool(channel_independent)
        self.exact_gamma = bool(exact_gamma)
        self.compute_dtype = compute_dtype

    def forward(
        self,
        image: torch.Tensor,
        severity: Union[torch.Tensor, float, Sequence[float]],
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        value_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return distortion(
            image, severity,
            seed=seed, generator=generator, value_range=value_range,
            L_max=self.L_max, gamma=self.gamma, rho=self.rho,
            channel_independent=self.channel_independent,
            exact_gamma=self.exact_gamma,
            compute_dtype=self.compute_dtype,
        )


__all__ = [
    "distortion",
    "SpeckleCoherentV1",
    "DISTORTION_REGISTRY",
    "REGISTRY_ID",
    "VERSION",
    "REGISTRY_ID_INT",
]
