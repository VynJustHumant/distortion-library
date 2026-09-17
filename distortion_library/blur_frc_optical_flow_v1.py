# distortion_library/blur_frc_optical_flow_v1.py
"""
PyTorch distortion: Frame Rate Conversion (FRC) blur via optical-flow
frame interpolation / temporal integration.

Registry ID:  blur_frc_optical_flow_v1

Module name follows the `blur_*` convention used by the rest of the
distortion library (e.g. `blur_atmospheric_turbulence_v1.py`).
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = [
    "blur_frc_optical_flow_v1",
    "DISTORTION_REGISTRY",
    "DISTORTION_NAME",
    "DISTORTION_ID",
    "register_distortion",
]


DISTORTION_NAME = "blur_frc_optical_flow_v1"
DEBUG = False

_DEFAULT_SEED = 0x5EED5EED

_MEM_WARN_N = 15
_MEM_WARN_PIXELS = 512 * 512 * 8


# ---------------------------------------------------------------------------
# Global registry — integrated with the shared pipeline registry if present.
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from motion_blur_linear import DISTORTION_REGISTRY  # type: ignore
except Exception:  # standalone fallback
    DISTORTION_REGISTRY: dict = {}


def register_distortion(name: str):
    def deco(fn):
        DISTORTION_REGISTRY[name] = fn
        fn.__distortion_name__ = name
        return fn
    return deco


def _name_to_id(name: str) -> int:
    # 24-bit id: exactly representable in fp32 labels (max 2**24).
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little") % (2 ** 24)


DISTORTION_ID = _name_to_id(DISTORTION_NAME)


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------
def _hash_seed(base_seed: int, name: str, b: int) -> int:
    key = f"{int(base_seed)}|{name}|{int(b)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") % (2 ** 31)


def _resolve_base_seed(seed: Optional[int], generator: Optional[torch.Generator]) -> int:
    if seed is not None:
        return int(seed)
    if generator is not None:
        return int(generator.initial_seed())
    return _DEFAULT_SEED


def _make_gen(base_seed: int, b_idx: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(_hash_seed(base_seed, DISTORTION_NAME, b_idx))
    return g


# ---------------------------------------------------------------------------
# sRGB <-> linear light
# ---------------------------------------------------------------------------
def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    a = x / 12.92
    b = ((x + 0.055) / 1.055).clamp_min(1e-12).pow(2.4)
    return torch.where(x <= 0.04045, a, b)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    a = x * 12.92
    b = 1.055 * x.clamp_min(1e-12).pow(1.0 / 2.4) - 0.055
    return torch.where(x <= 0.0031308, a, b)


# ---------------------------------------------------------------------------
# Warp via F.grid_sample
# ---------------------------------------------------------------------------
def _make_base_grid(B: int, H: int, W: int,
                    device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    return base.expand(B, -1, -1, -1).contiguous()


def _warp(img: torch.Tensor, flow: torch.Tensor, base_grid: torch.Tensor) -> torch.Tensor:
    B, C, H, W = img.shape
    fx = flow[:, 0] * (2.0 / max(W - 1, 1))
    fy = flow[:, 1] * (2.0 / max(H - 1, 1))
    f = torch.stack([fx, fy], dim=-1)
    grid = base_grid + f
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="reflection", align_corners=True)


# ---------------------------------------------------------------------------
# Vectorised flow synthesis
# ---------------------------------------------------------------------------
def _z2u(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z * 0.7071067811865476))


def _synth_flow(
    B: int, H: int, W: int,
    sigma_u: torch.Tensor,
    base_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sigma_u.dim() == 1:
        sigma_u = sigma_u.view(B, 1, 1, 1)

    h_s = max(4, H // 32)
    w_s = max(4, W // 32)
    n_eps = 2 * h_s * w_s
    block = n_eps + 7

    # Per-sample generator: sample b's RNG stream is independent of B,
    # which is required for batch invariance.  The loop is O(B) and B is
    # typically 1–8, so the overhead is negligible.
    rows = []
    for b in range(B):
        g_b = _make_gen(base_seed, b, device)
        rows.append(
            torch.randn(1, block, generator=g_b, device=device, dtype=dtype)
        )
    raw = torch.cat(rows, dim=0)

    eps = raw[:, :n_eps].view(B, 2, h_s, w_s)
    off = n_eps
    cx = (_z2u(raw[:, off + 0]) - 0.5) * 0.5
    cy = (_z2u(raw[:, off + 1]) - 0.5) * 0.5
    coef = (_z2u(raw[:, off + 2]) - 0.5) * 0.5
    A = raw[:, off + 3: off + 7].view(B, 2, 2) * 0.1
    cx = cx.view(B, 1, 1, 1)
    cy = cy.view(B, 1, 1, 1)
    coef = coef.view(B, 1, 1, 1)

    noise = F.interpolate(eps, size=(H, W), mode="bilinear", align_corners=False)
    n_std = noise.flatten(1).std(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-6)
    noise = noise / n_std

    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)
    dx = xx - cx
    dy = yy - cy

    u_rx = coef * dx
    u_ry = coef * dy
    a00 = A[:, 0, 0].view(B, 1, 1, 1)
    a01 = A[:, 0, 1].view(B, 1, 1, 1)
    a10 = A[:, 1, 0].view(B, 1, 1, 1)
    a11 = A[:, 1, 1].view(B, 1, 1, 1)
    u_ax = a00 * dx + a01 * dy
    u_ay = a10 * dx + a11 * dy

    u_x = noise[:, 0:1] + u_rx + u_ax
    u_y = noise[:, 1:2] + u_ry + u_ay
    u = torch.cat([u_x, u_y], dim=1)

    u_std = u.flatten(1).std(dim=1, keepdim=True).view(B, 1, 1, 1).clamp_min(1e-6)
    u = (u / u_std) * sigma_u
    return u


# ---------------------------------------------------------------------------
# FRC temporal integration — vectorised over N
# ---------------------------------------------------------------------------
def _frc_blur(
    x: torch.Tensor,
    u: torch.Tensor,
    N: int,
    sigma_t: float,
    base_grid: torch.Tensor,
) -> torch.Tensor:
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    if N <= 1:
        alphas = torch.full((1,), 0.5, device=device, dtype=dtype)
        wt = torch.ones((1,), device=device, dtype=dtype)
    else:
        alphas = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)
        wt = torch.exp(-((alphas - 0.5) ** 2) / (2.0 * max(sigma_t, 1e-6) ** 2))
    weights = (wt / wt.sum()).view(N, 1, 1, 1, 1)

    I0 = _warp(x, -0.5 * u, base_grid)
    I1 = _warp(x,  0.5 * u, base_grid)

    I0e = I0.unsqueeze(0).expand(N, -1, -1, -1, -1)
    I1e = I1.unsqueeze(0).expand(N, -1, -1, -1, -1)
    ue = u.unsqueeze(0).expand(N, -1, -1, -1, -1)
    grid_e = base_grid.unsqueeze(0).expand(N, -1, -1, -1, -1)
    a = alphas.view(N, 1, 1, 1, 1)

    I0f = I0e.reshape(N * B, C, H, W)
    I1f = I1e.reshape(N * B, C, H, W)
    uf = ue.reshape(N * B, 2, H, W)
    gf = grid_e.reshape(N * B, H, W, 2)
    af = a.expand(N, B, 1, 1, 1).reshape(N * B, 1, 1, 1)

    Ia = _warp(I0f, -af * uf, gf)
    Ib = _warp(I1f, (1.0 - af) * uf, gf)
    Ik = (1.0 - af) * Ia + af * Ib

    Ik = Ik.reshape(N, B, C, H, W)
    return (weights * Ik).sum(dim=0)


# ---------------------------------------------------------------------------
# Severity handling
# ---------------------------------------------------------------------------
def _prep_severity(
    severity: Union[float, torch.Tensor],
    B_orig: int,
    T: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(severity):
        severity = torch.tensor(float(severity), device=device, dtype=torch.float32)
    else:
        severity = severity.to(device=device, dtype=torch.float32)

    if severity.dim() == 0:
        sev = severity.expand(B_orig)
    else:
        flat = severity.reshape(-1)
        n = flat.numel()
        if n == 1:
            sev = flat[0].expand(B_orig)
        elif n == B_orig:
            sev = flat
        else:
            raise ValueError(
                f"severity has {n} elements; expected scalar or {B_orig} "
                f"(got shape {tuple(severity.shape)})"
            )

    sev = sev.clamp(0.01, 1.0)
    sev_flat = sev.repeat_interleave(T) if T > 1 else sev
    return sev, sev_flat


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
@register_distortion(DISTORTION_NAME)
def blur_frc_optical_flow_v1(
    image: torch.Tensor,
    severity: Union[float, torch.Tensor],
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    value_range: Sequence[float] = (0.0, 1.0),
    sigma_min: float = 0.5,
    sigma_max: float = 8.0,
    gamma_u: float = 1.0,
    N: int = 7,
    sigma_t: float = 0.25,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate Frame Rate Conversion (FRC) blur via optical-flow frame
    interpolation and temporal integration.

    Parameters
    ----------
    image : torch.Tensor
        ``(C,H,W)``, ``(B,C,H,W)`` or ``(B,T,C,H,W)``.  Any dtype/device.
    severity : float | torch.Tensor
        Scalar, ``(B,)``, ``(B,1)`` or ``(B,1,1,1)``.  Clamped to
        ``[0.01, 1.0]``.  Wrong size → ``ValueError``.
    seed : int, optional
        Master seed.  Per-sample seeds derived via
        ``hash((seed, "blur_frc_optical_flow_v1", b)) % 2**31`` → batch-
        invariant.  If both ``seed`` and ``generator`` are ``None``,
        ``_DEFAULT_SEED`` is used.
    generator : torch.Generator, optional
        Only consulted via ``initial_seed()``; state never advanced.
    value_range : (float, float)
        Input normalisation range; mapped to ``[0,1]`` internally.

    Calibration overrides
    ---------------------
    sigma_min, sigma_max : float
        Optical-flow magnitude bounds in pixel units.  Per-sample flow
        std is ``sigma_u = sigma_min + (sigma_max - sigma_min) *
        s**gamma_u``.  Larger = wider motion → more integration blur.
        Default ``0.5`` / ``8.0``.
    gamma_u : float
        Power-law exponent for severity → sigma mapping.  ``1.0`` is
        linear; ``>1`` compresses low-severity region; ``<1`` expands.
    N : int
        Number of temporal samples for FRC integration.  Must be constant
        (not batch-dependent) to preserve batch invariance.  Default 7.
    sigma_t : float
        Temporal shutter weight width: ``w_k ∝ exp(-(α_k-0.5)²/(2σ_t²))``
        with ``α_k = k/(N-1)``.  ``σ_t → 0⁺`` = motion freeze;
        ``σ_t ≥ 0.5`` ≈ uniform shutter.

    Extra keyword arguments are silently accepted for pipeline
    composability.

    Returns
    -------
    y : torch.Tensor
        Same shape / device / dtype / memory format as input.  Finite.
    label : torch.Tensor
        ``(2,)`` for single image, ``(B,2)`` for batch/video — always CPU
        ``float32``, detached, serializable: ``[DISTORTION_ID, s]``.
    """
    if image.numel() == 0:
        lab = torch.zeros((2,), dtype=torch.float32)
        return image.clone(), lab

    orig_dtype = image.dtype
    orig_channels_last = (
        image.dim() == 4 and image.is_contiguous(memory_format=torch.channels_last)
    )

    is_single = image.dim() == 3
    is_video = image.dim() == 5

    if is_single:
        image = image.unsqueeze(0)
        B_orig, T = 1, 1
    elif is_video:
        B_orig, T, C, H, W = image.shape
        image = image.reshape(B_orig * T, C, H, W)
    else:
        B_orig, T = image.shape[0], 1

    B_flat, C, H, W = image.shape
    device = image.device

    N_eff = int(max(1, N))
    if N_eff > _MEM_WARN_N and (B_flat * H * W) > _MEM_WARN_PIXELS:
        warnings.warn(
            f"blur_frc_optical_flow_v1: high-memory configuration "
            f"N={N_eff}, B={B_flat}, HxW={H}x{W} "
            f"(~{N_eff * B_flat * C * H * W * 4 / 1e9:.2f} GB fp32 for the "
            f"expanded FRC buffer).  Consider reducing N or tiling.",
            RuntimeWarning,
            stacklevel=2,
        )

    sev, sev_flat = _prep_severity(severity, B_orig, T, device)

    lo, hi = float(value_range[0]), float(value_range[1])
    x = ((image.float() - lo) / (hi - lo)).clamp(0.0, 1.0)
    L = _srgb_to_linear(x)

    sigma_u_flat = sigma_min + (sigma_max - sigma_min) * sev_flat.pow(gamma_u)

    base_seed = _resolve_base_seed(seed, generator)
    u = _synth_flow(B_flat, H, W, sigma_u_flat, base_seed, device, torch.float32)

    base_grid = _make_base_grid(B_flat, H, W, device, torch.float32)

    Blur = _frc_blur(L, u, N_eff, float(sigma_t), base_grid).clamp(0.0, 1.0)
    B_srgb = _linear_to_srgb(Blur)

    s_view = sev_flat.view(B_flat, 1, 1, 1)
    y = (1.0 - s_view) * x + s_view * B_srgb
    y = y.clamp(0.0, 1.0)

    y = y * (hi - lo) + lo
    y = y.to(orig_dtype)
    y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))

    if DEBUG:
        assert torch.isfinite(y).all(), "non-finite output in FRC blur"

    if is_video:
        y = y.reshape(B_orig, T, C, H, W)
    if is_single:
        y = y.squeeze(0)
    if orig_channels_last and y.dim() == 4:
        y = y.contiguous(memory_format=torch.channels_last)

    sev_cpu = sev.detach().to("cpu", torch.float32).reshape(-1)
    if is_single:
        label = torch.cat([
            torch.tensor([float(DISTORTION_ID)], dtype=torch.float32),
            sev_cpu[:1],
        ])
    else:
        label = torch.stack([
            torch.full((B_orig,), float(DISTORTION_ID), dtype=torch.float32),
            sev_cpu,
        ], dim=1)

    return y, label
