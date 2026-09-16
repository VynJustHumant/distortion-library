"""
blur_atmospheric_turbulence_v1
==============================
Differentiable PyTorch distortion: atmospheric turbulence blur + heat shimmer.
Linear-RGB optical stage.  Place BEFORE lens / sensor / ISP / compression.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

# FIX #1 — registry GLOBAL
try:                                    # pragma: no cover
    from motion_blur_linear import DISTORTION_REGISTRY  # type: ignore
except Exception:                       # pragma: no cover
    DISTORTION_REGISTRY: dict = {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NAME = "blur_atmospheric_turbulence_v1"
_ID = 1004
_DEFAULT_SEED = 0x5EED5EED
_LAMBDA_REF_NM = 550.0
_WAVELENGTHS_NM = (450.0, 550.0, 650.0)   # RGB (assumes RGB channel order)

# Integer dtypes: dipromote ke fp32, dibulatkan sebelum cast-back.
_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_HALF_DTYPES = (torch.float16, torch.bfloat16)

# FIX #7 — parameter kalibrasi dalam dict
_DEFAULTS = dict(
    r0_min_factor=0.02,      # fraction of min(H,W) at s=1
    r0_max_factor=0.50,      # fraction of min(H,W) at s=0
    gamma_r0=1.5,
    L0_min_factor=2.0,
    L0_max_factor=10.0,
    gamma_L0=1.0,
    sigma_tilt_min=0.01,     # [rad]
    sigma_tilt_max=1.00,
    gamma_tilt=1.2,
    sigma_ho_min=0.02,       # [rad]
    sigma_ho_max=2.00,
    gamma_ho=1.3,
    # FIX #5 — disp_scale:
    # α = (λ/2π)∇⊥φ_tilt [rad]; d_fisik = F·α butuh focal length F [px].
    # Kita serap (λ/2π) dan F ke satu scalar. 0.03 ≈ 3% dari setengah lebar
    # citra pada σ_tilt=1 rad. Override via kwargs.
    disp_scale=0.03,
)


def register_distortion(name: str):
    def _deco(fn):
        DISTORTION_REGISTRY[name] = fn
        return fn
    return _deco


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
_R2_GRID_CACHE: dict = {}
_KR_CACHE: dict = {}


def _r2_grid(kh: int, kw: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (kh, kw, str(device), str(dtype))
    g = _R2_GRID_CACHE.get(key)
    if g is None:
        ys = torch.arange(kh, device=device, dtype=dtype) - (kh - 1) * 0.5
        xs = torch.arange(kw, device=device, dtype=dtype) - (kw - 1) * 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        g = yy * yy + xx * xx
        _R2_GRID_CACHE[key] = g
    return g


def _kernel_radius(H: int, W: int, sigma_ho_max: float) -> int:
    """FIX #3 — radius kernel di-cache, no .item() di forward."""
    key = (H, W, float(sigma_ho_max))
    r = _KR_CACHE.get(key)
    if r is None:
        sigma_max = sigma_ho_max * (
            min(_WAVELENGTHS_NM) / _LAMBDA_REF_NM
        ) ** (-6.0 / 5.0)
        kr = int(math.ceil(3.0 * sigma_max))
        kr = max(1, min(kr, max(1, min(H, W) // 4)))
        _KR_CACHE[key] = kr
        r = kr
    return r


# ---------------------------------------------------------------------------
# Per-sample hashed RNG
# ---------------------------------------------------------------------------
def _per_sample_seed(base_seed: int, salt: str, b: int) -> int:
    key = f"{int(base_seed)}:{_ID}:{salt}:{b}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % (2 ** 31)


def _resolve_base_seed(seed: Optional[int], generator: Optional[torch.Generator]) -> int:
    """
    Prioritas:
      1. seed tidak None       → base_seed = seed
      2. elif generator ada    → base_seed = draw from generator
      3. else                  → base_seed = _DEFAULT_SEED

    FIX #8 — generator HARUS CPU.  GPU generator ditolak karena memaksa
    sinkronisasi device dan merusak cross-device determinism.
    """
    if seed is not None:
        return int(seed) % (2 ** 31)
    if generator is not None:
        if generator.device.type != "cpu":
            raise ValueError(
                f"generator must be a CPU generator (got device="
                f"'{generator.device.type}') for cross-device determinism"
            )
        s = torch.randint(0, 2 ** 31 - 1, (1,), generator=generator,
                          dtype=torch.int64)
        return int(s.item())
    return _DEFAULT_SEED


def _randn_hashed(
    B: int, tail_shape: tuple, base_seed: int, salt: str,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """
    FIX #2 — noise per-sample independen via SHA256(base_seed, salt, b).
    Loop O(B) hanya untuk inisialisasi generator; komputasi vectorised.
    """
    outs = []
    for b in range(B):
        s_b = _per_sample_seed(base_seed, salt, b)
        g = torch.Generator(device="cpu")
        g.manual_seed(s_b)
        n = torch.randn((1, *tail_shape), generator=g, device="cpu",
                        dtype=torch.float32)
        outs.append(n)
    return torch.cat(outs, dim=0).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def _calibrate(s: torch.Tensor, q_min: float, q_max: float, gamma: float) -> torch.Tensor:
    return q_min + (q_max - q_min) * s.clamp(0.0, 1.0).pow(gamma)


def _kolmogorov_phase(B, H, W, r0, L0, device, dtype, base_seed, salt):
    fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype) * (2.0 * math.pi)
    fx = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype) * (2.0 * math.pi)
    KY, KX = torch.meshgrid(fy, fx, indexing="ij")
    K2 = (KX * KX + KY * KY).unsqueeze(0)

    k0_sq = (2.0 * math.pi / L0.view(-1, 1, 1)) ** 2
    r0_b = r0.view(-1, 1, 1).clamp_min(1e-3)

    Phi = 0.023 * r0_b.pow(-5.0 / 3.0) * (K2 + k0_sq).clamp_min(1e-12).pow(-11.0 / 6.0)

    noise = _randn_hashed(B, (H, W), base_seed, salt, device, dtype)
    Wf = torch.fft.fft2(noise)
    dkx = 2.0 * math.pi / float(W)
    dky = 2.0 * math.pi / float(H)
    F_phi = Wf * torch.sqrt((Phi * dkx * dky).clamp_min(1e-30))
    return torch.real(torch.fft.ifft2(F_phi))


def _lowfreq_field(B, H, W, device, dtype, base_seed, salt, lres=8):
    lh = max(2, min(lres, H))
    lw = max(2, min(lres, W))
    n = _randn_hashed(B, (1, lh, lw), base_seed, salt, device, dtype)
    n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=True).squeeze(1)
    std = n.reshape(B, -1).std(dim=1).clamp_min(1e-8).view(B, 1, 1)
    return n / std


def _gaussian_psf(B, C, kh, kw, sigma, device, dtype):
    r2 = _r2_grid(kh, kw, device, dtype).unsqueeze(0).unsqueeze(0)
    sig = sigma.view(B, C, 1, 1).clamp_min(0.05)
    k = torch.exp(-r2 / (2.0 * sig * sig))
    k = k / k.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@register_distortion(_NAME)
def blur_atmospheric_turbulence_v1(
    image: torch.Tensor,
    severity: Union[float, torch.Tensor],
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    value_range: Tuple[float, float] = (0.0, 1.0),
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Atmospheric-turbulence blur + heat shimmer (linear RGB).

    Parameters
    ----------
    image : Tensor (C,H,W) or (B,C,H,W).  Float, uint8, atau int (di-promote
        internal).  **Linear RGB domain.**  Gamma-encoded input harus
        di-linearise dulu.
    severity : float | Tensor
        scalar, (B,), atau (B,1,1,1).  Clamp ke [0.01, 1.0].  Mismatch
        non-scalar vs batch → ValueError.
    seed : int | None
        Base seed eksplisit.  Menang atas ``generator``.
    generator : torch.Generator | None
        **CPU-only.**  CUDA/other → ValueError.  Berlaku untuk seluruh batch
        (base_seed ditarik sekali, kemudian di-derive per sample via hash).
    value_range : (min, max)
    **kwargs
        Override parameter kalibrasi.  Kunci yang dikenali:
        r0_min_factor, r0_max_factor, gamma_r0,
        L0_min_factor, L0_max_factor, gamma_L0,
        sigma_tilt_min, sigma_tilt_max, gamma_tilt,
        sigma_ho_min, sigma_ho_max, gamma_ho,
        disp_scale.

    Returns
    -------
    distorted : Tensor, same shape/dtype/device/memory_format.
    label     : Tensor (2,) atau (B,2) = [distortion_id, severity], detached.
    """
    # ---- shape book-keeping ------------------------------------------------
    orig_dtype = image.dtype
    device = image.device
    channels_last_in = (
        image.dim() == 4 and image.is_contiguous(memory_format=torch.channels_last)
    )
    squeezed = image.dim() == 3
    if squeezed:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"image must be 3D or 4D, got {tuple(image.shape)}")
    B, C, H, W = image.shape

    # Integer/half dtypes di-promote ke fp32 supaya matematika normalisasi
    # tidak jatuh ke integer arithmetic.
    is_int_input = orig_dtype in _INT_DTYPES
    if orig_dtype in _HALF_DTYPES or is_int_input:
        work_dtype = torch.float32
    else:
        work_dtype = orig_dtype

    # ---- sanitise + normalise ---------------------------------------------
    x = image.to(work_dtype)
    x = torch.where(torch.isfinite(x), x,
                    torch.zeros((), device=device, dtype=work_dtype))

    vmin, vmax = float(value_range[0]), float(value_range[1])
    span = max(vmax - vmin, 1e-8)
    x = ((x - vmin) / span).clamp(0.0, 1.0)

    # ---- severity ----------------------------------------------------------
    if not isinstance(severity, torch.Tensor):
        sev = torch.tensor(float(severity), device=device, dtype=work_dtype)
    else:
        sev = severity.to(device=device, dtype=work_dtype)
    sev = sev.reshape(-1)
    if sev.numel() == 1:
        sev = sev.expand(B)
    elif sev.numel() != B:
        raise ValueError(
            f"severity must be scalar or have numel==B={B}, "
            f"got shape {tuple(severity.shape)}"
        )
    sev = sev.clamp(0.01, 1.0).contiguous()

    # ---- calibration  (FIX #5 — kwargs overrides) --------------------------
    D = dict(_DEFAULTS)
    for k, v in kwargs.items():
        if k in D:
            D[k] = v

    minHW = float(min(H, W))
    r0 = _calibrate(sev, D["r0_max_factor"] * minHW,
                         D["r0_min_factor"] * minHW, D["gamma_r0"])
    L0 = _calibrate(sev, D["L0_max_factor"] * minHW,
                         D["L0_min_factor"] * minHW, D["gamma_L0"])
    sigma_tilt = _calibrate(sev, D["sigma_tilt_min"], D["sigma_tilt_max"], D["gamma_tilt"])
    sigma_ho = _calibrate(sev, D["sigma_ho_min"], D["sigma_ho_max"], D["gamma_ho"])

    # ---- base seed --------------------------------------------------------
    base_seed = _resolve_base_seed(seed, generator)

    # ---- Kolmogorov phase -------------------------------------------------
    phi = _kolmogorov_phase(B, H, W, r0, L0, device, work_dtype,
                            base_seed, salt="phase")
    phi_std = phi.reshape(B, -1).std(dim=1).clamp_min(1e-8).view(B, 1, 1)
    phi = phi / phi_std * sigma_ho.view(B, 1, 1)

    # ---- tilt / heat-shimmer displacement ---------------------------------
    dx = _lowfreq_field(B, H, W, device, work_dtype, base_seed, salt="tilt_dx")
    dy = _lowfreq_field(B, H, W, device, work_dtype, base_seed, salt="tilt_dy")
    scale = D["disp_scale"] * sigma_tilt.view(B, 1, 1)
    dx = dx * scale
    dy = dy * scale

    # ---- chromatic scaling -------------------------------------------------
    if C == 3:
        lam = torch.tensor(_WAVELENGTHS_NM, device=device, dtype=work_dtype).view(1, 3) \
              / _LAMBDA_REF_NM
    else:
        lam = torch.ones(1, C, device=device, dtype=work_dtype)

    dx_c = dx.unsqueeze(1) * lam.view(1, C, 1, 1)
    dy_c = dy.unsqueeze(1) * lam.view(1, C, 1, 1)

    # ---- warp grid ---------------------------------------------------------
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=work_dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=work_dtype)
    GY, GX = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([GX, GY], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,2)

    # JANGAN clamp ke [-1,1] — biarkan grid_sample + padding_mode='reflection'
    # yang menangani overshoot.  Kalau di-clamp, refleksi tidak pernah aktif
    # dan pixel tepi "menempel".
    grid = base + torch.stack([dx_c, dy_c], dim=-1)                 # (B,C,H,W,2)

    x_flat = x.reshape(B * C, 1, H, W)
    grid_flat = grid.reshape(B * C, H, W, 2)
    x_warp = F.grid_sample(
        x_flat, grid_flat,
        mode="bilinear", padding_mode="reflection", align_corners=True,
    ).reshape(B, C, H, W)
    x_warp = x_warp.clamp(0.0, 1.0)

    # ---- PSF convolution ---------------------------------------------------
    sigma_psf = sigma_ho.view(B, 1).expand(B, C) * lam.pow(-6.0 / 5.0)
    sigma_psf = sigma_psf.clamp_min(0.05)

    kr = _kernel_radius(H, W, D["sigma_ho_max"])   # FIX #3
    kh = kw = 2 * kr + 1

    psf = _gaussian_psf(B, C, kh, kw, sigma_psf, device, work_dtype)
    weight = psf.reshape(B * C, 1, kh, kw)

    x_in = x_warp.reshape(1, B * C, H, W)
    x_conv = F.conv2d(x_in, weight, padding=(kr, kr), groups=B * C).reshape(B, C, H, W)
    x_conv = x_conv.clamp(0.0, 1.0)

    # ---- single identity blend --------------------------------------------
    s = sev.view(B, 1, 1, 1)
    y = (1.0 - s) * x + s * x_conv
    y = y.clamp(0.0, 1.0)

    # ---- denormalise -------------------------------------------------------
    y = y * span + vmin

    # FIX (isfinite before int cast) — cek finite SELALU di work_dtype
    # (floating).  Integer tidak punya konsep non-finite; memanggil isfinite
    # pada tensor int secara semantik salah meski PyTorch mengizinkan.
    y = torch.where(torch.isfinite(y), y,
                    torch.zeros((), device=device, dtype=work_dtype))

    # Bulatkan SEBELUM cast ke integer supaya 0.9 → 1, bukan 0 (truncate).
    if is_int_input:
        y = torch.round(y)
    y = y.to(orig_dtype)

    if squeezed:
        y = y.squeeze(0)
    if channels_last_in and y.dim() == 4:
        y = y.contiguous(memory_format=torch.channels_last)

    # ---- label -------------------------------------------------------------
    sev_det = sev.detach().reshape(-1).to(torch.float32)
    id_col = torch.full_like(sev_det, float(_ID))
    label_full = torch.stack([id_col, sev_det], dim=-1)
    label = label_full[0] if squeezed else label_full
    label = label.to(device=device)

    return y, label


__all__ = [
    "blur_atmospheric_turbulence_v1",
    "DISTORTION_REGISTRY",
    "register_distortion",
      ]
