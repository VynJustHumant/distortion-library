"""
blur_atmospheric_turbulence_v1
==============================
Differentiable PyTorch distortion: atmospheric turbulence blur + heat shimmer.
"""
from __future__ import annotations
import hashlib
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F

try:
    from motion_blur_linear import DISTORTION_REGISTRY
except Exception:
    DISTORTION_REGISTRY: dict = {}

_NAME = "blur_atmospheric_turbulence_v1"
_ID = 1004
_DEFAULT_SEED = 0x5EED5EED
_LAMBDA_REF_NM = 550.0
_WAVELENGTHS_NM = (450.0, 550.0, 650.0)
_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_HALF_DTYPES = (torch.float16, torch.bfloat16)

_DEFAULTS = dict(
    r0_min_factor=0.02, r0_max_factor=0.50, gamma_r0=1.5,
    L0_min_factor=2.0, L0_max_factor=10.0, gamma_L0=1.0,
    sigma_tilt_min=0.01, sigma_tilt_max=1.00, gamma_tilt=1.2,
    sigma_ho_min=0.02, sigma_ho_max=2.00, gamma_ho=1.3,
    disp_scale=0.03,
)

def register_distortion(name):
    def _deco(fn):
        DISTORTION_REGISTRY[name] = fn
        return fn
    return _deco

_R2_GRID_CACHE = {}
_KR_CACHE = {}

def _r2_grid(kh, kw, device, dtype):
    key = (kh, kw, str(device), str(dtype))
    g = _R2_GRID_CACHE.get(key)
    if g is None:
        ys = torch.arange(kh, device=device, dtype=dtype) - (kh - 1) * 0.5
        xs = torch.arange(kw, device=device, dtype=dtype) - (kw - 1) * 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        g = yy * yy + xx * xx
        _R2_GRID_CACHE[key] = g
    return g

def _kernel_radius(H, W, sigma_ho_max):
    key = (H, W, float(sigma_ho_max))
    r = _KR_CACHE.get(key)
    if r is None:
        sigma_max = sigma_ho_max * (min(_WAVELENGTHS_NM) / _LAMBDA_REF_NM) ** (-6.0 / 5.0)
        kr = int(math.ceil(3.0 * sigma_max))
        kr = max(1, min(kr, max(1, min(H, W) // 4)))
        _KR_CACHE[key] = kr
        r = kr
    return r

def _per_sample_seed(base_seed, salt, b):
    key = f"{int(base_seed)}:{_ID}:{salt}:{b}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % (2 ** 31)

def _resolve_base_seed(seed, generator):
    if seed is not None:
        return int(seed) % (2 ** 31)
    if generator is not None:
        if generator.device.type != "cpu":
            raise ValueError("generator must be a CPU generator")
        s = torch.randint(0, 2 ** 31 - 1, (1,), generator=generator, dtype=torch.int64)
        return int(s.item())
    return _DEFAULT_SEED

def _randn_hashed(B, tail_shape, base_seed, salt, device, dtype):
    outs = []
    for b in range(B):
        s_b = _per_sample_seed(base_seed, salt, b)
        g = torch.Generator(device="cpu")
        g.manual_seed(s_b)
        n = torch.randn((1, *tail_shape), generator=g, device="cpu", dtype=torch.float32)
        outs.append(n)
    return torch.cat(outs, dim=0).to(device=device, dtype=dtype)

def _calibrate(s, q_min, q_max, gamma):
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

@register_distortion(_NAME)
def blur_atmospheric_turbulence_v1(
    image, severity, seed=None, generator=None,
    value_range=(0.0, 1.0), **kwargs,
):
    orig_dtype = image.dtype
    device = image.device
    channels_last_in = (
        image.dim() == 4 and image.is_contiguous(memory_format=torch.channels_last)
    )
    squeezed = image.dim() == 3
    if squeezed:
        image = image.unsqueeze(0)
    B, C, H, W = image.shape

    is_int_input = orig_dtype in _INT_DTYPES
    if orig_dtype in _HALF_DTYPES or is_int_input:
        work_dtype = torch.float32
    else:
        work_dtype = orig_dtype

    x = image.to(work_dtype)
    x = torch.where(torch.isfinite(x), x, torch.zeros((), device=device, dtype=work_dtype))

    vmin, vmax = float(value_range[0]), float(value_range[1])
    span = max(vmax - vmin, 1e-8)
    x = ((x - vmin) / span).clamp(0.0, 1.0)

    if not isinstance(severity, torch.Tensor):
        sev = torch.tensor(float(severity), device=device, dtype=work_dtype)
    else:
        sev = severity.to(device=device, dtype=work_dtype)
    sev = sev.reshape(-1)
    if sev.numel() == 1:
        sev = sev.expand(B)
    elif sev.numel() != B:
        raise ValueError(f"severity must be scalar or numel==B={B}")
    sev = sev.clamp(0.01, 1.0).contiguous()

    D = dict(_DEFAULTS)
    for k, v in kwargs.items():
        if k in D:
            D[k] = v

    minHW = float(min(H, W))
    r0 = _calibrate(sev, D["r0_max_factor"] * minHW, D["r0_min_factor"] * minHW, D["gamma_r0"])
    L0 = _calibrate(sev, D["L0_max_factor"] * minHW, D["L0_min_factor"] * minHW, D["gamma_L0"])
    sigma_tilt = _calibrate(sev, D["sigma_tilt_min"], D["sigma_tilt_max"], D["gamma_tilt"])
    sigma_ho = _calibrate(sev, D["sigma_ho_min"], D["sigma_ho_max"], D["gamma_ho"])

    base_seed = _resolve_base_seed(seed, generator)

    phi = _kolmogorov_phase(B, H, W, r0, L0, device, work_dtype, base_seed, "phase")
    phi_std = phi.reshape(B, -1).std(dim=1).clamp_min(1e-8).view(B, 1, 1)
    phi = phi / phi_std * sigma_ho.view(B, 1, 1)

    dx = _lowfreq_field(B, H, W, device, work_dtype, base_seed, "tilt_dx")
    dy = _lowfreq_field(B, H, W, device, work_dtype, base_seed, "tilt_dy")
    scale = D["disp_scale"] * sigma_tilt.view(B, 1, 1)
    dx = dx * scale
    dy = dy * scale

    if C == 3:
        lam = torch.tensor(_WAVELENGTHS_NM, device=device, dtype=work_dtype).view(1, 3) / _LAMBDA_REF_NM
    else:
        lam = torch.ones(1, C, device=device, dtype=work_dtype)

    dx_c = dx.unsqueeze(1) * lam.view(1, C, 1, 1)
    dy_c = dy.unsqueeze(1) * lam.view(1, C, 1, 1)

    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=work_dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=work_dtype)
    GY, GX = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([GX, GY], dim=-1).unsqueeze(0).unsqueeze(0)
    grid = base + torch.stack([dx_c, dy_c], dim=-1)

    x_flat = x.reshape(B * C, 1, H, W)
    grid_flat = grid.reshape(B * C, H, W, 2)
    x_warp = F.grid_sample(x_flat, grid_flat, mode="bilinear",
                            padding_mode="reflection", align_corners=True).reshape(B, C, H, W)
    x_warp = x_warp.clamp(0.0, 1.0)

    sigma_psf = sigma_ho.view(B, 1).expand(B, C) * lam.pow(-6.0 / 5.0)
    sigma_psf = sigma_psf.clamp_min(0.05)
    kr = _kernel_radius(H, W, D["sigma_ho_max"])
    kh = kw = 2 * kr + 1
    psf = _gaussian_psf(B, C, kh, kw, sigma_psf, device, work_dtype)
    weight = psf.reshape(B * C, 1, kh, kw)
    x_in = x_warp.reshape(1, B * C, H, W)
    x_conv = F.conv2d(x_in, weight, padding=(kr, kr), groups=B * C).reshape(B, C, H, W)
    x_conv = x_conv.clamp(0.0, 1.0)

    s = sev.view(B, 1, 1, 1)
    y = (1.0 - s) * x + s * x_conv
    y = y.clamp(0.0, 1.0)
    y = y * span + vmin
    y = torch.where(torch.isfinite(y), y, torch.zeros((), device=device, dtype=work_dtype))

    if is_int_input:
        y = torch.round(y)
    y = y.to(orig_dtype)

    if squeezed:
        y = y.squeeze(0)
    if channels_last_in and y.dim() == 4:
        y = y.contiguous(memory_format=torch.channels_last)

    sev_det = sev.detach().reshape(-1).to(torch.float32)
    id_col = torch.full_like(sev_det, float(_ID))
    label_full = torch.stack([id_col, sev_det], dim=-1)
    label = label_full[0] if squeezed else label_full
    label = label.to(device=device)

    return y, label

__all__ = ["blur_atmospheric_turbulence_v1", "DISTORTION_REGISTRY", "register_distortion"]
