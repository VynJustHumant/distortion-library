"""
compression_recompression_v1  (v3.1)
Differentiable multi-generation JPEG re-compression artifact model.

Changelog
---------
v1  -> v2   : identity blend, registry integration, DISTORTION_ID=1006,
              power-law N_eff, kwargs honored, tests, layered isfinite,
              chroma upsample mode.
v2  -> v3   : device-safe module cache, sibling registry import path,
              assert precedence fix, fp32/CPU labels, real gradcheck via
              `use_ste` toggle, module reuse, value_range(-1,1) tests.
v3  -> v3.1 : LRU-bounded module cache, single register() call,
              RuntimeWarning on use_ste=False, quantitative STE gradient
              assertions.

Reference: SPEC §3.1-3.6, §4-7.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Registry identifiers
# ==============================================================
REGISTRY_ID = "compression_recompression_v1"
DISTORTION_ID = 1006

# [v3-2] Match sibling module import path used across the codebase.
try:
    from motion_blur_linear import DISTORTION_REGISTRY as _SHARED_REGISTRY  # type: ignore
except Exception:                                                  # pragma: no cover
    _SHARED_REGISTRY = None

_LOCAL_REGISTRY: dict = {}


# ==============================================================
# JPEG reference tables
# ==============================================================
_STD_LUM = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

_STD_CHR = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def _make_dct_matrix(n: int = 8, dtype=torch.float32) -> torch.Tensor:
    """D[u, x] = alpha(u) * cos((2x+1) u pi / (2n))."""
    u = torch.arange(n, dtype=dtype)
    x = torch.arange(n, dtype=dtype)
    alpha = torch.where(
        u == 0,
        torch.full_like(u, math.sqrt(1.0 / n)),
        torch.full_like(u, math.sqrt(2.0 / n)),
    )
    angle = (2.0 * x[None, :] + 1.0) * u[:, None] * math.pi / (2.0 * n)
    return alpha[:, None] * torch.cos(angle)


# ==============================================================
# Straight-through estimator for round()
# ==============================================================
class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# ==============================================================
# Core module
# ==============================================================
class _Recompression(nn.Module):
    """Differentiable JPEG-like re-compression, N generations."""

    def __init__(
        self,
        Q_min: float = 1.0,
        Q_max: float = 100.0,
        gamma: float = 2.0,
        gamma_N: Optional[float] = None,
        N_max: int = 8,
        chroma_upsample: str = "nearest",
        use_ste: bool = True,                                      # [v3-5]
    ) -> None:
        super().__init__()
        self.Q_min = float(Q_min)
        self.Q_max = float(Q_max)
        self.gamma = float(gamma)
        self.gamma_N = float(gamma) if gamma_N is None else float(gamma_N)
        self.N_max = int(N_max)
        self.chroma_upsample = str(chroma_upsample)
        self.use_ste = bool(use_ste)
        assert self.chroma_upsample in ("nearest", "bilinear")

        self.register_buffer("D", _make_dct_matrix(8, dtype=torch.float64), persistent=False)
        self.register_buffer("Q_lum", _STD_LUM, persistent=False)
        self.register_buffer("Q_chr", _STD_CHR, persistent=False)

    # ---------- helpers ----------
    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

    @staticmethod
    def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.cat([y, cb, cr], dim=1)

    @staticmethod
    def _ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
        y, cb, cr = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        cb = cb - 0.5
        cr = cr - 0.5
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], dim=1)

    def _quality_scale(self, Q: torch.Tensor) -> torch.Tensor:
        low = 5000.0 / torch.clamp(Q, min=1e-3)
        high = 200.0 - 2.0 * Q
        return torch.where(Q < 50.0, low, high)

    def _quant_step(self, Q: torch.Tensor, table: torch.Tensor,
                    dtype: torch.dtype) -> torch.Tensor:
        S = self._quality_scale(Q)
        base = table.to(dtype=dtype)
        delta = torch.floor((S[:, None, None] * base + 50.0) / 100.0)
        return torch.clamp(delta, 1.0, 255.0)

    def _round(self, x: torch.Tensor) -> torch.Tensor:              # [v3-5]
        return _RoundSTE.apply(x) if self.use_ste else x

    def _dct_quant_idct(self, x: torch.Tensor,
                        delta: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        D = self.D.to(dtype=x.dtype)

        xs = x - 0.5
        L = (H // 8) * (W // 8)
        blocks = F.unfold(xs, kernel_size=8, stride=8).view(B, C, 8, 8, L)

        Fcoef = torch.einsum("ux,bcxyl,vy->bcuvl", D, blocks, D)
        Fcoef = self._sanitize(Fcoef)

        d = delta[:, None, :, :, None]
        q = self._round(Fcoef / d)
        Fq = d * q
        Fq = self._sanitize(Fq)

        rec = torch.einsum("xu,bcuvl,vy->bcxyl", D, Fq, D)
        rec = rec.reshape(B, C * 64, L)
        out = F.fold(rec, output_size=(H, W), kernel_size=8, stride=8)
        out = out + 0.5
        return self._sanitize(out)

    def _jpeg_once(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3, "compression_recompression_v1 expects RGB input"

        ph = (-H) % 16
        pw = (-W) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")

        ycbcr = self._sanitize(self._rgb_to_ycbcr(x))
        y = ycbcr[:, 0:1]
        cb = ycbcr[:, 1:2]
        cr = ycbcr[:, 2:3]

        cb_ds = F.avg_pool2d(cb, kernel_size=2, stride=2)
        cr_ds = F.avg_pool2d(cr, kernel_size=2, stride=2)

        delta_lum = self._quant_step(Q, self.Q_lum, x.dtype)
        delta_chr = self._quant_step(Q, self.Q_chr, x.dtype)

        y_rec = self._dct_quant_idct(y, delta_lum)
        chroma = torch.cat([cb_ds, cr_ds], dim=1)
        chroma_rec = self._dct_quant_idct(chroma, delta_chr)

        up_kwargs = dict(size=(H + ph, W + pw), mode=self.chroma_upsample)
        if self.chroma_upsample != "nearest":
            up_kwargs["align_corners"] = False
        cb_up = F.interpolate(chroma_rec[:, 0:1], **up_kwargs)
        cr_up = F.interpolate(chroma_rec[:, 1:2], **up_kwargs)

        rgb = self._ycbcr_to_rgb(torch.cat([y_rec, cb_up, cr_up], dim=1))
        rgb = rgb[:, :, :H, :W]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return self._sanitize(rgb)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(severity, 0.01, 1.0)                       # (B,)

        # Quality factor Q(s) : power-law
        Q = self.Q_max - (self.Q_max - self.Q_min) * s.pow(self.gamma)
        Q = torch.clamp(Q, self.Q_min, self.Q_max)

        # Generation count N_eff(s) : power-law
        N_eff = 1.0 + (self.N_max - 1.0) * s.pow(self.gamma_N)
        N_eff = torch.clamp(N_eff, 1.0, float(self.N_max))

        n_lo = torch.floor(N_eff)
        n_hi = torch.ceil(N_eff)
        lam = (N_eff - n_lo).view(-1, 1, 1, 1)

        # Fixed loop length -> torch.compile friendly.
        seq = [x]
        for _ in range(self.N_max):
            seq.append(self._jpeg_once(seq[-1], Q))
        stack = torch.stack(seq, dim=0)                            # (N_max+1,B,C,H,W)

        B, C, H, W = x.shape
        idx_lo = n_lo.long().view(1, B, 1, 1, 1).expand(1, B, C, H, W)
        idx_hi = n_hi.long().view(1, B, 1, 1, 1).expand(1, B, C, H, W)
        x_lo = torch.gather(stack, 0, idx_lo).squeeze(0)
        x_hi = torch.gather(stack, 0, idx_hi).squeeze(0)

        x_distorted = (1.0 - lam) * x_lo + lam * x_hi
        x_distorted = self._sanitize(x_distorted)

        # [v2-1] Explicit identity blend ensures MSE(s=0.01) << 0.01.
        s_b = s.view(-1, 1, 1, 1)
        return (1.0 - s_b) * x + s_b * x_distorted


# ==============================================================
# Module cache — device-safe, LRU-bounded
# ==============================================================
_DEFAULT_CFG = dict(
    Q_min=1.0, Q_max=100.0, gamma=2.0, gamma_N=2.0,
    N_max=8, chroma_upsample="nearest", use_ste=True,
)

_MODULE_CACHE: "OrderedDict[tuple, _Recompression]" = OrderedDict()
_CACHE_MAX = 32                                                    # [v3.1-1]


def _resolve_cfg(
    Q_min, Q_max, gamma, gamma_N, N_max, chroma_upsample, use_ste
) -> dict:
    return dict(
        Q_min=_DEFAULT_CFG["Q_min"] if Q_min is None else float(Q_min),
        Q_max=_DEFAULT_CFG["Q_max"] if Q_max is None else float(Q_max),
        gamma=_DEFAULT_CFG["gamma"] if gamma is None else float(gamma),
        gamma_N=_DEFAULT_CFG["gamma_N"] if gamma_N is None else float(gamma_N),
        N_max=_DEFAULT_CFG["N_max"] if N_max is None else int(N_max),
        chroma_upsample=_DEFAULT_CFG["chroma_upsample"]
            if chroma_upsample is None else str(chroma_upsample),
        use_ste=_DEFAULT_CFG["use_ste"] if use_ste is None else bool(use_ste),
    )


def _get_module(device: torch.device, cfg: dict) -> _Recompression:
    key = (str(device), tuple(sorted(cfg.items())))
    cached = _MODULE_CACHE.get(key)
    if cached is not None:
        _MODULE_CACHE.move_to_end(key)                             # [v3.1-1]
        return cached
    mod = _Recompression(**cfg).to(device)                         # [v3-1]
    _MODULE_CACHE[key] = mod
    while len(_MODULE_CACHE) > _CACHE_MAX:                         # [v3.1-1]
        _MODULE_CACHE.popitem(last=False)
    return mod


# ==============================================================
# Public API
# ==============================================================
def distortion(
    image: torch.Tensor,
    severity: Union[float, torch.Tensor],
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    value_range: Tuple[float, float] = (0.0, 1.0),
    Q_min: Optional[float] = None,
    Q_max: Optional[float] = None,
    gamma: Optional[float] = None,
    gamma_N: Optional[float] = None,
    N_max: Optional[int] = None,
    chroma_upsample: Optional[str] = None,
    use_ste: Optional[bool] = None,                                # [v3-5]
    **kwargs,
):
    """
    Multi-generation re-compression artifact.

    Parameters
    ----------
    image : Tensor
        (C,H,W) or (B,C,H,W).  C must be 3 (RGB).  Any dtype/device.
    severity : float or Tensor
        Scalar, (B,), or (B,1,1,1) in [0.01, 1.0].
    seed, generator : optional
        Accepted for API uniformity; this operator is deterministic.
    value_range : (lo, hi)
        Range of the incoming image.  Normalized to [0,1] internally.
    Q_min, Q_max : float, optional
        JPEG quality bounds (default 1, 100).
    gamma : float, optional
        Exponent of Q(s) (default 2.0).
    gamma_N : float, optional
        Exponent of N_eff(s) (default = gamma).
    N_max : int, optional
        Max generation count (default 8).
    chroma_upsample : {'nearest','bilinear'}, optional
        Chroma upsample mode (default 'nearest' = libjpeg-like).
    use_ste : bool, optional
        Enable straight-through quantizer (default True).  Set False only
        for gradcheck/gradient-verification; output becomes ~identity.

    Returns
    -------
    (image_distorted, label)
        image_distorted : same shape/device/dtype/memory_format as input.
        label           : (2,) or (B,2) fp32 on CPU = [DISTORTION_ID, severity].
    """
    orig_dtype = image.dtype
    orig_device = image.device

    # ---- reshape to (B,C,H,W) ----
    if image.dim() == 3:
        x = image.unsqueeze(0)
        squeezed = True
        orig_mem_fmt = torch.contiguous_format
    elif image.dim() == 4:
        x = image
        squeezed = False
        orig_mem_fmt = (
            torch.channels_last
            if x.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )
    else:
        raise ValueError(f"expected 3D or 4D input, got {image.dim()}D")

    B, C, H, W = x.shape
    if C != 3:
        raise ValueError(f"expected 3-channel RGB, got C={C}")

    # ---- severity ----
    if not torch.is_tensor(severity):
        sev = torch.as_tensor(severity, dtype=torch.float32, device=orig_device)
    else:
        sev = severity.to(device=orig_device, dtype=torch.float32)

    if sev.dim() == 0:
        sev_b = sev.expand(B)
    elif sev.dim() == 1:
        sev_b = sev.expand(B) if sev.numel() == 1 else sev
    elif sev.dim() == 4 and tuple(sev.shape[1:]) == (1, 1, 1):
        sev_b = sev.reshape(B)
    else:
        raise ValueError(f"unsupported severity shape {tuple(sev.shape)}")

    if sev_b.numel() != B:
        raise ValueError(f"severity batch mismatch: {sev_b.numel()} vs {B}")
    sev_b = torch.clamp(sev_b, 0.01, 1.0)

    # ---- normalize ----
    lo, hi = float(value_range[0]), float(value_range[1])
    scale = hi - lo
    # Preserve fp64 for gradcheck; promote half-precision to fp32.
    if x.dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    else:
        work_dtype = x.dtype
    x01 = (x.to(work_dtype) - lo) / scale
    x01 = torch.clamp(x01, 0.0, 1.0)
    x_proc = x01.contiguous(memory_format=torch.contiguous_format)

    # ---- resolve config, warn on testing escape hatch ----
    cfg = _resolve_cfg(Q_min, Q_max, gamma, gamma_N, N_max,
                       chroma_upsample, use_ste)

    if cfg["use_ste"] is False:                                    # [v3.1-3]
        warnings.warn(
            "compression_recompression_v1: use_ste=False bypasses the "
            "quantizer — output ≈ identity blend and severity has almost "
            "no effect. Intended for gradient/gradcheck verification only.",
            RuntimeWarning,
            stacklevel=2,
        )

    mod = _get_module(orig_device, cfg)

    # ---- forward ----
    out = mod(x_proc, sev_b)

    # ---- sanitize / clamp / denormalize ----
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    out = torch.clamp(out, 0.0, 1.0)
    out = out * scale + lo
    out = out.to(orig_dtype)

    # ---- restore memory format ----
    if not squeezed:
        out = out.contiguous(memory_format=orig_mem_fmt)
    else:
        out = out.squeeze(0)

    # ---- label: fp32 on CPU (serialization-safe) ----
    dist_id = torch.full((B,), float(DISTORTION_ID), dtype=torch.float32)
    sev_ret = sev_b.detach().to(dtype=torch.float32).cpu()
    label = torch.stack([dist_id, sev_ret], dim=-1)
    if squeezed:
        label = label.squeeze(0)

    return out, label


# ==============================================================
# Registry API
# ==============================================================
def register(registry: Optional[dict] = None) -> dict:
    """Register in the provided (or shared/local) DISTORTION_REGISTRY."""
    reg = registry if registry is not None else (
        _SHARED_REGISTRY if _SHARED_REGISTRY is not None else _LOCAL_REGISTRY
    )
    reg[REGISTRY_ID] = distortion
    return reg


# [v3.1-2] Single call; resolves shared-vs-local internally.
register()


def get_distortion():
    return REGISTRY_ID, distortion
