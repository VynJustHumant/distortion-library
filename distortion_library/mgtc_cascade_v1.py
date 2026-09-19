"""
mgtc_cascade_v1 — Differentiable Multi-Generation Video Transcoding Cascade.
Distortion ID : 1007
Registry name : "mgtc_cascade_v1", version "1.9.2"

Deterministic: no random operations; `seed` / `generator` are NOT accepted.
Positional signature: (image, severity, value_range=(0,1), **kwargs).
Supports input shapes (C,H,W), (B,C,H,W), (B,T,C,H,W).
Returns (distorted, label) with label (2,) or (B,2).

Differentiability: for gradcheck / gradgradcheck construct the module with
use_ste=False and keep inputs in a narrow range (e.g. 0.4–0.6).
"""
import inspect
import math
import os
import threading
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from distortion_library.blur_atmospheric_turbulence_v1 import DISTORTION_REGISTRY
except Exception:
    try:
        from blur_atmospheric_turbulence_v1 import DISTORTION_REGISTRY
    except Exception:
        DISTORTION_REGISTRY: Dict[str, type] = {}


def register_distortion(name, version="1.0.0"):
    def deco(obj):
        obj.DISTORTION_NAME = name
        obj.VERSION = version
        DISTORTION_REGISTRY[name] = obj
        return obj
    return deco


DEFAULT_CALIBRATION: Dict[str, Tuple[float, float, float]] = {
    "qp":      (10.0, 45.0, 1.0),
    "delta_I": ( 2.0,  8.0, 0.8),
    "delta_P": ( 2.0,  8.0, 0.8),
    "tau":     ( 2.0,  0.4, 1.0),
    "N_eff":   ( 1.0,  4.0, 0.7),
}


def _calib_map(s, lo, hi, gamma):
    return lo + (hi - lo) * s.clamp_min(0.0).pow(gamma)


_STRUCT_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_BLUR_CACHE:   "OrderedDict[tuple, object]" = OrderedDict()
_STRUCT_MAX = 256
_BLUR_MAX   = 64
_CACHE_LOCK = threading.Lock()


def _cache_get_impl(cache, key, factory, max_size):
    with _CACHE_LOCK:
        v = cache.get(key)
        if v is not None:
            cache.move_to_end(key)
            return v
    v = factory()
    with _CACHE_LOCK:
        existing = cache.get(key)
        if existing is not None:
            return existing
        cache[key] = v
        while len(cache) > max_size:
            cache.popitem(last=False)
    return v


def _make_dct(K, device, dtype):
    n = torch.arange(K, device=device, dtype=dtype)
    k = n.view(-1, 1)
    A = torch.cos(math.pi * (2 * n + 1) * k / (2 * K))
    alpha = torch.full((K,), math.sqrt(2.0 / K), device=device, dtype=dtype)
    alpha[0] = math.sqrt(1.0 / K)
    return A * alpha.view(-1, 1)


def _get_dct(K, device, dtype):
    return _cache_get_impl(_STRUCT_CACHE, ("dct", K, str(device), str(dtype)),
                           lambda: _make_dct(K, device, dtype), _STRUCT_MAX)


def _make_grid(H, W, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return (xx, yy)


def _get_grid(H, W, device, dtype):
    return _cache_get_impl(_STRUCT_CACHE, ("grid", H, W, str(device), str(dtype)),
                           lambda: _make_grid(H, W, device, dtype), _STRUCT_MAX)


def _make_blur_kernel(sigma, device, dtype):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-t * t / (2 * sigma * sigma))
    k = k / k.sum().clamp_min(1e-8)
    return (k, r)


def _get_blur_kernel(sigma, device, dtype):
    s_rounded = round(float(sigma), 6)
    key = ("blur", s_rounded, str(device), str(dtype))
    return _cache_get_impl(_BLUR_CACHE, key,
                           lambda: _make_blur_kernel(s_rounded, device, dtype),
                           _BLUR_MAX)


def _gaussian_blur(x, sigma=0.4):
    if sigma <= 0:
        return x
    k, r = _get_blur_kernel(sigma, x.device, x.dtype)
    C = x.shape[1]
    kx = k.view(1, 1, 1, k.numel()).repeat(C, 1, 1, 1)
    ky = k.view(1, 1, k.numel(), 1).repeat(C, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, r), groups=C)
    return F.conv2d(x, ky, padding=(r, 0), groups=C)


def _block_dct(x, A, K):
    H, W = x.shape[-2:]
    pad_h, pad_w = (-H) % K, (-W) % K
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    Hp, Wp = x.shape[-2:]
    lead = x.shape[:-2]; nl = len(lead)
    x = x.reshape(*lead, Hp // K, K, Wp // K, K)
    x = x.permute(*range(nl), nl, nl + 2, nl + 1, nl + 3).contiguous()
    return A @ x @ A.transpose(-1, -2), (H, W, Hp, Wp)


def _block_idct(C, A, K, Hp, Wp, H, W):
    x = A.transpose(-1, -2) @ C @ A
    lead = x.shape[:-4]; nl = len(lead)
    bh, bw = x.shape[-4], x.shape[-3]
    x = x.permute(*range(nl), nl, nl + 2, nl + 1, nl + 3).contiguous()
    x = x.reshape(*lead, bh * K, bw * K)
    return x[..., :H, :W]


def _ste_quantize(x, delta, use_ste=True):
    d = delta.clamp_min(1e-8)
    qv = d * torch.round(x / d)
    if use_ste:
        return x + (qv - x).detach()
    return qv


def _estimate_flow(prev, curr, K, max_flow=8.0, eps=1e-6):
    if prev.dtype not in (torch.float32, torch.float64) or \
       curr.dtype not in (torch.float32, torch.float64):
        prev = prev.float()
        curr = curr.float()
    elif prev.dtype != curr.dtype:
        prev = prev.to(torch.float64)
        curr = curr.to(torch.float64)

    lum_p = prev.mean(dim=1, keepdim=True)
    lum_c = curr.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-0.5, 0.0, 0.5]], device=prev.device,
                      dtype=prev.dtype).view(1, 1, 1, 3)
    ky = kx.transpose(-1, -2).contiguous()
    Ix = F.conv2d(lum_p, kx, padding=(0, 1))
    Iy = F.conv2d(lum_p, ky, padding=(1, 0))
    It = lum_c - lum_p
    denom = (Ix * Ix + Iy * Iy).clamp_min(eps)
    flow = torch.cat([-Ix * It / denom, -Iy * It / denom], dim=1)
    flow = F.avg_pool2d(flow, K, K, ceil_mode=True)
    flow = F.interpolate(flow, size=prev.shape[-2:],
                         mode="bilinear", align_corners=False)
    return torch.tanh(flow / max_flow) * max_flow


def _warp(x, flow):
    assert flow.dim() == 4 and flow.shape[1] == 2, \
        f"_warp expects flow (B,2,H,W), got {tuple(flow.shape)}"
    B, C, H, W = x.shape
    assert flow.shape[0] == B and tuple(flow.shape[2:]) == (H, W), \
        f"_warp shape mismatch: x={tuple(x.shape)} flow={tuple(flow.shape)}"
    xx, yy = _get_grid(H, W, x.device, x.dtype)
    vx = flow[:, 0].to(xx.dtype)
    vy = flow[:, 1].to(xx.dtype)
    gx = 2.0 * (xx.unsqueeze(0) + vx) / max(W - 1, 1) - 1.0
    gy = 2.0 * (yy.unsqueeze(0) + vy) / max(H - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def _check_finite(t, tag):
    if not torch.isfinite(t).all():
        raise RuntimeError(f"mgtc_cascade_v1: non-finite tensor at {tag}")


@register_distortion("mgtc_cascade_v1", version="1.9.2")
class MGTCCascadeV1(nn.Module):
    DISTORTION_ID = 1007

    def __init__(self, calibration=None, block_size=8, i_period=8,
                 deblock_sigma=0.4, max_flow=8.0, debug=False,
                 use_ste=True):
        super().__init__()
        assert block_size % 2 == 0 and block_size >= 2, \
            "block_size must be even and >= 2"
        assert i_period >= 1, "i_period must be >= 1"
        assert deblock_sigma >= 0, \
            f"deblock_sigma must be >= 0 (got {deblock_sigma})"
        self.cal = dict(DEFAULT_CALIBRATION)
        if calibration:
            self.cal.update(calibration)
        lo_N, hi_N, _ = self.cal["N_eff"]
        assert lo_N >= 1.0, f"N_eff.lo must be >= 1.0 (got {lo_N})"
        assert hi_N >= 1.0, f"N_eff.hi must be >= 1.0 (got {hi_N})"

        self.K = int(block_size)
        self.i_period = int(i_period)
        self.deblock_sigma = float(deblock_sigma)
        self.max_flow = float(max_flow)
        self.use_ste = bool(use_ste)
        self.debug = bool(debug) or os.environ.get("MGTC_DEBUG") == "1"

    def _params(self, s):
        qp   = _calib_map(s, *self.cal["qp"])
        dI   = _calib_map(s, *self.cal["delta_I"])
        dP   = _calib_map(s, *self.cal["delta_P"])
        tau  = _calib_map(s, *self.cal["tau"])
        Neff = _calib_map(s, *self.cal["N_eff"])
        two  = qp.new_tensor(2.0)
        delta_I = torch.pow(two, (qp - dI - 4.0) / 6.0)
        delta_P = torch.pow(two, (qp + dP - 4.0) / 6.0)
        return {"qp": qp, "delta_I": delta_I, "delta_P": delta_P,
                "tau": tau, "Neff": Neff}

    def _one_generation(self, X, params):
        B, T, C, H, W = X.shape
        K = self.K
        A = _get_dct(K, X.device, X.dtype)
        dI_b = params["delta_I"].view(B, 1, 1, 1).to(X.dtype)
        dP_b = params["delta_P"].view(B, 1, 1, 1).to(X.dtype)

        frames = []
        prev_dec = None
        for t in range(T):
            Xt = X[:, t]
            is_I = (t % self.i_period) == 0
            if is_I or prev_dec is None:
                R, d, pred = Xt, dI_b, None
            else:
                flow = _estimate_flow(prev_dec, Xt, K, self.max_flow)
                pred = _warp(prev_dec, flow)
                R, d = Xt - pred, dP_b

            Cb, (H0, W0, Hp, Wp) = _block_dct(R, A, K)
            Cq = _ste_quantize(Cb, d.view(B, 1, 1, 1, 1, 1), use_ste=self.use_ste)
            R_hat = _block_idct(Cq, A, K, Hp, Wp, H0, W0)

            dec = R_hat if pred is None else (pred + R_hat)
            dec = _gaussian_blur(dec, sigma=self.deblock_sigma)
            dec = torch.clamp(dec, 1e-6, 1.0 - 1e-6)
            if self.debug:
                _check_finite(dec, f"gen t={t}")
            frames.append(dec)
            prev_dec = dec

        return torch.stack(frames, dim=1)

    def forward(self, X, s):
        if isinstance(s, torch.Tensor) and s.device != X.device:
            s = s.to(X.device)

        B = X.shape[0]
        if s.dim() == 0:
            s = s.reshape(1).expand(B)
        s = s.reshape(-1)
        if s.numel() == 1:
            s = s.expand(B)
        elif s.numel() != B:
            raise ValueError(
                f"severity has {s.numel()} values but batch is {B}")

        params = self._params(s)
        Nmax = self.N_max

        gens, Xn = [], X
        for _ in range(Nmax):
            Xn = self._one_generation(Xn, params)
            gens.append(Xn)

        center = (params["Neff"] - 1.0).view(B, 1).to(X.dtype)
        n_idx  = torch.arange(Nmax, device=X.device, dtype=X.dtype).view(1, -1)
        tau_b  = params["tau"].view(B, 1).clamp_min(1e-3).to(X.dtype)
        logits = -((n_idx - center) / tau_b).pow(2)
        w = torch.softmax(logits, dim=-1).view(B, Nmax, 1, 1, 1, 1)

        stacked = torch.stack(gens, dim=1)
        return (stacked * w).sum(dim=1)

    @property
    def N_max(self):
        lo, hi, _ = self.cal["N_eff"]
        return max(1, int(math.ceil(max(hi, lo))))


_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

_ALLOWED_KWARGS = frozenset(
    p for p in inspect.signature(MGTCCascadeV1.__init__).parameters
    if p != "self"
)


def mgtc_cascade_v1(image, severity, value_range=(0.0, 1.0), **kwargs):
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    unknown = set(kwargs) - _ALLOWED_KWARGS
    if unknown:
        raise TypeError(
            f"mgtc_cascade_v1 got unexpected keyword argument(s): "
            f"{sorted(unknown)}. Supported: {sorted(_ALLOWED_KWARGS)}. "
            f"This deterministic module does NOT accept `seed` or `generator`."
        )

    orig_dtype, orig_dim, dev = image.dtype, image.dim(), image.device
    vmin, vmax = value_range

    if "i_period" in kwargs and int(kwargs["i_period"]) < 1:
        raise ValueError(f"i_period must be >= 1, got {kwargs['i_period']}")
    if "deblock_sigma" in kwargs and float(kwargs["deblock_sigma"]) < 0:
        raise ValueError(
            f"deblock_sigma must be >= 0, got {kwargs['deblock_sigma']}")

    if isinstance(severity, (int, float)):
        s_t = torch.tensor(float(severity), device=dev)
    elif isinstance(severity, torch.Tensor):
        s_t = severity.to(dev)
    else:
        s_t = torch.as_tensor(severity, device=dev)
    s_t = torch.clamp(s_t.float(), 0.01, 1.0)

    channels_last = channels_last_3d = False
    if orig_dim == 3:
        x = image.unsqueeze(0).unsqueeze(0)
        is_single, squeeze = True, (0, 1)
    elif orig_dim == 4:
        x = image.unsqueeze(1)
        is_single, squeeze = False, (1,)
        channels_last = image.is_contiguous(memory_format=torch.channels_last)
    elif orig_dim == 5:
        x = image
        is_single, squeeze = False, ()
        if hasattr(torch, "channels_last_3d"):
            try:
                channels_last_3d = image.is_contiguous(
                    memory_format=torch.channels_last_3d)
            except Exception:
                channels_last_3d = False
    else:
        raise ValueError(f"Unsupported input dims {orig_dim}")

    x = x.float()
    x01 = torch.clamp((x - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    B = x01.shape[0]

    if s_t.dim() == 0:
        s_flat = s_t.reshape(1).expand(B)
    else:
        s_flat = s_t.reshape(-1)
        if s_flat.numel() == 1:
            s_flat = s_flat.expand(B)
        elif s_flat.numel() != B:
            raise ValueError(
                f"severity has {s_flat.numel()} values but batch is {B}")

    module = MGTCCascadeV1(**kwargs)
    module = module.to(device=dev).eval()

    X_s = module(x01, s_flat)
    s_blend = s_flat.view(B, 1, 1, 1, 1)
    Y = torch.clamp((1.0 - s_blend) * x01 + s_blend * X_s, 0.0, 1.0)
    Y = torch.clamp(Y * (vmax - vmin) + vmin, vmin, vmax)
    Y = torch.where(torch.isfinite(Y), Y, torch.zeros_like(Y))

    if orig_dtype in _INT_DTYPES:
        Y = torch.round(Y)
    Y = Y.to(orig_dtype)

    for d in sorted(squeeze, reverse=True):
        Y = Y.squeeze(d)

    if channels_last and Y.dim() == 4:
        Y = Y.contiguous(memory_format=torch.channels_last)
    if channels_last_3d and Y.dim() == 5:
        Y = Y.contiguous(memory_format=torch.channels_last_3d)

    if is_single:
        s_val = float(s_flat.detach().reshape(-1)[0])
        label = torch.tensor([float(MGTCCascadeV1.DISTORTION_ID), s_val],
                             dtype=torch.float32, device=dev)
    else:
        s_vals = s_flat.detach().reshape(-1).float()
        id_col = torch.full((s_vals.shape[0],),
                            float(MGTCCascadeV1.DISTORTION_ID),
                            dtype=torch.float32, device=dev)
        label = torch.stack([id_col, s_vals], dim=-1)

    return Y, label


def _cache_snapshot():
    with _CACHE_LOCK:
        return (list(_STRUCT_CACHE.keys()), list(_BLUR_CACHE.keys()))


__all__ = [
    "mgtc_cascade_v1", "MGTCCascadeV1",
    "DISTORTION_REGISTRY", "DEFAULT_CALIBRATION",
    "_cache_snapshot",
]
