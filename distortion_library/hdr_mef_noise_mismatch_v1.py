"""HDR Multi-Exposure Fusion Noise Mismatch distortion (PyTorch).

Registry ID   : hdr_mef_noise_mismatch_v1
Version       : 1.1.7
Distortion ID : 1009  (fp32-safe)
ISP stage     : sensor / fusion (before tone / gamma / compression)

Distributional notes
--------------------
For bracket ``k`` the pre-fusion per-pixel noise ``n_k`` is Gaussian with
mean 0 and variance

    sigma_k^2(L) = kappa * L / g_k + (sigma_r^2 + sigma_q^2) / g_k^2

After unit-variance Gaussian smoothing (population std, correction=0) the
marginal remains Gaussian with a shaped power spectrum. Fused noise
``n_f = sum_k W_k * sigma_k * eps_k`` is a weighted mixture of K Gaussians
and is exactly Gaussian iff ``W_k * sigma_k`` is constant in ``k``; in
general slightly leptokurtic. Temporal AR(1) preserves Gaussianity.
"""
from __future__ import annotations

import math
import threading
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    NumLike = Union[
        float, int,
        "np.floating", "np.integer", "np.ndarray",
        list, tuple,
        torch.Tensor,
    ]
else:
    NumLike = Union[float, int, list, tuple, torch.Tensor]


try:
    from distortion_library.blur_atmospheric_turbulence_v1 import DISTORTION_REGISTRY
except Exception:
    try:
        from blur_atmospheric_turbulence_v1 import DISTORTION_REGISTRY
    except Exception:
        DISTORTION_REGISTRY: Dict[str, object] = {}


def register_distortion(name: str, version: str = "1.1.7"):
    """Decorator: register a distortion callable in the shared registry.

    Module-internal machinery; intentionally not re-exported via ``__all__``.
    """
    def deco(fn):
        DISTORTION_REGISTRY[name] = fn
        fn.DISTORTION_NAME = name
        fn.DISTORTION_VERSION = version
        return fn
    return deco


DISTORTION_NAME = "hdr_mef_noise_mismatch_v1"
DISTORTION_VERSION = "1.1.7"
DISTORTION_ID = 1009


_ALLOWED_KWARGS = frozenset({
    "K", "ev_min", "ev_max", "L_max",
    "kappa", "sigma_r", "sigma_q",
    "tau", "lam", "sigma_s",
    "rho_t", "rho_f", "gamma",
    "compute_dtype",
})


try:
    import numpy as _np
    _NUMERIC_TYPES: tuple = (int, float, _np.integer, _np.floating)
except ImportError:
    _np = None
    _NUMERIC_TYPES = (int, float)


def _num(kwargs: dict, key: str, default: Any, *, as_int: bool = False) -> Union[int, float]:
    """Strict numeric coercion for hyperparameters."""
    v = kwargs.get(key, default)
    if isinstance(v, bool):
        raise TypeError(f"{key} must be numeric, got bool")
    if not isinstance(v, _NUMERIC_TYPES):
        raise TypeError(
            f"{key} must be a real number, got {v!r} ({type(v).__name__})"
        )
    if as_int:
        if float(v).is_integer():
            return int(v)
        raise TypeError(f"{key} must be an integer, got {v!r}")
    return float(v)


class _KernelLRU:
    """LRU cache for separable 1-D Gaussian kernels."""

    def __init__(self, max_size: int = 128):
        self._max = int(max_size)
        self._lock = threading.Lock()
        self._store: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    @property
    def max_size(self) -> int:
        return self._max

    def get(self, key: tuple) -> Optional[torch.Tensor]:
        with self._lock:
            v = self._store.get(key)
            if v is not None:
                self._store.move_to_end(key)
            return v

    def put(self, key: tuple, value: torch.Tensor) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


_KERNEL_CACHE = _KernelLRU(max_size=128)


_NO_INDEX = -1


def _device_key(device) -> Tuple[str, int]:
    """Normalize ``torch.device`` for cache keying.

    - Resolved ``"cuda:N"`` -> ``("cuda", N)``
    - Unresolved ``"cuda"`` -> ``("cuda", torch.cuda.current_device())``
    - Unresolved ``"cuda"`` with CUDA unavailable -> ``("cuda", 0)``
    - Devices without an index (``cpu``, ``meta``, ...) -> ``(type, _NO_INDEX)``
    """
    d = torch.device(device)
    if d.index is not None:
        return (d.type, d.index)
    if d.type == "cuda":
        try:
            return (d.type, int(torch.cuda.current_device()))
        except Exception:
            return (d.type, 0)
    return (d.type, _NO_INDEX)


def _get_gauss_kernel(sigma: float, dtype: torch.dtype, device) -> torch.Tensor:
    key = (round(float(sigma), 6), str(dtype), _device_key(device))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    if sigma <= 1e-6:
        k = torch.ones(1, dtype=dtype, device=device)
    else:
        radius = max(1, int(math.ceil(3.0 * float(sigma))))
        x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
        k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
        k = k / k.sum().clamp_min(1e-12)
    _KERNEL_CACHE.put(key, k)
    return k


def _pad_reflect_safe(x: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    if pad <= 0:
        return x
    size = x.shape[dim]
    if size <= 1 or pad >= size:
        mode = "replicate"
    else:
        mode = "reflect"
    if dim == 2:
        pad_arg = (0, 0, pad, pad)
    elif dim == 3:
        pad_arg = (pad, pad, 0, 0)
    else:
        return x
    try:
        return F.pad(x, pad_arg, mode=mode)
    except Exception:
        return F.pad(x, pad_arg, mode="constant", value=0.0)


def _sep_gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-6:
        return x
    k = _get_gauss_kernel(sigma, x.dtype, x.device)
    r = k.numel() // 2
    if r == 0:
        return x
    _, C, _, _ = x.shape
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()
    x = _pad_reflect_safe(x, r, dim=3)
    x = F.conv2d(x, kx, groups=C)
    x = _pad_reflect_safe(x, r, dim=2)
    x = F.conv2d(x, ky, groups=C)
    return x


def _srgb_to_linear(u: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF: signal u in [0,1] -> linear radiance L."""
    a = 0.04045
    lo = u / 12.92
    hi = ((u.clamp_min(a) + 0.055) / 1.055).clamp_min(0.0).pow(2.4)
    return torch.where(u <= a, lo, hi)


def _srgb_from_linear(L: torch.Tensor) -> torch.Tensor:
    """sRGB OETF: linear radiance L -> signal u in [0,1]."""
    L = L.clamp_min(0.0)
    a = 0.0031308
    lo = 12.92 * L
    hi = 1.055 * L.clamp_min(a).pow(1.0 / 2.4) - 0.055
    return torch.where(L <= a, lo, hi).clamp(0.0, 1.0)


def _mix_seed(base_seed: int, dist_id: int, b: int, k: int) -> int:
    x = (int(base_seed) & 0xFFFFFFFF) ^ ((int(dist_id) & 0xFFFFFFFF) * 0x9E3779B1 & 0xFFFFFFFF)
    x = (x ^ ((b + 1) * 0x85EBCA6B & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x ^ ((k + 1) * 0xC2B2AE35 & 0xFFFFFFFF)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x & 0x7FFFFFFF


def _validate_kwargs(kwargs: dict) -> None:
    unknown = set(kwargs.keys()) - _ALLOWED_KWARGS
    if unknown:
        raise TypeError(
            f"hdr_mef_noise_mismatch_v1 got unexpected keyword argument(s): "
            f"{sorted(unknown)}. Allowed: {sorted(_ALLOWED_KWARGS)}"
        )


def _validate_hparams(*, K, rho_t, rho_f, lam, L_max, sigma_s, tau,
                      kappa, gamma, sigma_r, sigma_q, ev_min, ev_max) -> None:
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if not (0.0 <= rho_t <= 1.0):
        raise ValueError(f"rho_t must be in [0, 1], got {rho_t}")
    if not (0.0 <= rho_f <= 1.0):
        raise ValueError(f"rho_f must be in [0, 1], got {rho_f}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if L_max <= 0.0:
        raise ValueError(f"L_max must be > 0, got {L_max}")
    if sigma_s < 0.0:
        raise ValueError(f"sigma_s must be >= 0, got {sigma_s}")
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if kappa < 0.0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    if sigma_r < 0.0:
        raise ValueError(f"sigma_r must be >= 0, got {sigma_r}")
    if sigma_q < 0.0:
        raise ValueError(f"sigma_q must be >= 0, got {sigma_q}")
    if ev_min > ev_max:
        raise ValueError(f"ev_min ({ev_min}) must be <= ev_max ({ev_max})")


_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _resolve_compute_dtype(image: torch.Tensor,
                           compute_dtype: Optional[torch.dtype]) -> torch.dtype:
    if compute_dtype is not None:
        if not isinstance(compute_dtype, torch.dtype):
            raise TypeError(
                f"compute_dtype must be a torch.dtype or None, "
                f"got {type(compute_dtype).__name__}"
            )
        if compute_dtype not in _FLOAT_DTYPES:
            raise ValueError(
                f"compute_dtype must be a floating-point dtype, "
                f"got {compute_dtype}"
            )
        return compute_dtype
    return torch.float64 if image.dtype == torch.float64 else torch.float32


@register_distortion(DISTORTION_NAME, version=DISTORTION_VERSION)
def hdr_mef_noise_mismatch_v1(
    image: torch.Tensor,
    severity: NumLike,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    value_range: Tuple[float, float] = (0.0, 1.0),
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HDR Multi-Exposure Fusion Noise Mismatch distortion."""
    _validate_kwargs(kwargs)

    K = _num(kwargs, "K", 3, as_int=True)
    ev_min = _num(kwargs, "ev_min", -2.0)
    ev_max = _num(kwargs, "ev_max", 2.0)
    L_max = _num(kwargs, "L_max", 4.0)
    kappa = _num(kwargs, "kappa", 1.0)
    sigma_r = _num(kwargs, "sigma_r", 0.01)
    sigma_q = _num(kwargs, "sigma_q", 0.005)
    tau = _num(kwargs, "tau", 0.2)
    lam = _num(kwargs, "lam", 0.5)
    sigma_s = _num(kwargs, "sigma_s", 1.0)
    rho_t = _num(kwargs, "rho_t", 0.7)
    rho_f = _num(kwargs, "rho_f", 0.2)
    gamma = _num(kwargs, "gamma", 1.0)

    cdt = _resolve_compute_dtype(image, kwargs.get("compute_dtype", None))

    _validate_hparams(K=K, rho_t=rho_t, rho_f=rho_f, lam=lam,
                      L_max=L_max, sigma_s=sigma_s, tau=tau,
                      kappa=kappa, gamma=gamma,
                      sigma_r=sigma_r, sigma_q=sigma_q,
                      ev_min=ev_min, ev_max=ev_max)

    orig_dtype = image.dtype

    orig_mem_fmt: Optional[torch.memory_format] = None
    if image.dim() == 4 and image.is_contiguous(memory_format=torch.channels_last):
        orig_mem_fmt = torch.channels_last
    elif image.dim() == 5 and hasattr(torch, "channels_last_3d") and \
            image.is_contiguous(memory_format=torch.channels_last_3d):
        orig_mem_fmt = torch.channels_last_3d

    x = image
    is_single = image.dim() == 3
    if is_single:
        x = x.unsqueeze(0)
    is_video = x.dim() == 5
    if not is_video and x.dim() != 4:
        raise ValueError(f"Unsupported input ndim={image.dim()} shape={tuple(image.shape)}")

    x = x.to(cdt)
    if is_video:
        B, T, C, H, W = x.shape
    else:
        B, C, H, W = x.shape
        T = 1
        # Promote to (B, T=1, C, H, W) so the pipeline operates on a
        # uniform 5D shape. The synthetic T=1 dim is removed before return.
        x = x.unsqueeze(1)

    if sigma_s > 0.0:
        r_est = int(math.ceil(3.0 * sigma_s))
        limit = min(H, W) / 6.0
        if sigma_s > limit:
            warnings.warn(
                f"sigma_s={sigma_s:.3f} on {H}x{W} image -> {2*r_est+1}-pixel "
                f"separable kernel; conv2d cost grows as O(sigma_s^2). "
                f"Consider sigma_s <= {limit:.3f}.",
                RuntimeWarning, stacklevel=2,
            )

    v0, v1 = float(value_range[0]), float(value_range[1])
    lo, hi = (v0, v1) if v0 <= v1 else (v1, v0)
    span = float(hi - lo) if hi > lo else 1.0

    u = ((x - lo) / span).clamp(0.0, 1.0)
    L = _srgb_to_linear(u)

    s = torch.as_tensor(severity, dtype=cdt, device=x.device)
    if s.numel() == 1:
        s = s.reshape(1).expand(B).contiguous()
    else:
        s = s.reshape(-1)
        if s.numel() != B:
            raise ValueError(f"severity must be scalar or length {B}, got {s.numel()}")
    s = s.clamp(0.01, 1.0)
    alpha = ((s - 0.01) / 0.99).clamp_min(0.0).pow(gamma)
    alpha_b = alpha.view(B, 1, 1, 1, 1)

    if seed is None:
        if generator is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, dtype=torch.int64).item())
        else:
            seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    base_seed = int(seed) & 0x7FFFFFFF

    if K == 1:
        evs = torch.tensor([ev_min], dtype=cdt, device=x.device)
    else:
        evs = torch.linspace(ev_min, ev_max, K, dtype=cdt, device=x.device)
    gains = torch.pow(2.0, evs).view(1, 1, K, 1, 1, 1)

    L_sat = (L_max / gains).clamp_min(1e-8)
    L_e = L.unsqueeze(2)
    L_k = torch.minimum(L_e, L_sat)

    var_k = (kappa * L_k / gains + (sigma_r ** 2 + sigma_q ** 2) / (gains ** 2))
    var_k = var_k.clamp_min(1e-8)
    sigma_k = var_k.sqrt()

    device = x.device
    eps = torch.empty((B, T, K, C, H, W), dtype=cdt, device=device)
    fpn = torch.empty((B, 1, K, C, H, W), dtype=cdt, device=device)

    for b in range(B):
        seed_b = _mix_seed(base_seed, DISTORTION_ID, b, 0)
        g = torch.Generator(device=device)
        g.manual_seed(seed_b)

        xi = torch.randn((T, K, C, H, W), generator=g, dtype=cdt, device=device)
        xi_b = _sep_gaussian_blur(xi.reshape(T * K, C, H, W), sigma_s)
        xi_b = xi_b / xi_b.std(correction=0).clamp_min(1e-8)
        eps[b] = xi_b.reshape(T, K, C, H, W)

        f = torch.randn((1, K, C, H, W), generator=g, dtype=cdt, device=device)
        f_b = _sep_gaussian_blur(f.reshape(K, C, H, W), sigma_s).reshape(1, K, C, H, W)
        f_b = f_b / f_b.std(correction=0).clamp_min(1e-8)
        fpn[b] = f_b

    if T > 1 and rho_t > 0.0:
        c_t = math.sqrt(max(1.0 - rho_t * rho_t, 0.0))
        eps_ar = torch.empty_like(eps)
        eps_ar[:, 0] = eps[:, 0]
        for t in range(1, T):
            eps_ar[:, t] = rho_t * eps_ar[:, t - 1] + c_t * eps[:, t]
        eps = eps_ar

    c_f = math.sqrt(max(1.0 - rho_f * rho_f, 0.0))
    n_k = sigma_k * (c_f * eps + rho_f * fpn)

    center = (L_max * 0.5) / gains
    w_naive = torch.exp(-((L_k - center) ** 2) / (2.0 * tau * tau))
    W_naive = w_naive / w_naive.sum(dim=2, keepdim=True).clamp_min(1e-8)

    inv_var = 1.0 / var_k
    W_opt = inv_var / inv_var.sum(dim=2, keepdim=True).clamp_min(1e-8)

    W_mix = (1.0 - lam) * W_naive + lam * W_opt

    n_f = (W_mix * n_k).sum(dim=2)
    L_prime = (L + n_f).clamp(0.0, L_max)

    u_out = _srgb_from_linear(L_prime)
    x_out = u_out * span + lo

    x_final = (1.0 - alpha_b) * x + alpha_b * x_out
    x_final = x_final.clamp(lo, hi)

    fallback = torch.full_like(x_final, (lo + hi) * 0.5)
    x_final = torch.where(torch.isfinite(x_final), x_final, fallback)

    y = x_final.to(orig_dtype)  # always 5D (B, T, C, H, W) at this point
    if not is_video:
        y = y.squeeze(1)         # remove the synthetic T=1
    if is_single:
        y = y.squeeze(0)         # remove the synthetic B=1
    if orig_mem_fmt is not None:
        try:
            y = y.contiguous(memory_format=orig_mem_fmt)
        except Exception:
            y = y.contiguous()
    else:
        y = y.contiguous()

    s_cpu = s.detach().cpu().to(torch.float32)
    if is_single:
        label = torch.tensor([float(DISTORTION_ID), float(s_cpu[0].item())], dtype=torch.float32)
    else:
        ids = torch.full((B,), float(DISTORTION_ID), dtype=torch.float32)
        label = torch.stack([ids, s_cpu], dim=-1)
    label = label.detach().cpu()

    return y, label


distortion = hdr_mef_noise_mismatch_v1


class HdrMefNoiseMismatchV1(nn.Module):
    """Layer wrapper around :func:`hdr_mef_noise_mismatch_v1`."""

    DISTORTION_NAME = DISTORTION_NAME
    DISTORTION_VERSION = DISTORTION_VERSION
    DISTORTION_ID = DISTORTION_ID

    def __init__(
        self,
        severity: NumLike = 0.5,
        seed: Optional[int] = None,
        value_range: Tuple[float, float] = (0.0, 1.0),
        K: int = 3,
        ev_min: float = -2.0,
        ev_max: float = 2.0,
        L_max: float = 4.0,
        kappa: float = 1.0,
        sigma_r: float = 0.01,
        sigma_q: float = 0.005,
        tau: float = 0.2,
        lam: float = 0.5,
        sigma_s: float = 1.0,
        rho_t: float = 0.7,
        rho_f: float = 0.2,
        gamma: float = 1.0,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.severity = severity
        self.seed = seed
        self.value_range = tuple(value_range)
        self.K = int(K)
        self.ev_min = float(ev_min)
        self.ev_max = float(ev_max)
        self.L_max = float(L_max)
        self.kappa = float(kappa)
        self.sigma_r = float(sigma_r)
        self.sigma_q = float(sigma_q)
        self.tau = float(tau)
        self.lam = float(lam)
        self.sigma_s = float(sigma_s)
        self.rho_t = float(rho_t)
        self.rho_f = float(rho_f)
        self.gamma = float(gamma)
        self.compute_dtype = compute_dtype

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return hdr_mef_noise_mismatch_v1(
            image,
            severity=self.severity,
            seed=self.seed,
            value_range=self.value_range,
            K=self.K,
            ev_min=self.ev_min,
            ev_max=self.ev_max,
            L_max=self.L_max,
            kappa=self.kappa,
            sigma_r=self.sigma_r,
            sigma_q=self.sigma_q,
            tau=self.tau,
            lam=self.lam,
            sigma_s=self.sigma_s,
            rho_t=self.rho_t,
            rho_f=self.rho_f,
            gamma=self.gamma,
            compute_dtype=self.compute_dtype,
        )


__all__ = [
    "hdr_mef_noise_mismatch_v1",
    "distortion",
    "HdrMefNoiseMismatchV1",
    "DISTORTION_NAME",
    "DISTORTION_VERSION",
    "DISTORTION_ID",
    "DISTORTION_REGISTRY",
]
