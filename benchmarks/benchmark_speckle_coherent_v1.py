"""Throughput benchmark for speckle_coherent_v1.

Table columns:
    variant   : distortion configuration
    rng-mode  : `gen` (shared stateful generator) or `seed=0` (per-sample)
    ms/batch  : wall-clock per batch
    img/s     : images per second

Environment metadata (torch/cuda/device) is printed once at the top so
results are attributable to a specific stack.

Usage:
    python benchmarks/benchmark_speckle_coherent_v1.py --sizes 256 512 1024
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from distortion_library.speckle_coherent_v1 import distortion  # noqa: E402


_W_VARIANT = 18
_W_RNG     = 10
_W_MS      = 12
_W_IPS     = 12
_SEP       = 2
_N_SEPS    = 3


def _header_width() -> int:
    return _W_VARIANT + _W_RNG + _W_MS + _W_IPS + _N_SEPS * _SEP


def _hrule(char: str = "-") -> str:
    return char * _header_width()


def _header() -> str:
    return (
        f"{'variant':<{_W_VARIANT}}  "
        f"{'rng-mode':<{_W_RNG}}  "
        f"{'ms/batch':>{_W_MS}}  "
        f"{'img/s':>{_W_IPS}}"
    )


def _row(variant: str, rng_mode: str, ms: float, img_s: float) -> str:
    return (
        f"{variant:<{_W_VARIANT}}  "
        f"{rng_mode:<{_W_RNG}}  "
        f"{ms:>{_W_MS}.3f}  "
        f"{img_s:>{_W_IPS}.1f}"
    )


def _environment_banner() -> str:
    """Human-readable stack description for attributing benchmark numbers."""
    lines = [
        f"python      : {platform.python_version()}  ({platform.system()} {platform.machine()})",
        f"torch       : {torch.__version__}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda runtime: {torch.version.cuda}")
        lines.append(f"cudnn       : {torch.backends.cudnn.version()}")
        try:
            lines.append(f"gpu[0]      : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            lines.append(f"gpu[0] smem : {props.total_memory / 2**30:.1f} GiB")
        except Exception:
            pass
    else:
        lines.append("device      : cpu (no CUDA available)")
    return "\n".join(lines)


def bench(fn, iters: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run(size: int, B: int = 8, C: int = 3) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtypes = [torch.float32]
    if device == "cuda":
        dtypes += [torch.float16, torch.bfloat16]

    variants = [
        ("iid-lognormal",    dict()),
        ("exact-gamma",      dict(exact_gamma=True)),
        ("correlated",       dict(rho=1.5)),
        ("correlated-indep", dict(rho=1.5, channel_independent=True)),
        ("iid-indep",        dict(channel_independent=True)),
    ]

    for dt in dtypes:
        print(f"\n=== size={size}  B={B}  C={C}  device={device}  dtype={dt} ===")
        print(_header())
        print(_hrule())

        x = torch.rand(B, C, size, size, device=device, dtype=dt)
        s = torch.full((B,), 0.5, device=device, dtype=dt)

        for name, kw in variants:
            g = torch.Generator(device=device).manual_seed(0)
            t_gen = bench(lambda: distortion(x, s, generator=g, **kw))
            print(_row(name, "gen", t_gen * 1e3, B / t_gen))

            t_seed = bench(lambda: distortion(x, s, seed=0, **kw), iters=20)
            print(_row(name, "seed=0", t_seed * 1e3, B / t_seed))

    print("\nNotes:")
    print("  - gen mode : single shared stateful generator, compile-friendly.")
    print("  - seed=0   : per-sample deterministic generators (worst case).")
    print("  - *-indep  : channel_independent=True, C independent noise fields.")
    print("  - elementwise paths scale ~O(H·W); rho adds O(k²·H·W).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--channels", type=int, default=3)
    args = ap.parse_args()

    print("=" * _header_width())
    print("speckle_coherent_v1 benchmark")
    print("=" * _header_width())
    print(_environment_banner())
    print("=" * _header_width())

    for sz in args.sizes:
        run(sz, args.batch, args.channels)


if __name__ == "__main__":
    main()
