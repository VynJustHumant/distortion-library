"""
Throughput benchmark for compression_recompression_v1.

Usage
-----
    python benchmarks/bench_compression_recompression_v1.py                # auto device
    python benchmarks/bench_compression_recompression_v1.py --device cuda
    python benchmarks/bench_compression_recompression_v1.py --size 1024 --batch 8
"""
from __future__ import annotations

# ---------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _HERE.parents if (p / "distortion_library").is_dir()),
    _HERE.parent.parent,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------
import argparse
import time

import torch

from distortion_library.compression_recompression_v1 import distortion  # [FIX]


# ---------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:                   # [FIX]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> float:                     # [FIX]
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


def _time_fn(fn, device: torch.device, n_warmup: int = 5,
             n_iter: int = 30) -> float:
    for _ in range(n_warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n_iter


def bench_one(h: int, w: int, B: int, device: torch.device,
              dtype: torch.dtype, backward: bool = False):
    x = torch.rand(B, 3, h, w, device=device, dtype=dtype)
    s = torch.full((B,), 0.5, device=device, dtype=dtype)
    if backward:
        x.requires_grad_(True)

    def step():
        if backward and x.grad is not None:
            x.grad.zero_()                                       # [FIX] in-place
        y, _ = distortion(x, s)
        if backward:
            y.sum().backward()

    _reset_peak(device)                                          # [FIX]
    t = _time_fn(step, device)
    return t, B / t, _peak_mb(device)                            # [FIX] 3-tuple


# ---------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------
def bench_scaling(device, dtype, sizes=(256, 512, 1024), B=4):
    print(f"\n=== Spatial scaling (B={B}, dtype={dtype}, device={device}) ===")
    header = f"{'H×W':>10} | {'fwd ms':>10} | {'img/s':>10} | {'Mpix/s':>10} | {'peak MB':>10} | {'rel':>6}"
    print(header)
    base = None
    for sz in sizes:
        t, ips, peak = bench_one(sz, sz, B, device, dtype)
        mpix = B * 3 * sz * sz / t / 1e6
        if base is None:
            base = t
        print(f"{sz:>4}×{sz:<4} | {t*1e3:>10.2f} | {ips:>10.1f} | "
              f"{mpix:>10.1f} | {peak:>10.1f} | {t/base:>5.2f}×")


def bench_batch(device, dtype, size=512, batches=(1, 2, 4, 8, 16)):
    print(f"\n=== Batch scaling ({size}×{size}, dtype={dtype}) ===")
    print(f"{'B':>4} | {'fwd ms':>10} | {'img/s':>10} | {'peak MB':>10} | {'rel':>6}")
    base = None
    for B in batches:
        try:
            t, ips, peak = bench_one(size, size, B, device, dtype)
            if base is None:
                base = t
            print(f"{B:>4} | {t*1e3:>10.2f} | {ips:>10.1f} | "
                  f"{peak:>10.1f} | {t/base:>5.2f}×")
        except RuntimeError as e:
            print(f"{B:>4} | OOM / error: {str(e)[:60]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()


def bench_dtype(device, size=512, B=4):
    print(f"\n=== Dtype ({size}×{size}, B={B}, device={device}) ===")
    dtypes = [torch.float32]
    if device.type == "cuda":
        dtypes = [torch.float16, torch.float32, torch.bfloat16]
    for dt in dtypes:
        try:
            t, ips, peak = bench_one(size, size, B, device, dt)
            print(f"{str(dt):>18} | {t*1e3:>8.2f} ms | {ips:>8.1f} img/s | "
                  f"peak {peak:>6.1f} MB")
        except RuntimeError as e:
            print(f"{str(dt):>18} | err: {str(e)[:60]}")


def bench_dtype_accuracy(device, size=256, B=2):                 # [FIX]
    """Compare fp16 / bf16 output against fp32 reference."""
    print(f"\n=== Dtype accuracy ({size}×{size}, B={B}) ===")
    x32 = torch.rand(B, 3, size, size, device=device, dtype=torch.float32)
    s32 = torch.full((B,), 0.5, device=device, dtype=torch.float32)
    y32, _ = distortion(x32, s32)

    for dt in (torch.float16, torch.bfloat16):
        xd = x32.to(dt)
        sd = s32.to(dt)
        yd, _ = distortion(xd, sd)
        err = (yd.to(torch.float32) - y32).abs().max().item()
        print(f"{str(dt):>18} | max|y_dtype - y_fp32| = {err:.4e}")


def bench_backward(device, dtype, size=256, B=2):
    print(f"\n=== Forward vs. fwd+bwd ({size}×{size}, B={B}, dtype={dtype}) ===")
    t_fwd, ips_fwd, _ = bench_one(size, size, B, device, dtype, backward=False)
    t_bwd, ips_bwd, peak_bwd = bench_one(size, size, B, device, dtype, backward=True)
    print(f"  forward  : {t_fwd*1e3:>8.2f} ms | {ips_fwd:>7.1f} img/s")
    print(f"  fwd+bwd  : {t_bwd*1e3:>8.2f} ms | {ips_bwd:>7.1f} img/s "
          f"(ratio {t_bwd/t_fwd:.2f}×)  peak {peak_bwd:.1f} MB")


def bench_nmax(device, dtype, size=256, B=2, nmaxs=(1, 2, 4, 8)):
    """Validate O(N_max) linear scaling in the number of JPEG generations."""
    print(f"\n=== N_max scaling ({size}×{size}, B={B}, dtype={dtype}) ===")
    print(f"{'N_max':>6} | {'fwd ms':>10} | {'rel':>6}  (expect ~linear)")
    x = torch.rand(B, 3, size, size, device=device, dtype=dtype)
    s = torch.full((B,), 0.5, device=device, dtype=dtype)
    base = None
    for nm in nmaxs:
        def step(nm=nm):
            distortion(x, s, N_max=nm)
        t = _time_fn(step, device)
        if base is None:
            base = t
        print(f"{nm:>6} | {t*1e3:>10.2f} | {t/base:>5.2f}×")


# ---------------------------------------------------------------
# entry
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None,
                        help="'cuda', 'cpu', or auto")
    parser.add_argument("--dtype", default="float32",
                        choices=["float16", "float32", "float64", "bfloat16"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    dtype = {
        "float16":  torch.float16,
        "float32":  torch.float32,
        "float64":  torch.float64,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    print(f"torch={torch.__version__}  device={device}  dtype={dtype}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    bench_scaling(device, dtype)
    bench_batch(device, dtype, size=args.size)
    bench_dtype(device, args.size, args.batch)
    if device.type == "cuda":
        bench_dtype_accuracy(device)                             # [FIX]
    bench_backward(device, dtype)
    bench_nmax(device, dtype)


if __name__ == "__main__":
    main()
