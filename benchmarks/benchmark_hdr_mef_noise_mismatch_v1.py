"""Throughput benchmark for hdr_mef_noise_mismatch_v1."""
from __future__ import annotations

import argparse
import platform
import time

import torch

from distortion_library.hdr_mef_noise_mismatch_v1 import (
    hdr_mef_noise_mismatch_v1,
    DISTORTION_VERSION,
)


def _environment_banner() -> str:
    cudnn_v = torch.backends.cudnn.version()
    lines = [
        f"distortion   : hdr_mef_noise_mismatch_v1 v{DISTORTION_VERSION}",
        f"python       : {platform.python_version()} ({platform.platform()})",
        f"torch        : {torch.__version__}",
        f"cuda runtime : {torch.version.cuda}",
        f"cudnn        : {cudnn_v if cudnn_v is not None else 'n/a'}",
        f"cuda avail   : {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            lines += [
                f"device       : {props.name}",
                f"cc           : sm_{props.major}{props.minor}",
                f"smem/block   : {props.shared_memory_per_block // 1024} KiB",
                f"total mem    : {props.total_memory // (1024**3)} GiB",
            ]
        except Exception:
            pass
    return "\n".join(lines)


def bench(x: torch.Tensor,
          severity: float = 0.5,
          n_warmup: int = 3,
          n_iter: int = 10,
          seed: int = 0) -> float:
    for _ in range(n_warmup):
        hdr_mef_noise_mismatch_v1(x, severity=severity, seed=seed)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        hdr_mef_noise_mismatch_v1(x, severity=severity, seed=seed)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--resolutions", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--video-T", type=int, default=0)
    ap.add_argument("--iter", type=int, default=10)
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(_environment_banner())
    print(f"device={dev}  iter={args.iter}\n")

    for B in args.batches:
        for R in args.resolutions:
            if args.video_T > 0:
                x = torch.rand(B, args.video_T, 3, R, R, device=dev)
            else:
                x = torch.rand(B, 3, R, R, device=dev)
            t = bench(x, n_iter=args.iter)
            mpix = B * (args.video_T or 1) * R * R / 1e6
            throughput = mpix / t if t > 0 else float("inf")
            print(f"  B={B:<2d} R={R:<4d} T={args.video_T or 1:<2d} "
                  f"time={t * 1e3:8.2f} ms  throughput={throughput:7.2f} Mpix/s")


if __name__ == "__main__":
    main()
