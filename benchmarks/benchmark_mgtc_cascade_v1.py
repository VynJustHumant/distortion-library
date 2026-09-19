import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "distortion_library"))

import torch
from mgtc_cascade_v1 import mgtc_cascade_v1, MGTCCascadeV1


def bench(B, T, C, H, W, dtype, device, iters, warmup):
    x = torch.rand(B, T, C, H, W, device=device, dtype=dtype)
    s = torch.full((B,), 0.6, device=device, dtype=dtype)

    for _ in range(warmup):
        mgtc_cascade_v1(x, s)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y, lab = mgtc_cascade_v1(x, s)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    assert torch.isfinite(y.float()).all()
    assert y.float().min() >= -1e-3 and y.float().max() <= 1.0 + 1e-3
    assert lab.shape == (B, 2)
    assert (lab[:, 0] == MGTCCascadeV1.DISTORTION_ID).all()
    assert torch.allclose(lab[:, 1].float(), s.float(), atol=1e-5)

    frames = B * T
    return dt, frames / dt, frames * H * W / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    dtypes = [torch.float32]
    if args.device == "cuda":
        dtypes += [torch.float16, torch.bfloat16]

    for dt in dtypes:
        print(f"\ndevice={args.device} dtype={dt}")
        print(f"{'shape':<22}{'time(ms)':>10}{'img/s':>12}{'Mpix/s':>12}")
        for res in (256, 512, 1024):
            for B, T in ((1, 8), (2, 8)):
                try:
                    t, imgs, pix = bench(B, T, 3, res, res, dt,
                                         args.device, args.iters, args.warmup)
                    print(f"{f'B={B},T={T},{res}x{res}':<22}"
                          f"{t*1e3:>10.2f}{imgs:>12.1f}{pix/1e6:>12.2f}")
                except RuntimeError as e:
                    print(f"{f'B={B},T={T},{res}x{res}':<22}"
                          f"  skipped: {str(e)[:50]}")
                    if args.device == "cuda":
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
