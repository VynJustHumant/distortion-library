"""Severity sweep visualisation for speckle_coherent_v1.

Usage:
    python examples/sweep_speckle_coherent_v1.py \
        --shape 256 256 --rho 2.0 \
        --out examples/outputs/sweep_speckle_coherent_v1.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from distortion_library.speckle_coherent_v1 import distortion  # noqa: E402

_DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), "outputs", "sweep_speckle_coherent_v1.png"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    H, W = args.shape
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij"
    )
    base = 0.3 + 0.4 * torch.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.05)
    base = base.unsqueeze(0).repeat(3, 1, 1)

    sevs = [0.01, 0.1, 0.3, 0.6, 1.0]
    rows = []
    sep = torch.ones(3, H, 8)
    for s in sevs:
        iid, _ = distortion(base, s, seed=args.seed, rho=None)
        cor, _ = distortion(base, s, seed=args.seed, rho=args.rho)
        rows.append(torch.cat([base, sep, iid, sep, cor], dim=2))
        print(f"s={s:.2f}  iid MSE={(iid - base).pow(2).mean():.5f}   "
              f"corr MSE={(cor - base).pow(2).mean():.5f}")

    grid = torch.cat(rows, dim=1)

    try:
        import torchvision
        torchvision.utils.save_image(grid, args.out)
    except Exception:
        try:
            from PIL import Image
            arr = (grid.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(args.out)
        except Exception as e:
            print(f"Could not write PNG ({e}); skipping export.")

    print(f"Wrote {args.out}")
    print(f"Columns: [clean | iid lognormal | correlated rho={args.rho}]")
    print(f"Rows:    s ∈ {sevs}")


if __name__ == "__main__":
    main()
