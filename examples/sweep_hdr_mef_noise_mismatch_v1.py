"""Severity sweep visualization for hdr_mef_noise_mismatch_v1."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from distortion_library.hdr_mef_noise_mismatch_v1 import (
    hdr_mef_noise_mismatch_v1,
    DISTORTION_ID,
)

_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs",
    "sweep_hdr_mef_noise_mismatch_v1.png",
)


def make_synthetic(C: int = 3, H: int = 256, W: int = 256) -> torch.Tensor:
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, H),
        torch.linspace(0.0, 1.0, W),
        indexing="ij",
    )
    base = 0.15 + 0.6 * (0.5 * xx + 0.5 * yy)
    return base.unsqueeze(0).expand(C, H, W).clone().unsqueeze(0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Visualize severity sweep of hdr_mef_noise_mismatch_v1.",
    )
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--severities", type=float, nargs="+",
                    default=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--ev-min", type=float, default=-2.0)
    ap.add_argument("--ev-max", type=float, default=2.0)
    ap.add_argument("--sigma-s", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--rho-t", type=float, default=0.7)
    ap.add_argument("--rho-f", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    base = make_synthetic(3, args.res, args.res)
    if args.video:
        x = base.unsqueeze(1).expand(1, args.T, 3, args.res, args.res).contiguous()
    else:
        x = base

    common = dict(
        K=args.K, ev_min=args.ev_min, ev_max=args.ev_max,
        sigma_s=args.sigma_s, lam=args.lam,
        rho_t=args.rho_t, rho_f=args.rho_f, gamma=args.gamma,
    )

    outs = []
    for s in args.severities:
        y, label = hdr_mef_noise_mismatch_v1(x, severity=s, seed=args.seed, **common)
        # 4D input → label (B, 2); check the id column of sample 0
        assert label[0, 0].item() == DISTORTION_ID
        outs.append(y[0])
        print(f"severity={s:.3f}  label={label.tolist()}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; saving raw tensors only.")
        for i, y in enumerate(outs):
            torch.save(y, f"sweep_hdr_mef_{i}.pt")
        return

    n = len(outs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, s, y in zip(axes, args.severities, outs):
        img = y
        if img.dim() == 4:
            img = img[0]
        img = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"s={s:.2f}")
        ax.axis("off")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
