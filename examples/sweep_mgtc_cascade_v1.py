import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "distortion_library"))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mgtc_cascade_v1 import mgtc_cascade_v1


def parse_shape(s):
    try:
        parts = tuple(map(int, s.split(",")))
        assert len(parts) == 5 and all(p > 0 for p in parts)
        return parts
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"--shape must be B,T,C,H,W (e.g. 1,8,3,96,96); got {s!r}") from e


def make_clip(B, T, C, H, W, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H),
                            torch.linspace(0, 1, W), indexing="ij")
    base = ((xx + yy) / 2).unsqueeze(0).repeat(C, 1, 1)
    clip = torch.stack(
        [torch.roll(base, shifts=int(t * 0.05 * W), dims=-1) for t in range(T)],
        0).unsqueeze(0)
    clip = clip + 0.05 * torch.randn(clip.shape, generator=g)
    return clip.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sweep.png")
    ap.add_argument("--shape", type=parse_shape, default=(1, 8, 3, 96, 96))
    args = ap.parse_args()
    B, T, C, H, W = args.shape
    x = make_clip(B, T, C, H, W)

    rows = []
    for s in (0.01, 0.1, 0.25, 0.5, 0.75, 1.0):
        y, _ = mgtc_cascade_v1(x, s)
        rows.append((s, y, F.mse_loss(y, x).item()))

    fig, axes = plt.subplots(len(rows), T, figsize=(2 * T, 2 * len(rows)))
    if T == 1:
        axes = axes.reshape(len(rows), 1)
    for i, (s, y, mse) in enumerate(rows):
        for t in range(T):
            ax = axes[i, t]
            ax.imshow(y[0, t].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.axis("off")
            if t == 0:
                ax.set_title(f"s={s:.2f}\nMSE={mse:.4f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
