"""
Severity sweep visualizer for compression_recompression_v1.

Usage
-----
    python examples/sweep_compression.py --out examples/out/sweep_compression --size 128

Outputs (PNG):
    sweep_grid.png       grid: rows = test images, cols = severities (+original)
    mse_vs_severity.png  MSE(s) curve per image, log scale
    per_channel_mse.png  R/G/B MSE at s=1.0
"""
from __future__ import annotations

# ---------------------------------------------------------------
# sys.path bootstrap — must run before any distortion_library import
# ---------------------------------------------------------------
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _HERE.parents if (p / "distortion_library").is_dir()),
    _HERE.parent.parent,  # fallback: repo/examples/file.py -> repo/
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------
# stdlib / third-party
# ---------------------------------------------------------------
import argparse
import math                                    # [FIX] moved to top
from pathlib import Path as _P                 # noqa: F811 (explicit alias)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------
# module under test
# ---------------------------------------------------------------
from distortion_library.compression_recompression_v1 import distortion  # [FIX]


SEVERITIES = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]
_MSE_FLOOR = 1e-12                             # [FIX] guard against log(0)


# ---------------------------------------------------------------
# deterministic test images (no RNG)
# ---------------------------------------------------------------
def _checkerboard(h: int, w: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    tile = ((yy // 8) + (xx // 8)) % 2
    return tile.float().unsqueeze(0).expand(3, -1, -1).clone()


def _gradient(h: int, w: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    r = xx.float() / max(w - 1, 1)
    g = yy.float() / max(h - 1, 1)
    b = 1.0 - 0.5 * (r + g)
    return torch.stack([r, g, b], dim=0)


def _smooth(h: int, w: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    x = xx.float() / w
    y = yy.float() / h
    r = 0.5 + 0.4 * torch.sin(2 * math.pi * x) * torch.cos(2 * math.pi * y)
    g = 0.5 + 0.4 * torch.sin(2 * math.pi * y + 1.0)
    b = 0.5 + 0.4 * torch.cos(2 * math.pi * x + 2.0)
    return torch.stack([r, g, b], dim=0).clamp(0, 1)


def build_inputs(size: int):
    return {
        "checkerboard": _checkerboard(size, size),
        "gradient":     _gradient(size, size),
        "smooth":       _smooth(size, size),
    }


# ---------------------------------------------------------------
# sweep
# ---------------------------------------------------------------
def run_sweep(size: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(size)

    # --- grid ---
    fig, axes = plt.subplots(
        len(inputs), len(SEVERITIES) + 1,
        figsize=(2.0 * (len(SEVERITIES) + 1), 2.0 * len(inputs)),
        squeeze=False,
    )
    curves: dict = {}
    for i, (name, img) in enumerate(inputs.items()):
        axes[i][0].imshow(img.permute(1, 2, 0).numpy())
        axes[i][0].set_ylabel(name, fontsize=9)
        axes[i][0].set_xticks([]); axes[i][0].set_yticks([])
        if i == 0:
            axes[i][0].set_title("original", fontsize=9)

        mses = []
        for j, s in enumerate(SEVERITIES):
            y, _ = distortion(img, s)
            mse = ((y - img) ** 2).mean().item()
            mses.append(mse)
            ax = axes[i][j + 1]
            ax.imshow(y.clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"s={s:g}\nMSE={mse:.2e}", fontsize=8)
        curves[name] = mses

    fig.suptitle("compression_recompression_v1 — severity sweep", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = out_dir / "sweep_grid.png"
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)

    # --- MSE curve ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, mses in curves.items():
        # [FIX] clip before log scale
        clipped = [max(m, _MSE_FLOOR) for m in mses]
        ax.plot(SEVERITIES, clipped, marker="o", label=name)
    ax.set_xlabel("severity s")
    ax.set_ylabel("MSE vs. original (log scale)")
    ax.set_yscale("log")
    ax.set_title("MSE vs severity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    curve_path = out_dir / "mse_vs_severity.png"
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    # --- per-channel MSE at s=1 ---
    fig, ax = plt.subplots(figsize=(6, 3))
    width = 0.25
    for ci, cname in enumerate(["R", "G", "B"]):
        vals = []
        for name, img in inputs.items():
            y, _ = distortion(img, 1.0)
            vals.append(((y[ci] - img[ci]) ** 2).mean().item())
        xs = np.arange(len(inputs)) + (ci - 1) * width
        ax.bar(xs, vals, width=width, label=cname)
    ax.set_xticks(np.arange(len(inputs)))
    ax.set_xticklabels(list(inputs.keys()))
    ax.set_ylabel("MSE (s=1.0)")
    ax.set_title("Per-channel MSE at s=1.0")
    ax.legend()
    fig.tight_layout()
    chan_path = out_dir / "per_channel_mse.png"
    fig.savefig(chan_path, dpi=120)
    plt.close(fig)

    print(f"[sweep] wrote:\n  {grid_path}\n  {curve_path}\n  {chan_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("examples/out/sweep_compression"))
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    run_sweep(args.size, args.out)


if __name__ == "__main__":
    main()
