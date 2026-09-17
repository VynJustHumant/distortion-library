# examples/sweep_frc_optical_flow.py
"""
Severity sweep for blur_frc_optical_flow_v1.

Produces `sweep_frc_optical_flow.png`: a grid showing the input image
and its response to seven increasing severities, all rendered from the
same seed so the differences are attributable purely to severity.

Only requires PIL + numpy (no torchvision).

Run from anywhere:
    python examples/sweep_frc_optical_flow.py --out sweep_frc.png
    python -m examples.sweep_frc_optical_flow  --out sweep_frc.png

The script is self-contained: it prepends the repo root to ``sys.path``
so ``python examples/sweep_frc_optical_flow.py`` works from the repo
root without setting ``PYTHONPATH=.``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# --- sys.path bootstrap (must precede the distortion_library import) ------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from distortion_library.blur_frc_optical_flow_v1 import (
    blur_frc_optical_flow_v1 as frc,
)


SEVERITIES = [0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0]


def _load_image(path: Optional[str], device: torch.device) -> torch.Tensor:
    if path is None:
        # Synthetic structured image: gradient + fine checkerboard +
        # coloured rectangles — the checkerboard exposes aliasing, the
        # rectangles expose structure, the gradient exposes global shift.
        H = W = 256
        ys = torch.linspace(0, 1, H).view(H, 1, 1)
        xs = torch.linspace(0, 1, W).view(1, W, 1)
        img = torch.cat(
            [ys.expand(H, W, 1), xs.expand(H, W, 1), 0.5 * torch.ones(H, W, 1)],
            dim=-1,
        )
        chk = (
            (torch.arange(H).view(H, 1) // 8 + torch.arange(W).view(1, W) // 8) % 2
        ).float().unsqueeze(-1)
        img = (0.7 * img + 0.3 * chk).clamp(0, 1)
        img[64:128, 64:128] = torch.tensor([1.0, 0.2, 0.2])
        img[160:192, 32:224] = torch.tensor([0.1, 0.9, 0.1])
        return img.permute(2, 0, 1).to(device)

    arr = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def _save_grid(
    tiles: List[torch.Tensor],
    out_path: Path,
    nrow: int,
    pad: int = 4,
    pad_value: float = 1.0,
) -> None:
    """Compose a grid of (C,H,W) tensors into a single PNG via PIL."""
    n = len(tiles)
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol
    C, H, W = tiles[0].shape

    total_h = nrows * H + (nrows + 1) * pad
    total_w = ncol * W + (ncol + 1) * pad
    canvas = torch.full((C, total_h, total_w), pad_value, dtype=tiles[0].dtype)

    for i, t in enumerate(tiles):
        r, c = divmod(i, ncol)
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[:, y0 : y0 + H, x0 : x0 + W] = t

    arr = (
        canvas.clamp(0.0, 1.0).permute(1, 2, 0).contiguous().numpy() * 255.0
    ).astype("uint8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None,
                    help="Optional input path; synthetic image if omitted.")
    ap.add_argument("--out", type=str, default="sweep_frc_optical_flow.png")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    x = _load_image(args.image, device)

    tiles: List[torch.Tensor] = [x]
    for s in SEVERITIES:
        y, _ = frc(x, severity=s, seed=args.seed)
        tiles.append(y.detach().cpu())

    out_path = Path(args.out)
    _save_grid(tiles, out_path, nrow=len(SEVERITIES) + 1, pad=4, pad_value=1.0)
    print(
        f"wrote {out_path} ({len(tiles)} tiles: input + severities "
        f"{', '.join(f'{s:.2f}' for s in SEVERITIES)})"
    )


if __name__ == "__main__":
    main()
