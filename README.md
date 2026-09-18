# Distortion Library

![Tests](https://github.com/VynJustHumant/distortion-library/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade PyTorch library for differentiable image degradation
and physics-based augmentation for computer vision.

## What This Is

Physics-based distortion modules for CV robustness testing and synthetic
dataset generation. Every module is:

- **Physics-based** — turbulence follows Kolmogorov, FRC blur uses
  optical-flow temporal integration, JPEG re-compression follows DCT
  quantization with multi-generation accumulation.
- **Differentiable** — end-to-end, `gradcheck`-clean.
- **Reproducible** — SHA-256 deterministic per-sample seeding.
- **Batch-invariant** — single sample output equals its batch counterpart.
- **Tested** — pytest suite per module.

## Available Distortions

| ID   | Name                            | Category    | Input          |
|------|---------------------------------|-------------|----------------|
| 1004 | Atmospheric Turbulence Blur     | Optical     | Image          |
| —    | FRC Optical Flow Blur           | Temporal    | Image + Video  |
| 1006 | Multi-Generation Re-Compression | Compression | Image (RGB)    |

**Legend:**
- *Image* = `(C,H,W)` and `(B,C,H,W)`
- *Video* = also `(B,T,C,H,W)`

## Quick Start

### Static image — atmospheric turbulence

```python
import torch
from distortion_library import blur_atmospheric_turbulence_v1

x = torch.rand(1, 3, 256, 256)          # (B, C, H, W)
distorted, label = blur_atmospheric_turbulence_v1(x, severity=0.5, seed=42)
# label = [1004.0, 0.5]
```

### Video — FRC optical flow blur

```python
from distortion_library import blur_frc_optical_flow_v1

video = torch.rand(2, 8, 3, 128, 128)   # (B, T, C, H, W)
distorted, label = blur_frc_optical_flow_v1(video, severity=0.5, seed=42)
```

### RGB image — JPEG re-compression

```python
from distortion_library import compression_recompression_v1

x = torch.rand(2, 3, 256, 256)          # (B, C, H, W)
distorted, label = compression_recompression_v1(x, severity=0.5)
```

## Visual Output

### Atmospheric Turbulence Blur — Severity Sweep

![Atmospheric Turbulence Sweep](examples/outputs/sweep_atmospheric.png)

Input image followed by seven increasing severities (0.05 → 1.00).
Heat shimmer and tilt displacement become visible from s ≈ 0.30 onward.

### FRC Optical Flow Blur — Severity Sweep

![FRC Severity Sweep](examples/outputs/sweep_frc_optical_flow.png)

Input image followed by seven increasing severities (0.05 → 1.00), all
rendered from the same seed so the differences are attributable purely
to severity.

### Multi-Generation Re-Compression — Severity Sweep

![Compression Sweep Grid](examples/outputs/sweep_compression/sweep_grid.png)

Three test patterns (checkerboard, gradient, smooth) at seven severities.
The checkerboard exposes 8×8 block artifacts; the gradient reveals
banding; MSE vs. severity curve:

![Compression MSE Curve](examples/outputs/sweep_compression/mse_vs_severity.png)

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/ -v -m "not benchmark and not slow"
```

## Architecture

Every distortion follows the same wrapper contract:

```python
def xxx_wrapper(
    image: torch.Tensor,
    severity,
    seed=None,
    generator=None,
    value_range=(0.0, 1.0),
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (distorted_image, label)."""
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the full development workflow.

## License

MIT — see [LICENSE](LICENSE).
```
