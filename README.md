# Distortion Library

![Tests](https://github.com/VynJustHumant/distortion-library/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade PyTorch library for differentiable image degradation
and physics-based augmentation for computer vision.

## What This Is

Physics-based distortion modules for CV robustness testing. Every module is:

- **Physics-based** — moiré emerges from transmittance products, noise
  follows Poisson-Gaussian, turbulence follows Kolmogorov.
- **Differentiable** — end-to-end, `gradcheck`-clean.
- **Reproducible** — SHA-256 deterministic per-sample seeding.
- **Batch-invariant** — single sample output equals its batch counterpart.
- **Tested** — pytest suite per module.

## Available Distortions

| ID   | Name                        | Category  | Input            |
|------|-----------------------------|-----------|------------------|
| 1004 | Atmospheric Turbulence Blur | Optical   | Image            |
| 1005 | FRC Optical Flow Blur       | Temporal  | Image + Video    |

*Image* = `(C,H,W)` and `(B,C,H,W)`; *Video* = also `(B,T,C,H,W)`.

## Quick Start

### Static image

```python
import torch
from distortion_library import blur_atmospheric_turbulence_v1

x = torch.rand(1, 3, 256, 256)
distorted, label = blur_atmospheric_turbulence_v1(x, severity=0.5, seed=42)
Video (temporal)
python
from distortion_library import blur_frc_optical_flow_v1

video = torch.rand(2, 8, 3, 128, 128)   # (B, T, C, H, W)
distorted, label = blur_frc_optical_flow_v1(video, severity=0.5, seed=42)
Installation
bash
pip install -r requirements.txt
Testing
bash
pytest tests/ -v -m "not benchmark"
```
## License
MIT — see LICENSE.
