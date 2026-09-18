"""
Golden-file regression for compression_recompression_v1.

First run:
    - Creates tests/golden/*.pt and SKIPS (so you can commit the baseline).
    - Then re-run to verify.

Update baseline:
    UPDATE_GOLDEN=1 pytest tests/test_golden_compression_recompression_v1.py

Run only fast tests:
    pytest -m "not slow"

Notes
-----
- Golden values are tight to CPU / fp32 / a specific torch build.
  Cross-version drift >1e-5 fp32 is expected; regenerate on upgrade.
- Uses a *deterministic constructed* input (no torch.rand) so goldens
  are independent of RNG state.
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
    _HERE.parent.parent,  # fallback
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------
import os

import pytest
import torch

from distortion_library.compression_recompression_v1 import (     # [FIX]
    distortion, DISTORTION_ID,
)


GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN", "0") == "1"


# [FIX] no side effect at import time
@pytest.fixture(scope="module", autouse=True)
def _ensure_golden_dir():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    yield


def _fixed_input(shape=(2, 3, 64, 64), dtype=torch.float32) -> torch.Tensor:
    """Smooth, deterministic, RNG-free tensor in [0,1]."""
    n = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float64)
    v = 0.5 + 0.5 * torch.sin(n * 0.017) * torch.cos(n * 0.0031)
    return v.reshape(shape).to(dtype=dtype).clamp(0.0, 1.0)


def _safe_load(path: Path):
    """torch.load with weights_only=True on new torch, fallback on old."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # [FIX]
    except TypeError:
        # weights_only kwarg not available (torch < 2.0)
        return torch.load(path, map_location="cpu")


CASES = [
    ("s0.01",          0.01, {}),
    ("s0.50",          0.50, {}),
    ("s1.00",          1.00, {}),
    ("s0.50_nmax2",    0.50, {"N_max": 2}),
    ("s0.50_bilinear", 0.50, {"chroma_upsample": "bilinear"}),
]


@pytest.mark.slow                                                # [FIX]
@pytest.mark.parametrize("name,severity,kwargs", CASES)
def test_golden(name, severity, kwargs, _ensure_golden_dir):
    path = GOLDEN_DIR / f"compression_recompression_v1_{name}.pt"
    x = _fixed_input()
    y, label = distortion(x, severity, **kwargs)

    if UPDATE or not path.exists():
        torch.save({"y": y, "label": label}, path)
        if not UPDATE:
            pytest.skip(f"created golden {path.name}; re-run to compare")
        return

    ref = _safe_load(path)

    torch.testing.assert_close(
        y, ref["y"], atol=1e-5, rtol=1e-4,
        msg=f"image mismatch vs golden '{name}'",
    )
    torch.testing.assert_close(
        label, ref["label"], atol=0.0, rtol=0.0,
        msg=f"label mismatch vs golden '{name}'",
    )


@pytest.mark.slow                                                # [FIX]
def test_golden_id_in_label():
    """Sanity: label[..., 0] is always DISTORTION_ID."""
    x = _fixed_input()
    _, label = distortion(x, 0.5)
    assert (label[..., 0] == float(DISTORTION_ID)).all()
