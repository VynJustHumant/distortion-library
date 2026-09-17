# Development Guide

Workflow and conventions for modules in `distortion_library/`.

## 1. Setup (once per environment)

```bash
pip install pytest torch numpy pillow
```

## 2. Daily workflow

Run the fast suite every time you edit module code:

```bash
pytest tests/ -v -m "not benchmark"
```

Expected: all tests PASSED (except skipped ones).

Tests marked `slow` (fp64 gradcheck) can be skipped for quick iteration:

```bash
pytest tests/ -v -m "not slow and not benchmark"
```

## 3. Generate golden hash (optional)

After a module is stable, generate hashes for regression testing:

```bash
python scripts/generate_golden.py > /tmp/golden.txt
```

Copy the contents of `/tmp/golden.txt` into `GOLDEN_HASHES` in
`tests/test_atmospheric.py`.

If `GOLDEN_HASHES` is empty, the golden test auto-skips (no error).

## 4. Benchmark (optional, manual)

Check module performance:

```bash
pytest tests/ -v -m benchmark
```

Run weekly or after optimizations.

## 5. Visualization

Generate a severity sweep for the README:

```bash
python examples/sweep_frc_optical_flow.py --out examples/outputs/sweep_frc_optical_flow.png
```

The resulting PNG goes to `examples/outputs/`. Upload it to GitHub so
it appears in the README.

## CI

GitHub Actions automatically runs the following on every push to `main`:

```bash
pytest tests/ -v -m "not benchmark"
```

Configuration is in `.github/workflows/test.yml`.

## Repository Structure

```
distortion-library/
├── distortion_library/          # distortion module code
├── tests/                       # pytest suite
├── scripts/                     # helper tools (golden hash generator)
├── examples/                    # severity sweep scripts
│   └── outputs/                 # PNG outputs for documentation
├── pytest.ini                   # pytest config (markers, testpaths)
├── conftest.py                  # sys.path fix for CI
├── requirements.txt             # dependencies
├── README.md                    # user-facing documentation
└── DEVELOPMENT.md               # this file
```

## Adding a New Module

1. Write the module in `distortion_library/` — follow the naming
   convention (`blur_*`, `moire_*`, etc.).
2. Write the test in `tests/test_<module_name>.py`.
3. Register it in `distortion_library/__init__.py` (`import` + `__all__`).
4. Run `pytest tests/ -v -m "not slow and not benchmark"` until it passes.
5. Update the table in `README.md`.
6. Commit and push — CI verifies automatically.

## Module API Contract

Every module must follow the signature:

```python
def <module_name>(
    image: torch.Tensor,
    severity,
    seed=None,
    generator=None,
    value_range=(0.0, 1.0),
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (distorted_image, label)."""
```

- **Input:** `(C,H,W)`, `(B,C,H,W)`, or `(B,T,C,H,W)` for video.
- **Output:** same shape/dtype/device/memory_format as input.
- **Label:** `(2,)` single, `(B,2)` batch — CPU, float32, detached.
```
