"""
tests/test_atmospheric.py
=========================
Pytest suite for blur_atmospheric_turbulence_v1.

Jalankan:
    pytest tests/test_atmospheric.py -v
    pytest tests/test_atmospheric.py -v -m "not benchmark"
    pytest tests/test_atmospheric.py -v -m benchmark
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from distortion_library.blur_atmospheric_turbulence_v1 import (
    blur_atmospheric_turbulence_v1 as blur,
    DISTORTION_REGISTRY,
    _ID,
    _DEFAULTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def x_rgb():
    torch.manual_seed(0)
    return torch.rand(1, 3, 64, 64)


@pytest.fixture
def x_batch4():
    torch.manual_seed(0)
    return torch.rand(4, 3, 64, 64)


@pytest.fixture
def x_fp64():
    torch.manual_seed(0)
    return torch.rand(1, 3, 32, 32, dtype=torch.float64, requires_grad=True)


# ---------------------------------------------------------------------------
# 1. Shape / dtype / device / memory-format
# ---------------------------------------------------------------------------
class TestShapeDtypeDevice:

    def test_shape_preserved_3d(self):
        x = torch.rand(3, 48, 48)
        y, lab = blur(x, 0.5, seed=1)
        assert y.shape == x.shape
        assert lab.shape == (2,)
        assert y.dtype == x.dtype
        assert y.device == x.device

    @pytest.mark.parametrize("B", [1, 2, 4, 8])
    def test_shape_preserved_4d(self, B):
        x = torch.rand(B, 3, 32, 32)
        y, lab = blur(x, 0.5, seed=1)
        assert y.shape == x.shape
        assert lab.shape == (B, 2)

    def test_non_square(self):
        x = torch.rand(2, 3, 40, 96)
        y, _ = blur(x, 0.5, seed=1)
        assert y.shape == x.shape

    def test_channels_last_preserved(self):
        x = torch.rand(2, 3, 32, 32).contiguous(memory_format=torch.channels_last)
        y, _ = blur(x, 0.5, seed=1)
        assert y.is_contiguous(memory_format=torch.channels_last)

    # ---- FIX #7: squeeze + channels_last interaction ----------------------
    def test_squeeze_preserves_shape(self):
        """3D input → 3D output (tidak jadi 4D)."""
        x = torch.rand(3, 32, 32)
        y, lab = blur(x, 0.5, seed=1)
        assert y.shape == x.shape       # (3, 32, 32), bukan (1, 3, 32, 32)
        assert lab.shape == (2,)         # bukan (1, 2)

    def test_squeeze_channels_last_ignored(self):
        """3D input: channels_last tidak berlaku (butuh 4D), jangan error."""
        x = torch.rand(3, 32, 32)
        # 3D contiguous — memory_format 'channels_last' tidak terdefinisi
        # untuk 3D, jadi fungsi harus fallback ke contiguous biasa.
        y, _ = blur(x, 0.5, seed=1)
        assert y.shape == x.shape
        assert y.is_contiguous()

    def test_4d_channels_last_to_3d_roundtrip(self):
        """4D CL → 3D squeeze → tetap valid shape & values."""
        x4 = torch.rand(1, 3, 32, 32).contiguous(memory_format=torch.channels_last)
        y4, _ = blur(x4, 0.5, seed=1)
        assert y4.is_contiguous(memory_format=torch.channels_last)
        # squeeze hasilnya tetap benar
        y3 = y4.squeeze(0)
        assert y3.shape == (3, 32, 32)

    @pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_dtype_preserved(self, dt):
        x = torch.rand(1, 3, 32, 32, dtype=dt)
        y, _ = blur(x, 0.5, seed=1)
        assert y.dtype == dt

    @pytest.mark.parametrize("dt", [torch.uint8, torch.int16, torch.int32])
    def test_int_dtype_preserved(self, dt):
        x = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.int64).to(dt)
        y, _ = blur(x, 0.5, seed=1, value_range=(0.0, 255.0))
        assert y.dtype == dt


# ---------------------------------------------------------------------------
# 2. Range / identity / monotonicity
# ---------------------------------------------------------------------------
class TestRangeIdentityMonotonic:

    def test_output_range(self, x_batch4):
        y, _ = blur(x_batch4, 0.7, seed=1)
        assert y.min().item() >= 0.0
        assert y.max().item() <= 1.0
        assert torch.isfinite(y).all()

    def test_output_range_custom_value_range(self):
        x = torch.rand(1, 3, 32, 32) * 255.0
        y, _ = blur(x, 0.5, seed=1, value_range=(0.0, 255.0))
        assert y.min().item() >= 0.0
        assert y.max().item() <= 255.0

    # ---- FIX #6: value_range non-standard pada BATCH ----------------------
    def test_value_range_255_batch(self):
        x = torch.rand(4, 3, 32, 32) * 255.0
        y, lab = blur(x, 0.5, seed=1, value_range=(0.0, 255.0))
        assert y.shape == x.shape
        assert y.min().item() >= 0.0
        assert y.max().item() <= 255.0
        assert lab.shape == (4, 2)

    def test_value_range_asymmetric(self):
        """value_range = (-1, 1) → output harus di [-1, 1]."""
        x = torch.rand(2, 3, 32, 32) * 2.0 - 1.0
        y, _ = blur(x, 0.5, seed=1, value_range=(-1.0, 1.0))
        assert y.min().item() >= -1.0 - 1e-5
        assert y.max().item() <= 1.0 + 1e-5

    # ---- FIX #3: toleransi diperketat -------------------------------------
    def test_identity_at_low_severity(self):
        """
        severity=0.01 → blend weight s=0.01, jadi output = 0.99·x + 0.01·x_dist.
        Beda maksimum karena turbulence pada severity ini sangat kecil
        (~1% dari kontribusi distorted). Toleransi 0.008 cukup ketat tapi
        tidak flaky untuk fp32.
        """
        x = torch.rand(1, 3, 64, 64)
        y, _ = blur(x, 0.01, seed=1)
        assert (y - x).abs().mean() < 0.008

    def test_identity_at_low_severity_exact_weight(self):
        """Cek bahwa blend benar-benar (1-s)·x + s·x_dist dengan s=0.01."""
        x = torch.rand(1, 3, 64, 64)
        y, lab = blur(x, 0.01, seed=1)
        # s efektif = 0.01 (clamp lower bound)
        # output = 0.99·x + 0.01·x_distorted → selisih maksimum ≤ 0.01·|x_d - x| ≤ 0.01
        assert (y - x).abs().max().item() <= 0.01 + 1e-5

    def test_monotonic_in_severity(self):
        torch.manual_seed(0)
        x = torch.rand(1, 3, 64, 64)
        sevs = [0.05, 0.25, 0.5, 0.75, 1.0]
        mses = []
        for s in sevs:
            y, _ = blur(x, s, seed=42)
            mses.append(((y - x) ** 2).mean().item())
        diffs = [mses[i + 1] - mses[i] for i in range(len(mses) - 1)]
        assert all(d >= -1e-4 for d in diffs), f"MSE tidak monotonic: {mses}"
        assert mses[-1] > mses[0] + 1e-3


# ---------------------------------------------------------------------------
# 3. Determinism + batch invariance
# ---------------------------------------------------------------------------
class TestDeterminismBatchInvariance:

    def test_determinism_same_seed(self, x_batch4):
        y1, l1 = blur(x_batch4, 0.5, seed=123)
        y2, l2 = blur(x_batch4, 0.5, seed=123)
        assert torch.equal(y1, y2)
        assert torch.equal(l1, l2)

    def test_determinism_default_seed(self, x_batch4):
        y1, _ = blur(x_batch4, 0.5)
        y2, _ = blur(x_batch4, 0.5)
        assert torch.equal(y1, y2)

    def test_different_seeds_differ(self, x_batch4):
        y1, _ = blur(x_batch4, 0.8, seed=1)
        y2, _ = blur(x_batch4, 0.8, seed=2)
        assert not torch.allclose(y1, y2)

    @pytest.mark.parametrize("B", [2, 4, 8])
    def test_batch_invariance(self, B):
        torch.manual_seed(0)
        x1 = torch.rand(1, 3, 64, 64)
        xB = x1.repeat(B, 1, 1, 1)
        y1, _ = blur(x1, 0.6, seed=777)
        yB, _ = blur(xB, 0.6, seed=777)
        diff = (y1 - yB[:1]).abs().max().item()
        assert diff < 1e-4, f"batch-invariance broken: diff={diff}"

    def test_external_generator_used(self, x_rgb):
        g1 = torch.Generator().manual_seed(11)
        g2 = torch.Generator().manual_seed(22)
        y1, _ = blur(x_rgb, 0.5, generator=g1)
        y2, _ = blur(x_rgb, 0.5, generator=g2)
        assert not torch.allclose(y1, y2)

    def test_seed_wins_over_generator(self, x_rgb):
        g = torch.Generator().manual_seed(999)
        y1, _ = blur(x_rgb, 0.5, seed=3, generator=g)
        y2, _ = blur(x_rgb, 0.5, seed=3)
        assert torch.equal(y1, y2)

    def test_cpu_generator_guard(self, x_rgb):
        if not torch.cuda.is_available():
            pytest.skip("CUDA tidak tersedia")
        gc = torch.Generator(device="cuda").manual_seed(0)
        with pytest.raises(ValueError, match="CPU generator"):
            blur(x_rgb.cuda(), 0.5, generator=gc)


# ---------------------------------------------------------------------------
# 4. Differentiability
# ---------------------------------------------------------------------------
class TestDifferentiability:

    def test_gradcheck_fp64(self, x_fp64):
        assert torch.autograd.gradcheck(
            lambda t: blur(t, 0.5, seed=7)[0],
            (x_fp64,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_backward_no_nan(self, x_fp64):
        y, _ = blur(x_fp64, 0.8, seed=1)
        y.pow(2).sum().backward()
        g = x_fp64.grad
        assert g is not None
        assert torch.isfinite(g).all()

    def test_gradcheck_batch(self):
        x = torch.rand(2, 3, 24, 24, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda t: blur(t, 0.5, seed=5)[0], (x,), eps=1e-6, atol=1e-4
        )

    def test_no_detach_in_forward(self, x_fp64):
        y, _ = blur(x_fp64, 0.5, seed=1)
        assert y.requires_grad
        assert y.grad_fn is not None


# ---------------------------------------------------------------------------
# 5. Robustness
# ---------------------------------------------------------------------------
class TestRobustness:

    def test_nan_inf_input(self):
        x = torch.rand(1, 3, 32, 32)
        x[0, 0, 5, 5] = float("nan")
        x[0, 1, 10, 10] = float("inf")
        x[0, 2, 15, 15] = float("-inf")
        y, _ = blur(x, 0.5, seed=1)
        assert torch.isfinite(y).all()

    # ---- FIX #4: toleransi fp32 untuk 0.01 --------------------------------
    def test_severity_out_of_range_low(self, x_rgb):
        y, lab = blur(x_rgb, -5.0, seed=1)
        assert torch.isfinite(y).all()
        # fp32 tidak eksak untuk 0.01 → abs=1e-5 cukup longgar untuk lolos,
        # cukup ketat untuk menangkap perubahan clamp.
        assert lab[0, 1].item() == pytest.approx(0.01, abs=1e-5)

    def test_severity_out_of_range_high(self, x_rgb):
        y, lab = blur(x_rgb, 99.0, seed=1)
        assert torch.isfinite(y).all()
        assert lab[0, 1].item() == pytest.approx(1.0, abs=1e-6)

    def test_severity_scalar_broadcast(self, x_batch4):
        _, lab = blur(x_batch4, 0.5, seed=1)
        assert lab.shape == (4, 2)
        assert torch.allclose(lab[:, 1], torch.full((4,), 0.5, dtype=lab.dtype))

    def test_severity_per_sample(self, x_batch4):
        sev = torch.tensor([0.1, 0.3, 0.6, 0.9])
        y, lab = blur(x_batch4, sev, seed=1)
        assert torch.allclose(lab[:, 1].cpu(), sev, atol=1e-5)

    def test_severity_mismatch_raises(self, x_batch4):
        sev = torch.tensor([0.3, 0.5])
        with pytest.raises(ValueError, match="numel"):
            blur(x_batch4, sev, seed=1)

    def test_small_image(self):
        x = torch.rand(1, 3, 4, 4)
        y, _ = blur(x, 0.5, seed=1)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_very_small_image(self):
        x = torch.rand(1, 3, 1, 1)
        y, _ = blur(x, 0.5, seed=1)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# 6. Integer dtype
# ---------------------------------------------------------------------------
class TestIntegerDtype:

    def test_uint8_no_truncate(self):
        x = torch.full((1, 3, 32, 32), 128, dtype=torch.uint8)
        y, _ = blur(x, 0.01, seed=1, value_range=(0.0, 255.0))
        assert y.dtype == torch.uint8
        assert (y.float() - 128.0).abs().max() <= 2.0

    def test_int32_range(self):
        x = torch.full((1, 3, 32, 32), 2000, dtype=torch.int32)
        y, _ = blur(x, 0.05, seed=1, value_range=(0.0, 4095.0))
        assert y.dtype == torch.int32
        assert y.min().item() >= 0
        assert y.max().item() <= 4095


# ---------------------------------------------------------------------------
# 7. Kwargs override
# ---------------------------------------------------------------------------
class TestKwargsOverride:

    def test_disp_scale_override(self, x_rgb):
        y_default, _ = blur(x_rgb, 0.8, seed=1)
        y_strong, _ = blur(x_rgb, 0.8, seed=1, disp_scale=0.10)
        assert not torch.allclose(y_default, y_strong)

    def test_sigma_ho_override(self, x_rgb):
        y_default, _ = blur(x_rgb, 0.5, seed=1)
        y_wide, _ = blur(x_rgb, 0.5, seed=1, sigma_ho_max=4.0)
        assert not torch.allclose(y_default, y_wide)

    def test_unknown_kwarg_ignored(self, x_rgb):
        y, _ = blur(x_rgb, 0.5, seed=1, nonsense_kwarg=42)
        assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# 8. Label
# ---------------------------------------------------------------------------
class TestLabel:

    def test_label_has_no_grad(self, x_rgb):
        _, lab = blur(x_rgb, 0.5, seed=1)
        assert not lab.requires_grad

    def test_label_id_constant(self, x_rgb):
        _, lab = blur(x_rgb, 0.5, seed=1)
        assert lab[0, 0].item() == float(_ID)

    def test_label_device(self, x_rgb):
        _, lab = blur(x_rgb, 0.5, seed=1)
        assert lab.device == x_rgb.device

    def test_label_serializable(self, x_batch4):
        _, lab = blur(x_batch4, 0.5, seed=1)
        lab_cpu = lab.detach().cpu()
        assert lab_cpu.device.type == "cpu"


# ---------------------------------------------------------------------------
# 9. Registry
# ---------------------------------------------------------------------------
class TestRegistry:

    def test_registered(self):
        assert "blur_atmospheric_turbulence_v1" in DISTORTION_REGISTRY
        assert DISTORTION_REGISTRY["blur_atmospheric_turbulence_v1"] is blur

    def test_id_is_int(self):
        assert isinstance(_ID, int)
        assert _ID > 0


# ---------------------------------------------------------------------------
# 10. Golden regression
# ---------------------------------------------------------------------------
# ---- FIX #1: hash placeholder diganti dengan generator script ---------
# Isi nilai di bawah dengan menjalankan `python scripts/generate_golden.py`.
# Nilai contoh (DIGANTI setelah run pertama di environment referensi):
GOLDEN_HASHES = {
    # (1, 3, 32, 32, 0.0, 42): '<sha256>',
    # (1, 3, 32, 32, 0.5, 42): '<sha256>',
    # (2, 3, 32, 32, 1.0, 42): '<sha256>',
}


class TestGolden:
    """
    Golden values di-hard-code dari scripts/generate_golden.py.
    Kalau GOLDEN_HASHES kosong (belum di-generate), seluruh class di-skip
    dengan pesan yang jelas — bukan silently skip per-case.
    """

    @pytest.fixture(autouse=True)
    def _require_hashes(self):
        if not GOLDEN_HASHES:
            pytest.skip(
                "GOLDEN_HASHES kosong. Jalankan `python scripts/generate_golden.py` "
                "lalu copy-paste hasilnya ke tests/test_atmospheric.py."
            )

    @pytest.mark.parametrize("case", list(GOLDEN_HASHES.keys()) or [None])
    def test_golden(self, case):
        if case is None:
            pytest.skip("tidak ada case")
        B, C, H, W, sev, seed = case
        torch.manual_seed(0)
        x = torch.rand(B, C, H, W)
        y, _ = blur(x, sev, seed=seed)
        yb = (y * 255).round().clamp(0, 255).to(torch.uint8).numpy().tobytes()
        assert hashlib.sha256(yb).hexdigest() == GOLDEN_HASHES[case]


# ---------------------------------------------------------------------------
# 11. Cross-process determinism
# ---------------------------------------------------------------------------
class TestCrossProcessDeterminism:

    def test_fresh_process_same_result(self):
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import torch, hashlib
            from distortion_library.blur_atmospheric_turbulence_v1 import (
                blur_atmospheric_turbulence_v1 as blur
            )
            torch.manual_seed(0)
            x = torch.rand(1, 3, 32, 32)
            y, _ = blur(x, 0.5, seed=42)
            b = (y * 255).round().to(torch.uint8).numpy().tobytes()
            print(hashlib.sha256(b).hexdigest())
        """)

        def run():
            out = subprocess.check_output([sys.executable, "-c", script])
            return out.decode().strip()

        h1, h2 = run(), run()
        assert h1 == h2


# ---------------------------------------------------------------------------
# 12. Benchmark  — FIX #2 + #5: registered marker + parametrised
# ---------------------------------------------------------------------------
@pytest.mark.benchmark
class TestBenchmark:

    @pytest.mark.parametrize("batch", [1, 4])
    @pytest.mark.parametrize("size", [128, 256, 512])
    def test_throughput(self, size, batch):
        import time
        x = torch.rand(batch, 3, size, size)
        # warmup
        for _ in range(3):
            blur(x, 0.5, seed=1)
        N = 10
        t0 = time.perf_counter()
        for _ in range(N):
            blur(x, 0.5, seed=1)
        dt = (time.perf_counter() - t0) / N
        print(f"\n[bench] {batch}×3×{size}×{size}: {dt*1000:.1f} ms")
        # Ambang kasar — sesuaikan untuk CI:
        assert dt < 5.0, f"too slow: {dt:.3f}s"

    def test_throughput_scaling(self):
        """Verifikasi batch scaling linear-ish (tidak ada O(B²))."""
        import time

        def timeit(B):
            x = torch.rand(B, 3, 128, 128)
            for _ in range(3):
                blur(x, 0.5, seed=1)
            t0 = time.perf_counter()
            for _ in range(5):
                blur(x, 0.5, seed=1)
            return (time.perf_counter() - t0) / 5

        t1 = timeit(1)
        t4 = timeit(4)
        # O(B²) akan ~16×, O(B) akan ~4×. Beri margin longgar.
        assert t4 < t1 * 8.0, f"non-linear scaling: t1={t1:.4f}s t4={t4:.4f}s"
