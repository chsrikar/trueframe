"""
TrueFrame Lane A — Inversion Module
Unit & Integration Tests
=========================

Run with:
    python -m pytest tests/test_invert.py -v

Or directly:
    python tests/test_invert.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running from the repo root or from the tests/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from invert import invert_image, invert_batch, _invert_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(
    width: int = 64,
    height: int = 64,
    mode: str = "RGB",
    dtype: np.dtype = np.uint8,
    fill: int | tuple = None,
) -> Image.Image:
    """Create a synthetic PIL image for testing."""
    if dtype == np.uint16:
        # Pillow represents 16-bit grayscale as mode "I;16"
        if fill is None:
            arr = np.random.randint(0, 65536, (height, width), dtype=np.uint16)
        else:
            arr = np.full((height, width), fill, dtype=np.uint16)
        return Image.fromarray(arr, mode="I;16")

    channels = {"L": 1, "RGB": 3, "RGBA": 4}.get(mode, 3)
    if fill is None:
        if channels == 1:
            arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        else:
            arr = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    else:
        if channels == 1:
            arr = np.full((height, width), fill, dtype=np.uint8)
        else:
            arr = np.full((height, width, channels), fill, dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


# ---------------------------------------------------------------------------
# Test: _invert_array internals
# ---------------------------------------------------------------------------

class TestInvertArray(unittest.TestCase):
    """Low-level tests for the vectorised inversion helper."""

    def test_rgb_uint8_exact_values(self):
        """Each channel value must equal 255 - original."""
        arr = np.array([[[10, 20, 30], [100, 150, 200]]], dtype=np.uint8)
        inv = _invert_array(arr)
        expected = np.array([[[245, 235, 225], [155, 105, 55]]], dtype=np.uint8)
        np.testing.assert_array_equal(inv, expected)

    def test_grayscale_uint8(self):
        arr = np.array([[0, 127, 255]], dtype=np.uint8)
        inv = _invert_array(arr)
        np.testing.assert_array_equal(inv, np.array([[255, 128, 0]], dtype=np.uint8))

    def test_rgba_alpha_preserved(self):
        """Alpha channel must not be inverted."""
        arr = np.array([[[50, 100, 150, 200]]], dtype=np.uint8)  # RGBA
        inv = _invert_array(arr)
        # RGB inverted, A unchanged
        self.assertEqual(inv[0, 0, 0], 205)  # 255 - 50
        self.assertEqual(inv[0, 0, 1], 155)  # 255 - 100
        self.assertEqual(inv[0, 0, 2], 105)  # 255 - 150
        self.assertEqual(inv[0, 0, 3], 200)  # alpha unchanged

    def test_uint16_inversion(self):
        arr = np.array([[0, 1000, 65535]], dtype=np.uint16)
        inv = _invert_array(arr)
        np.testing.assert_array_equal(
            inv, np.array([[65535, 64535, 0]], dtype=np.uint16)
        )

    def test_round_trip_idempotent(self):
        """Inverting twice must return the original exactly."""
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        np.testing.assert_array_equal(_invert_array(_invert_array(arr)), arr)

    def test_unsupported_dtype_raises(self):
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            _invert_array(arr)


# ---------------------------------------------------------------------------
# Test: invert_image (file I/O)
# ---------------------------------------------------------------------------

class TestInvertImage(unittest.TestCase):
    """Integration tests for invert_image()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # --- Helper -----------------------------------------------------------

    def _save_and_invert(self, img: Image.Image, src_name: str, dst_name: str, **kw) -> tuple[Image.Image, str]:
        src = self.tmp_dir / src_name
        dst = self.tmp_dir / dst_name
        img.save(str(src))
        out = invert_image(str(src), str(dst), **kw)
        return Image.open(out), out

    # --- Basic correctness ------------------------------------------------

    def test_rgb_pixel_values(self):
        """A few sampled pixels must equal 255 - original."""
        fill = (80, 120, 200)
        img = _make_image(mode="RGB", fill=fill)
        inv, _ = self._save_and_invert(img, "src.png", "dst.png", mode="negative")
        inv_arr = np.array(inv)
        # All pixels should be (175, 135, 55)
        expected = tuple(255 - c for c in fill)
        for y in (0, 10, 31):
            for x in (0, 10, 31):
                self.assertEqual(tuple(inv_arr[y, x]), expected)

    def test_grayscale(self):
        fill = 60
        img = _make_image(mode="L", fill=fill)
        inv, _ = self._save_and_invert(img, "src_gray.png", "dst_gray.png")
        inv_arr = np.array(inv)
        self.assertTrue(np.all(inv_arr == 195))  # 255 - 60

    def test_rgba_alpha_preserved(self):
        """Alpha channel must be identical before and after inversion."""
        arr = np.zeros((32, 32, 4), dtype=np.uint8)
        arr[:, :, :3] = 80   # RGB
        arr[:, :, 3] = 128   # Alpha
        img = Image.fromarray(arr, "RGBA")
        inv, _ = self._save_and_invert(img, "src_rgba.png", "dst_rgba.png")
        inv_arr = np.array(inv)
        # Alpha channel untouched
        np.testing.assert_array_equal(inv_arr[:, :, 3], 128)
        # RGB inverted
        np.testing.assert_array_equal(inv_arr[:, :, :3], 175)

    def test_round_trip_lossless_png(self):
        """Inverting twice via PNG (lossless) must return the original exactly."""
        img = _make_image(mode="RGB")
        original_arr = np.array(img)

        src = self.tmp_dir / "orig.png"
        neg = self.tmp_dir / "neg.png"
        restored = self.tmp_dir / "restored.png"
        img.save(str(src))

        invert_image(str(src), str(neg), mode="negative")
        invert_image(str(neg), str(restored), mode="positive")

        restored_arr = np.array(Image.open(str(restored)))
        np.testing.assert_array_equal(original_arr, restored_arr,
                                      err_msg="Round-trip inversion must be lossless.")

    def test_mode_positive_identical_to_negative(self):
        """'positive' and 'negative' modes must produce byte-identical outputs."""
        img = _make_image(mode="RGB", fill=(42, 84, 168))
        src = self.tmp_dir / "src.png"
        img.save(str(src))
        out_neg = self.tmp_dir / "neg.png"
        out_pos = self.tmp_dir / "pos.png"
        invert_image(str(src), str(out_neg), mode="negative")
        invert_image(str(src), str(out_pos), mode="positive")
        neg_arr = np.array(Image.open(str(out_neg)))
        pos_arr = np.array(Image.open(str(out_pos)))
        np.testing.assert_array_equal(neg_arr, pos_arr)

    def test_returns_output_path(self):
        img = _make_image()
        src = self.tmp_dir / "x.png"
        dst = self.tmp_dir / "y.png"
        img.save(str(src))
        result = invert_image(str(src), str(dst))
        self.assertEqual(result, str(dst))

    def test_output_dir_created(self):
        img = _make_image()
        src = self.tmp_dir / "x.png"
        dst = self.tmp_dir / "nested" / "deep" / "out.png"
        img.save(str(src))
        invert_image(str(src), str(dst))
        self.assertTrue(dst.exists())

    # --- Error handling ---------------------------------------------------

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            invert_image(str(self.tmp_dir / "nonexistent.jpg"), str(self.tmp_dir / "out.png"))

    def test_invalid_mode_raises(self):
        img = _make_image()
        src = self.tmp_dir / "s.png"
        img.save(str(src))
        with self.assertRaises(ValueError):
            invert_image(str(src), str(self.tmp_dir / "o.png"), mode="banana")

    def test_corrupt_file_raises(self):
        corrupt = self.tmp_dir / "bad.png"
        corrupt.write_bytes(b"THIS IS NOT AN IMAGE FILE \x00\x01\x02")
        from PIL import UnidentifiedImageError
        with self.assertRaises(UnidentifiedImageError):
            invert_image(str(corrupt), str(self.tmp_dir / "out.png"))


# ---------------------------------------------------------------------------
# Test: invert_batch
# ---------------------------------------------------------------------------

class TestInvertBatch(unittest.TestCase):
    """Tests for the batch processing helper."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src_dir = Path(self.tmp.name) / "src"
        self.dst_dir = Path(self.tmp.name) / "dst"
        self.src_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_processes_all_images(self):
        for name in ("a.png", "b.png", "c.png"):
            _make_image(fill=100).save(str(self.src_dir / name))

        results = invert_batch(str(self.src_dir), str(self.dst_dir))
        self.assertEqual(len(results), 3)

    def test_batch_output_naming(self):
        _make_image().save(str(self.src_dir / "photo.png"))
        invert_batch(str(self.src_dir), str(self.dst_dir), suffix="_neg")
        self.assertTrue((self.dst_dir / "photo_neg.png").exists())

    def test_batch_empty_dir_returns_empty(self):
        results = invert_batch(str(self.src_dir), str(self.dst_dir))
        self.assertEqual(results, [])

    def test_batch_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            invert_batch("/no/such/directory", str(self.dst_dir))

    def test_batch_skips_non_image_files(self):
        (self.src_dir / "readme.txt").write_text("not an image")
        _make_image().save(str(self.src_dir / "real.png"))
        results = invert_batch(str(self.src_dir), str(self.dst_dir))
        self.assertEqual(len(results), 1)

    def test_batch_pixel_values_correct(self):
        fill = (30, 60, 90)
        _make_image(mode="RGB", fill=fill).save(str(self.src_dir / "img.png"))
        results = invert_batch(str(self.src_dir), str(self.dst_dir))
        inv_arr = np.array(Image.open(results[0]))
        expected = tuple(255 - c for c in fill)
        self.assertEqual(tuple(inv_arr[0, 0]), expected)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
