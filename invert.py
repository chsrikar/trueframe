"""
TrueFrame Forensics Pipeline — Lane A
Image Negative/Positive Inversion Module
=========================================

Purpose:
    Performs pixel-value inversion (negative ↔ positive conversion) on images
    as a preprocessing/analysis step to surface tampering artifacts such as:
      - Splicing boundaries
      - Cloned regions
      - Inconsistent noise / compression patterns
    These artifacts are often invisible in the original luminance space but
    become apparent after inversion.

Author:  TrueFrame Dev Team
Version: 1.0.0
License: MIT

Dependencies:
    pip install Pillow numpy

Optional:
    pip install opencv-python  (if the rest of Lane A uses cv2 arrays)

Library choice note:
    This module standardises on Pillow (PIL) for I/O and NumPy for array math,
    matching the classifier / Grad-CAM preprocessing modules elsewhere in Lane A.
    If you switch to OpenCV, remember BGR ↔ RGB channel order differences — do NOT
    mix PIL and cv2 arrays without explicit conversion.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trueframe.lane_a.invert")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Prefer lossless PNG for intermediate forensic outputs so that JPEG
# re-compression artefacts don't distort the very signals we're hunting.
PREFERRED_INTERMEDIATE_FORMAT: str = "PNG"


# ---------------------------------------------------------------------------
# Core inversion logic
# ---------------------------------------------------------------------------

def _invert_array(arr: np.ndarray) -> np.ndarray:
    """
    Invert a NumPy image array in-place using vectorised subtraction.

    Rules:
      - 8-bit  (uint8)  : inverted = 255   - value
      - 16-bit (uint16) : inverted = 65535 - value
      - Alpha channel (last channel of RGBA / LA) is **preserved** untouched.

    Args:
        arr: A NumPy array of shape (H, W), (H, W, 3), or (H, W, 4).

    Returns:
        A new NumPy array of the same dtype/shape with pixels inverted.

    Raises:
        ValueError: If the array dtype is not uint8 or uint16.
    """
    if arr.dtype == np.uint8:
        max_val = np.uint8(255)
    elif arr.dtype == np.uint16:
        max_val = np.uint16(65535)
    else:
        raise ValueError(
            f"Unsupported array dtype '{arr.dtype}'. Expected uint8 or uint16."
        )

    result = arr.copy()

    # RGBA / LA: invert all channels except the last (alpha)
    if arr.ndim == 3 and arr.shape[2] in (2, 4):
        result[..., :-1] = max_val - arr[..., :-1]
    else:
        # Grayscale (H, W) or RGB (H, W, 3)
        result = max_val - arr

    return result


def invert_image(
    input_path: str,
    output_path: str,
    mode: str = "negative",
    preserve_icc: bool = True,
) -> str:
    """
    Invert an image and save the result to *output_path*.

    Supports JPEG, PNG, BMP, TIFF, WEBP, and both 8-bit and 16-bit images.
    Alpha channels (RGBA / LA) are preserved untouched.

    Args:
        input_path:   Absolute or relative path to the source image.
        output_path:  Destination path for the inverted image. The directory
                      will be created if it does not exist.
        mode:         ``"negative"`` — convert positive → negative.
                      ``"positive"`` — convert negative → positive.
                      Mathematically identical; kept separate for pipeline
                      logging clarity (the mode is recorded in metadata).
        preserve_icc: If True, any embedded ICC colour profile is copied from
                      the source image to the output. Default True.

    Returns:
        The absolute path of the saved output file (str).

    Raises:
        FileNotFoundError:      Source file does not exist.
        ValueError:             Unsupported ``mode`` string, or unsupported
                                image dtype encountered.
        UnidentifiedImageError: Pillow cannot decode the file (corrupt/unsupported).
        OSError:                Any OS-level I/O failure.

    Example::

        out = invert_image(
            input_path="evidence/photo.jpg",
            output_path="forensics/photo_inverted.png",
            mode="negative",
        )
        print(f"Saved to {out}")
    """
    if mode not in ("negative", "positive"):
        raise ValueError(f"Invalid mode '{mode}'. Choose 'negative' or 'positive'.")

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    # --- Validate source --------------------------------------------------
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(
            "Extension '%s' is not in the official support list %s — attempting anyway.",
            input_path.suffix,
            SUPPORTED_EXTENSIONS,
        )

    # --- Load image -------------------------------------------------------
    t_start = time.perf_counter()
    try:
        img = Image.open(str(input_path))
        # Load pixel data now so corruption surfaces here rather than mid-processing.
        img.load()
    except UnidentifiedImageError as exc:
        raise UnidentifiedImageError(
            f"Cannot decode image (corrupt or unsupported format): {input_path}"
        ) from exc

    original_format: str = img.format or "UNKNOWN"
    original_mode: str = img.mode        # e.g. "RGB", "RGBA", "L", "I", ...
    original_size: tuple[int, int] = img.size  # (width, height)
    icc_profile = img.info.get("icc_profile") if preserve_icc else None

    logger.info(
        "Loaded  : %s  [format=%s | mode=%s | size=%dx%d]",
        input_path.name,
        original_format,
        original_mode,
        original_size[0],
        original_size[1],
    )

    # --- Convert to a workable mode ---------------------------------------
    # Pillow's "I" (32-bit signed int) and "F" (float) modes are exotic;
    # normalise to uint16 for 16-bit sources, uint8 for everything else.
    if original_mode == "I":
        # 32-bit signed → treat as 16-bit (forensic images rarely exceed 16-bit)
        img = img.convert("I;16")
        work_img = img
    elif original_mode == "F":
        raise ValueError(
            "Floating-point images ('F' mode) are not supported. "
            "Convert to 8-bit or 16-bit first."
        )
    else:
        work_img = img

    # --- Convert to NumPy, invert, convert back ---------------------------
    arr = np.array(work_img)

    # Determine bit depth for logging
    if arr.dtype == np.uint16:
        bit_depth = 16
    elif arr.dtype == np.uint8:
        bit_depth = 8
    else:
        # Attempt uint8 coercion for unusual dtypes
        logger.warning("Unusual array dtype '%s'; coercing to uint8.", arr.dtype)
        arr = arr.astype(np.uint8)
        bit_depth = 8

    logger.info(
        "Processing: mode=%s | bit_depth=%d-bit | array_shape=%s | dtype=%s",
        mode,
        bit_depth,
        arr.shape,
        arr.dtype,
    )

    inverted_arr = _invert_array(arr)

    # Reconstruct PIL Image
    result_img = Image.fromarray(inverted_arr, mode=work_img.mode)

    # Re-attach ICC profile if requested
    save_kwargs: dict = {}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    # --- Save output ------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Force PNG for intermediate forensic outputs to avoid JPEG re-compression
    # artefacts unless the caller explicitly chose a lossy extension.
    out_ext = output_path.suffix.lower()
    if out_ext in (".jpg", ".jpeg"):
        logger.warning(
            "Output path uses JPEG, which introduces re-compression artefacts that "
            "may distort forensic signals. Consider using PNG for intermediate outputs."
        )
        save_kwargs.setdefault("quality", 95)
        save_kwargs.setdefault("subsampling", 0)

    result_img.save(str(output_path), **save_kwargs)

    t_elapsed = time.perf_counter() - t_start

    # --- Log metadata (for Lane B report generation) ----------------------
    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_format": original_format,
        "original_mode": original_mode,
        "width": original_size[0],
        "height": original_size[1],
        "bit_depth": bit_depth,
        "inversion_mode": mode,
        "processing_time_s": round(t_elapsed, 4),
        "icc_preserved": icc_profile is not None,
    }
    logger.info("Metadata : %s", metadata)
    logger.info("Saved    : %s  (%.3fs)", output_path.name, t_elapsed)

    return str(output_path)


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def invert_batch(
    input_dir: str,
    output_dir: str,
    mode: str = "negative",
    preserve_icc: bool = True,
    suffix: str = "_inverted",
) -> list[str]:
    """
    Invert all supported images in *input_dir* and write results to *output_dir*.

    Output filenames follow the pattern: ``<original_stem><suffix>.png``

    Args:
        input_dir:    Directory containing source images.
        output_dir:   Directory for inverted outputs (created if absent).
        mode:         ``"negative"`` or ``"positive"``.
        preserve_icc: Preserve ICC colour profiles.
        suffix:       String appended to the original filename stem.
                      Default ``"_inverted"``.

    Returns:
        List of absolute paths of all saved output files.

    Raises:
        FileNotFoundError: *input_dir* does not exist.

    Example::

        paths = invert_batch(
            input_dir="evidence/batch/",
            output_dir="forensics/batch_inverted/",
            mode="negative",
        )
        print(f"Processed {len(paths)} images.")
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    candidates = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not candidates:
        logger.warning("No supported images found in %s", input_dir)
        return []

    logger.info("Batch mode: found %d image(s) in %s", len(candidates), input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[str] = []
    errors: list[str] = []

    for src in sorted(candidates):
        dst = output_dir / f"{src.stem}{suffix}.png"
        try:
            out = invert_image(
                input_path=str(src),
                output_path=str(dst),
                mode=mode,
                preserve_icc=preserve_icc,
            )
            results.append(out)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process %s: %s", src.name, exc)
            errors.append(src.name)

    logger.info(
        "Batch complete: %d succeeded, %d failed.",
        len(results),
        len(errors),
    )
    if errors:
        logger.warning("Failed files: %s", errors)

    return results


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invert",
        description=(
            "TrueFrame Lane A — Image Negative/Positive Inversion\n"
            "Invert pixel values to surface forensic tampering artefacts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image:
  python invert.py --input evidence/photo.jpg --output forensics/photo_neg.png

  # Batch folder:
  python invert.py --batch --input evidence/folder/ --output forensics/inverted/

  # Positive mode (invert-of-invert):
  python invert.py --input negative.png --output restored.png --mode positive

  # Strip ICC profile:
  python invert.py --input photo.jpg --output out.png --no-icc
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to a single image file, or a directory when --batch is set.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Output file path (single), or output directory (batch).",
    )
    parser.add_argument(
        "--mode", "-m",
        default="negative",
        choices=["negative", "positive"],
        help="Inversion mode (default: negative).",
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process an entire folder of images.",
    )
    parser.add_argument(
        "--suffix",
        default="_inverted",
        metavar="SUFFIX",
        help="Filename suffix for batch outputs (default: _inverted).",
    )
    parser.add_argument(
        "--no-icc",
        action="store_true",
        dest="strip_icc",
        help="Strip ICC colour profile from output (default: preserve).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    preserve_icc = not args.strip_icc

    if args.batch:
        saved = invert_batch(
            input_dir=args.input,
            output_dir=args.output,
            mode=args.mode,
            preserve_icc=preserve_icc,
            suffix=args.suffix,
        )
        print(f"\n[OK] Batch complete -- {len(saved)} file(s) written to: {args.output}")
    else:
        out_path = invert_image(
            input_path=args.input,
            output_path=args.output,
            mode=args.mode,
            preserve_icc=preserve_icc,
        )
        print(f"\n[OK] Saved: {out_path}")
