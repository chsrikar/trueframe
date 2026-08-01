# TrueFrame Lane A — Image Inversion Module

Pillow · NumPy · Python 3.9+

---

## Overview

Standalone forensic module for the **TrueFrame** image-analysis pipeline (Lane A).

Performs **negative ↔ positive pixel-value inversion** as a preprocessing / analysis step to surface tampering artefacts — splicing boundaries, cloned regions, inconsistent noise / compression patterns — that are invisible in the original colour/luminance space but become apparent after inversion.

---

## Directory structure

```
trueframe_invert/
├── invert.py          ← Core module (single deliverable)
├── tests/
│   └── test_invert.py ← Unit + integration tests
├── requirements.txt
└── README.md
```

---

## Quick start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Single image (CLI)

```bash
python invert.py --input evidence/photo.jpg --output forensics/photo_neg.png
```

### 3 · Batch folder

```bash
python invert.py --batch \
    --input  evidence/batch/ \
    --output forensics/inverted/ \
    --suffix _neg
```

### 4 · Python API

```python
from invert import invert_image, invert_batch

# Single image
out = invert_image(
    input_path="evidence/photo.jpg",
    output_path="forensics/photo_neg.png",
    mode="negative",          # or "positive"
    preserve_icc=True,        # keep embedded colour profile
)
print(f"Saved → {out}")

# Batch
results = invert_batch(
    input_dir="evidence/batch/",
    output_dir="forensics/inverted/",
    mode="negative",
    suffix="_inverted",
)
print(f"Processed {len(results)} image(s)")
```

---

## CLI reference

```
usage: invert.py [-h] --input PATH --output PATH [--mode {negative,positive}]
                 [--batch] [--suffix SUFFIX] [--no-icc] [--verbose]

Options:
  --input   / -i   Source image file or directory (with --batch)
  --output  / -o   Destination file or directory (with --batch)
  --mode    / -m   negative (default) | positive
  --batch   / -b   Process an entire folder
  --suffix         Filename suffix for batch outputs (default: _inverted)
  --no-icc         Strip ICC colour profile from output
  --verbose / -v   Enable DEBUG logging
```

---

## Inversion logic

| Bit depth | Formula |
|-----------|---------|
| 8-bit (`uint8`)  | `255 - pixel` |
| 16-bit (`uint16`) | `65535 - pixel` |

- **Alpha channel** (RGBA / LA) is **preserved untouched**.
- Uses NumPy vectorised operations — no pixel-by-pixel loops.
- For 8-bit RGB paths, `ImageOps.invert()` can be used directly (equivalent).

---

## Output format recommendation

| Use case | Recommended format |
|----------|--------------------|
| Intermediate forensic step | **PNG** (lossless — no re-compression artefacts) |
| Final deliverable / report | JPEG or TIFF (caller's choice) |

> **Warning** Saving an inverted image as JPEG introduces re-compression artefacts that may distort the forensic signals you are trying to detect. Prefer PNG for intermediate outputs.

---

## Supported formats

JPEG · PNG · BMP · TIFF · WEBP

---

## Edge cases handled

| Scenario | Behaviour |
|----------|-----------|
| RGBA / LA (transparency) | Alpha channel copied unchanged |
| Embedded ICC colour profile | Preserved by default (`preserve_icc=True`) |
| 16-bit images | Uses `65535 - pixel` |
| Corrupt / unreadable file | Raises `UnidentifiedImageError` |
| Non-existent input | Raises `FileNotFoundError` |
| Invalid mode string | Raises `ValueError` |
| Output directory absent | Created automatically (`parents=True`) |
| Batch with corrupt file | Logs error, continues remaining images |

---

## Metadata logged per image

```
input_path, output_path, original_format, original_mode,
width, height, bit_depth, inversion_mode,
processing_time_s, icc_preserved
```

Captured as a Python `dict` via the standard `logging` module. Pipe to Lane B's report generator as needed.

---

## Running tests

```bash
# With pytest (recommended)
pip install pytest
python -m pytest tests/test_invert.py -v

# Or with the built-in runner
python tests/test_invert.py
```

### Test coverage

| Class | Description |
|-------|-------------|
| `TestInvertArray` | `_invert_array()` — exact pixel values, alpha preservation, uint16, round-trip, dtype guard |
| `TestInvertImage` | `invert_image()` — RGB, greyscale, RGBA, PNG round-trip, mode parity, return value, dir creation, error paths |
| `TestInvertBatch` | `invert_batch()` — count, naming, empty dir, missing dir, non-image skip, pixel correctness |

---

## Integration with Lane A

```
Input image
    │
    ▼
[ invert_image() ]   ←── this module
    │
    ├──► inverted PNG → Grad-CAM overlay
    ├──► inverted PNG → Classifier preprocessing
    └──► metadata dict → Trust Fusion / Lane B report
```

Call `invert_image()` as a preprocessing step before any other Lane A module.
Pass the returned `output_path` as the input to the next module in the chain.

---

## Library choice rationale

This module uses **Pillow + NumPy** to match the rest of Lane A (classifier preprocessing, Grad-CAM).
If OpenCV (`cv2`) is introduced, remember that it uses **BGR** channel order vs. Pillow's **RGB** — always convert explicitly with `cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)` before passing arrays between libraries.

---

## Dependencies

```
Pillow>=10.0
numpy>=1.24
```

Optional (if matching an OpenCV-based pipeline):
```
opencv-python>=4.8
```
