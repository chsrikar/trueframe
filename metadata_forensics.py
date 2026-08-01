"""
metadata_forensics.py — EXIF / XMP metadata analysis for TRUEFRAME pipeline.
CPU-only module — no GPU required.

Works on:
  - Regular image file paths
  - Parquet-backed images (extracts bytes first, same as dataset.py)
"""
import io
import re
import struct
from pathlib import Path
from typing import Union

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Software signatures that indicate AI generation or editing tools
_AI_GENERATORS = [
    "stable diffusion", "midjourney", "dall-e", "dalle", "firefly",
    "imagen", "generative fill", "ai generated", "nightcafe", "dream",
    "runway", "novelai", "invokeai", "automatic1111", "comfyui",
]
_EDITING_SOFTWARE = [
    "photoshop", "lightroom", "gimp", "affinity", "capture one",
    "darktable", "rawtherapee", "luminar",
]


def _load_pil_from_source(image_source: Union[str, Path]) -> Image.Image:
    """Load a PIL Image from a file path or parquet#row reference."""
    path_str = str(image_source)
    if "#" in path_str:
        from dataset import get_parquet_image_bytes
        parquet_path, idx_str = path_str.split("#")
        img_bytes = get_parquet_image_bytes(parquet_path, int(idx_str))
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        return Image.open(path_str)


def _parse_gps(gps_data: dict) -> dict:
    """Convert raw GPS EXIF data to human-readable dict."""
    result = {}
    if not gps_data:
        return result
    for key, val in gps_data.items():
        tag = GPSTAGS.get(key, str(key))
        result[tag] = str(val)
    return result


def _detect_software_type(software_str: str) -> str:
    """Classify software string as 'ai_generator', 'editor', or 'camera'."""
    low = software_str.lower()
    for sig in _AI_GENERATORS:
        if sig in low:
            return "ai_generator"
    for sig in _EDITING_SOFTWARE:
        if sig in low:
            return "editor"
    return "camera_or_unknown"


def _parse_xmp_history(xmp_data: bytes) -> list[str]:
    """Extract XMP history action entries from raw XMP bytes."""
    if not xmp_data:
        return []
    try:
        text = xmp_data.decode("utf-8", errors="replace")
        # Look for stEvt:action values in XMP history
        actions = re.findall(r'stEvt:action="([^"]+)"', text)
        if not actions:
            # Also match <stEvt:action>...</stEvt:action>
            actions = re.findall(r'<stEvt:action>([^<]+)<', text)
        return actions
    except Exception:
        return []


def analyze_metadata(image_source: Union[str, Path]) -> dict:
    """
    Analyze image metadata (EXIF, XMP) for authenticity signals.

    Returns a dict with:
      - has_exif: bool
      - exif_fields_present: list of present EXIF tag names
      - camera_make, camera_model: str or None
      - datetime_original: str or None
      - has_gps: bool
      - gps_data: dict
      - software: str or None
      - software_type: 'ai_generator' | 'editor' | 'camera_or_unknown' | None
      - xmp_history: list of XMP history actions
      - ai_generation_signals: list of detected AI/edit signal strings
      - metadata_trust_signal: float in [0, 1]
          (1.0 = looks authentic; 0.0 = strong manipulation/AI signals)
      - notes: list of human-readable observations
    """
    result = {
        "has_exif": False,
        "exif_fields_present": [],
        "camera_make": None,
        "camera_model": None,
        "datetime_original": None,
        "has_gps": False,
        "gps_data": {},
        "software": None,
        "software_type": None,
        "xmp_history": [],
        "ai_generation_signals": [],
        "metadata_trust_signal": 0.5,
        "notes": [],
        "source": str(image_source),
    }

    path_str = str(image_source)
    is_parquet = "#" in path_str

    # Parquet-stored images typically have no EXIF (expected finding)
    if is_parquet:
        result["notes"].append(
            "Parquet-stored image — EXIF data is typically stripped at dataset creation."
        )
        result["metadata_trust_signal"] = 0.5  # neutral — no signal either way
        return result

    try:
        img = _load_pil_from_source(image_source)
    except Exception as e:
        result["notes"].append(f"Failed to open image: {e}")
        return result

    # ---- EXIF extraction ----
    try:
        exif_raw = img._getexif()  # Returns None if no EXIF
    except AttributeError:
        exif_raw = None

    if exif_raw:
        result["has_exif"] = True
        exif_decoded = {}
        for tag_id, value in exif_raw.items():
            tag = TAGS.get(tag_id, str(tag_id))
            exif_decoded[tag] = value

        present_tags = list(exif_decoded.keys())
        result["exif_fields_present"] = present_tags

        result["camera_make"] = exif_decoded.get("Make")
        result["camera_model"] = exif_decoded.get("Model")
        result["datetime_original"] = str(exif_decoded.get("DateTimeOriginal", ""))

        software_raw = exif_decoded.get("Software", "")
        if software_raw:
            result["software"] = str(software_raw).strip()
            result["software_type"] = _detect_software_type(result["software"])
            if result["software_type"] == "ai_generator":
                result["ai_generation_signals"].append(f"Software tag: {result['software']}")
            elif result["software_type"] == "editor":
                result["ai_generation_signals"].append(f"Editing software: {result['software']}")

        # GPS
        gps_info = exif_decoded.get("GPSInfo", {})
        if gps_info:
            result["has_gps"] = True
            result["gps_data"] = _parse_gps(gps_info)

    else:
        # Check if this looks like a social-media or messaging-app image
        _fname_low = Path(str(image_source)).name.lower()
        _social_prefixes = ("whatsapp", "telegram", "instagram", "signal",
                            "messenger", "snapchat", "twitter", "fb_img",
                            "screenshot", "img-", "img_")
        _is_social = any(_fname_low.startswith(p) for p in _social_prefixes)
        if _is_social:
            result["notes"].append(
                "No EXIF data — image sent via a messaging/social app "
                "(WhatsApp, Telegram, etc. strip EXIF for privacy). "
                "Absence of EXIF is expected and not a suspicious signal here."
            )
        else:
            result["notes"].append(
                "No EXIF data found — image may be AI-generated, screenshot, or EXIF-stripped."
            )

    # ---- XMP extraction ----
    try:
        if hasattr(img, "applist"):
            for marker, data in img.applist:
                if marker == "APP1" and b"http://ns.adobe.com/xap" in data:
                    result["xmp_history"] = _parse_xmp_history(data)
                    break
    except Exception:
        pass

    # ---- Compute trust signal ----
    trust = 0.5  # neutral baseline

    # Positive signals (camera-like metadata)
    if result["camera_make"]:
        trust += 0.15
    if result["camera_model"]:
        trust += 0.10
    if result["datetime_original"]:
        trust += 0.05
    if result["has_gps"]:
        trust += 0.05

    # Negative signals
    if result["software_type"] == "ai_generator":
        trust -= 0.40
        result["notes"].append("⚠️ AI generator software signature detected in EXIF.")
    elif result["software_type"] == "editor":
        trust -= 0.15
        result["notes"].append("ℹ️ Image editing software found in EXIF (possible manipulation).")

    if not result["has_exif"]:
        # Social/messaging apps (WhatsApp, Telegram, etc.) strip EXIF by design.
        # Don't penalise trust for expected behaviour.
        _fname_low2 = Path(str(image_source)).name.lower()
        _is_social2 = any(_fname_low2.startswith(p) for p in (
            "whatsapp", "telegram", "instagram", "signal",
            "messenger", "snapchat", "twitter", "fb_img",
            "screenshot", "img-", "img_"
        ))
        if not _is_social2:
            trust -= 0.10  # slight negative: genuine photos usually have EXIF

    if result["xmp_history"]:
        result["notes"].append(f"XMP history actions: {result['xmp_history']}")
        if any("save" in a.lower() or "export" in a.lower() for a in result["xmp_history"]):
            trust -= 0.05

    result["metadata_trust_signal"] = max(0.0, min(1.0, trust))

    # ---- Summary note ----
    if result["metadata_trust_signal"] >= 0.7:
        result["notes"].append("Metadata profile consistent with a genuine camera photo.")
    elif result["metadata_trust_signal"] >= 0.4:
        result["notes"].append("Metadata profile neutral — inconclusive signal.")
    else:
        result["notes"].append("Metadata profile suggests AI generation or significant editing.")

    return result


# --------------------------------------------------------------------------
# CLI test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json
    import pandas as pd

    test_manifest = Path(r"D:\demo\manifest_test.csv")
    if not test_manifest.exists():
        print(f"❌ Test manifest not found: {test_manifest}")
        sys.exit(1)

    df = pd.read_csv(test_manifest)
    print("=" * 70)
    print("   METADATA FORENSICS — Sample Analysis")
    print("=" * 70)

    # 1 sample per label
    for label in ["genuine", "ai_generated", "manipulated"]:
        subset = df[(df["label"] == label) & (~df["filepath"].str.contains("#"))]
        if len(subset) == 0:
            print(f"\n[{label}] No non-parquet samples found.")
            continue
        row = subset.sample(1, random_state=42).iloc[0]
        print(f"\n[{label.upper()}] {Path(row['filepath']).name}")
        findings = analyze_metadata(row["filepath"])
        print(f"  has_exif:            {findings['has_exif']}")
        print(f"  camera:              {findings['camera_make']} {findings['camera_model']}")
        print(f"  software:            {findings['software']} ({findings['software_type']})")
        print(f"  datetime_original:   {findings['datetime_original']}")
        print(f"  ai_signals:          {findings['ai_generation_signals']}")
        print(f"  metadata_trust:      {findings['metadata_trust_signal']:.2f}")
        print(f"  notes:               {findings['notes']}")
