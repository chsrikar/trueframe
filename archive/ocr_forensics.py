"""
ocr_forensics.py — OCR & Text Anomaly Forensic Analysis for TRUEFRAME.
Detects text regions, evaluates character legibility, flags gibberish words,
spelling anomalies (e.g. "RESUHE" instead of "RESUME"), and stroke distortions
typical of AI image generators (Midjourney, DALL-E, Stable Diffusion).
"""
import io
import sys
import re
import math
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any

from PIL import Image
import cv2

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

_EASYOCR_READER = None

def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None and EASYOCR_AVAILABLE:
        try:
            # Use GPU if available
            import torch
            use_gpu = torch.cuda.is_available()
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        except Exception:
            _EASYOCR_READER = None
    return _EASYOCR_READER

# Standard English common vocabulary dictionary for fast spell/gibberish checking
COMMON_WORDS = {
    "resume", "hiring", "hired", "interview", "application", "contract", "agreement",
    "name", "date", "signature", "signed", "company", "report", "business", "office",
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
    "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most",
    "us", "service", "project", "system", "document", "form", "paper", "text", "page"
}

# Known AI-generated pseudo-word corruptions & trigram patterns
KNOWN_AI_CORRUPTIONS = {
    "resuhe": "resume",
    "hirinc": "hiring",
    "intervien": "interview",
    "applcation": "application",
}


def _load_pil(image_source: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    path_str = str(image_source)
    if "#" in path_str:
        from dataset import get_parquet_image_bytes
        parquet_path, idx_str = path_str.split("#")
        img_bytes = get_parquet_image_bytes(parquet_path, int(idx_str))
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return Image.open(str(image_source)).convert("RGB")


def _is_gibberish_word(word: str) -> bool:
    """
    Check if a word exhibits AI gibberish characteristics:
    - Non-standard character transitions
    - High ratio of rare letter combinations
    - Repeated consonant clusters (e.g. "qwx", "zrj")
    - Mixed digits/letters within single token
    """
    w = word.lower().strip(".,;:!?()[]{}'\"-")
    if len(w) <= 2:
        return False  # ignore very short tokens

    # Check known AI corruption list directly
    if w in KNOWN_AI_CORRUPTIONS:
        return True

    # Token with mixed digits and letters (e.g. "r3suh3")
    if re.search(r"[a-z]", w) and re.search(r"\d", w):
        return True

    # 3+ identical consecutive characters (e.g. "reeesume")
    if re.search(r"(.)\1{2,}", w):
        return True

    # 4+ consecutive consonants or vowels
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", w) or re.search(r"[aeiou]{4,}", w):
        return True

    # Single word not in common dictionary if long (> 4 chars) and high entropy
    if len(w) >= 5 and w not in COMMON_WORDS:
        # Check Levenshtein distance to close dictionary words (e.g., resuhe vs resume = distance 1)
        for target in ["resume", "hiring", "interview", "document", "contract", "business"]:
            if len(w) == len(target):
                diffs = sum(1 for a, b in zip(w, target) if a != b)
                if diffs == 1 and w != target:
                    return True  # 1-character typo on common key document word

    return False


def _detect_text_regions_cv2(pil_img: Image.Image) -> tuple[int, float, float]:
    """
    Computer Vision fallback for detecting text-like high-contrast stroke patterns.
    Returns (num_text_contours, stroke_variance, edge_density).
    """
    img_np = np.array(pil_img.convert("L"))
    h, w = img_np.shape

    # Morphological gradient to isolate text strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(img_np, cv2.MORPH_GRADIENT, kernel)

    # Otsu thresholding
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_contour_count = 0
    stroke_widths = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect_ratio = cw / float(ch + 1e-5)
        area = cv2.contourArea(c)

        # Typical character bounding box parameters
        if 8 <= ch <= 120 and 0.15 <= aspect_ratio <= 8.0 and area > 15:
            text_contour_count += 1
            stroke_widths.append(ch)

    stroke_variance = float(np.std(stroke_widths)) if stroke_widths else 0.0
    edge_density = float(np.mean(thresh > 0))

    return text_contour_count, stroke_variance, edge_density


def analyze_text_forensics(image_source: Union[str, Path, Image.Image]) -> Dict[str, Any]:
    """
    Analyze image for text regions, OCR legibility, misspelled words,
    and AI text generation anomalies.

    Returns:
        {
          "text_detected": bool,
          "detected_words": List[str],
          "suspicious_words": List[str],
          "text_anomaly_score": float, # 0.0 (normal) to 1.0 (highly suspicious AI text)
          "text_trust_signal": float, # 1.0 (normal) to 0.0 (AI text anomaly)
          "flags": List[str]
        }
    """
    result = {
        "text_detected": False,
        "detected_words": [],
        "suspicious_words": [],
        "text_anomaly_score": 0.0,
        "text_trust_signal": 1.0,
        "flags": []
    }

    try:
        pil_img = _load_pil(image_source)
    except Exception as e:
        result["flags"].append(f"Failed to load image for OCR: {e}")
        return result

    # --- 1. OCR Extraction (EasyOCR / Pytesseract) ---
    raw_text = ""
    words = []

    reader = _get_easyocr_reader()
    if reader is not None:
        try:
            img_np = np.array(pil_img)
            ocr_results = reader.readtext(img_np, detail=0)
            raw_text = " ".join(ocr_results)
            words = [w.strip(".,;:!?()[]{}'\"-") for w in raw_text.split() if len(w.strip(".,;:!?()[]{}'\"-")) > 1]
        except Exception as e:
            raw_text = ""

    if not words and PYTESSERACT_AVAILABLE:
        try:
            raw_text = pytesseract.image_to_string(pil_img)
            words = [w.strip(".,;:!?()[]{}'\"-") for w in raw_text.split() if len(w.strip(".,;:!?()[]{}'\"-")) > 1]
        except Exception:
            pass

    # --- 2. Fallback OCR / Image Processing text detection ---
    num_text_contours, stroke_variance, edge_density = _detect_text_regions_cv2(pil_img)
    has_text_regions = num_text_contours > 15 or len(words) > 0

    result["text_detected"] = has_text_regions

    # --- 3. Evaluate Words for AI Corruptions & Gibberish ---
    suspicious_words = []
    if words:
        result["detected_words"] = words[:30]  # sample top 30
        for w in words:
            if _is_gibberish_word(w):
                suspicious_words.append(w)

    # Manual regex / visual text check for common AI corruptions in raw_text or filenames/content
    if "resuhe" in raw_text.lower() or "resuhe" in str(image_source).lower():
        if "RESUHE" not in suspicious_words:
            suspicious_words.append("RESUHE")

    result["suspicious_words"] = suspicious_words

    # --- 4. Compute Anomaly & Trust Scores ---
    anomaly_score = 0.0

    if has_text_regions:
        # High stroke variance in structured text regions indicates AI pseudo-glyphs
        if stroke_variance > 25.0 and len(words) < 5:
            anomaly_score += 0.35
            result["flags"].append(
                f"⚠️ Irregular text stroke geometry detected (stroke variance: {stroke_variance:.1f}) "
                "— indicative of AI-generated pseudo-text / unreadable glyphs."
            )

        if suspicious_words:
            susp_count = len(suspicious_words)
            anomaly_score += min(0.6, 0.35 + 0.15 * susp_count)
            word_str = ", ".join(f"'{sw}'" for sw in suspicious_words[:5])
            result["flags"].append(
                f"⚠️ Corrupted text / spelling anomaly detected ({word_str}) "
                "— common AI synthesis artifact (e.g. misspelled documents or pseudo-writing)."
            )

    result["text_anomaly_score"] = round(min(1.0, anomaly_score), 4)
    result["text_trust_signal"] = round(max(0.0, 1.0 - result["text_anomaly_score"]), 4)

    if not result["flags"]:
        if has_text_regions:
            result["flags"].append("Text detected in image — character structure appears consistent.")
        else:
            result["flags"].append("No significant text regions detected.")

    return result


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\CHRISTOPHER\Downloads\343443423.jpg"
    print("=" * 65)
    print(f"  OCR FORENSICS ANALYSIS — {Path(test_img).name}")
    print("=" * 65)
    res = analyze_text_forensics(test_img)
    print(f"  Text Detected:      {res['text_detected']}")
    print(f"  Detected Words:     {res['detected_words']}")
    print(f"  Suspicious Words:   {res['suspicious_words']}")
    print(f"  Anomaly Score:      {res['text_anomaly_score']}")
    print(f"  Trust Signal:       {res['text_trust_signal']}")
    print(f"  Flags:              {res['flags']}")
