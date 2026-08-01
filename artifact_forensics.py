"""
artifact_forensics.py — ELA + FFT forensic artifact analysis for TRUEFRAME.
CPU-only module — no GPU required.

ELA (Error Level Analysis): Re-encodes at JPEG quality 75 and measures
  pixel-level differences. High, non-uniform ELA scores indicate regions
  that were edited after the original save — common in CASIA Tp/ images.

FFT Analysis: Examines the frequency spectrum of the grayscale image.
  AI-generated images (especially GAN-based) often show periodic grid
  artifacts in the frequency domain not seen in real photographs.
"""
import io
import sys
import numpy as np
from pathlib import Path
from typing import Union

from PIL import Image, ImageChops, ImageEnhance


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
def _load_pil(image_source: Union[str, Path]) -> Image.Image:
    path_str = str(image_source)
    if "#" in path_str:
        from dataset import get_parquet_image_bytes
        parquet_path, idx_str = path_str.split("#")
        img_bytes = get_parquet_image_bytes(parquet_path, int(idx_str))
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return Image.open(str(image_source)).convert("RGB")


# --------------------------------------------------------------------------
# ELA
# --------------------------------------------------------------------------
def _run_ela(pil_img: Image.Image, quality: int = 75, scale: float = 10.0) -> tuple[Image.Image, float]:
    """
    Perform Error Level Analysis.

    Returns:
        ela_image: PIL.Image showing the amplified error map (for visualization).
        mean_ela_score: float — mean pixel intensity of the ELA difference map.
            Higher values indicate more compression artifacts / potential editing.
    """
    # Re-encode to JPEG at reduced quality in memory
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # Compute absolute difference
    diff = ImageChops.difference(pil_img, recompressed)

    # Amplify for visualization
    diff_arr = np.array(diff, dtype=np.float32)
    mean_score = float(diff_arr.mean())

    # Scale for visualization
    scaled = np.clip(diff_arr * scale, 0, 255).astype(np.uint8)
    ela_img = Image.fromarray(scaled, mode="RGB")

    return ela_img, mean_score


def _ela_region_std(pil_img: Image.Image, quality: int = 75) -> float:
    """
    Standard deviation of ELA scores across 16×16 regions.
    High std indicates non-uniform editing (manipulation).
    Low std = globally consistent compression (likely genuine or fully AI).
    """
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    diff_arr = np.array(ImageChops.difference(pil_img, recompressed), dtype=np.float32)
    h, w, _ = diff_arr.shape

    region_means = []
    block = 16
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = diff_arr[y:y+block, x:x+block]
            region_means.append(float(patch.mean()))

    return float(np.std(region_means)) if region_means else 0.0


# --------------------------------------------------------------------------
# FFT Analysis
# --------------------------------------------------------------------------
def _run_fft(pil_img: Image.Image) -> dict:
    """
    Analyze frequency domain for periodic GAN/AI fingerprint artifacts.

    Returns a dict with:
      - peak_ratio: ratio of top spectral peaks to mean spectral energy.
          High values suggest periodic patterns (GAN-typical).
      - grid_artifact_score: heuristic strength of grid-like frequency peaks.
      - fft_energy_high_freq: fraction of total energy in high-frequency bins.
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    # 2D FFT — shift zero-frequency component to center
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Log-magnitude spectrum (suppress DC spike)
    log_mag = np.log1p(magnitude)

    # Remove DC spike at center (replace with local mean to avoid bias)
    cy, cx = h // 2, w // 2
    r = max(3, min(h, w) // 50)
    dc_region = log_mag.copy()
    dc_region[cy-r:cy+r, cx-r:cx+r] = log_mag.mean()

    mean_val = float(dc_region.mean())
    std_val = float(dc_region.std())

    # Top-N spectral peaks relative to mean
    flat = dc_region.flatten()
    top_k = 20
    top_vals = np.partition(flat, -top_k)[-top_k:]
    peak_ratio = float(top_vals.mean() / (mean_val + 1e-8))

    # High-frequency energy (outer ring of spectrum)
    cy_f, cx_f = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy_f)**2 + (X - cx_f)**2)
    max_dist = np.sqrt(cy_f**2 + cx_f**2)
    high_freq_mask = dist > 0.4 * max_dist
    total_energy = float(log_mag.sum()) + 1e-8
    hf_energy = float(log_mag[high_freq_mask].sum()) / total_energy

    # Grid artifact detection: look for horizontal/vertical line peaks in FFT
    row_spectrum = log_mag.sum(axis=1)   # sum along columns → row profile
    col_spectrum = log_mag.sum(axis=0)   # sum along rows → col profile

    def _grid_score(spectrum):
        spectrum = spectrum - spectrum.mean()
        spectrum[len(spectrum)//2 - 2: len(spectrum)//2 + 2] = 0  # mask DC
        if spectrum.max() < 1e-8:
            return 0.0
        return float(spectrum.max() / (spectrum.std() + 1e-8))

    grid_score = (_grid_score(row_spectrum) + _grid_score(col_spectrum)) / 2.0

    return {
        "peak_ratio": round(peak_ratio, 4),
        "grid_artifact_score": round(grid_score, 4),
        "fft_energy_high_freq": round(hf_energy, 4),
        "fft_mean": round(mean_val, 4),
        "fft_std": round(std_val, 4),
    }


# --------------------------------------------------------------------------
# Main API
# --------------------------------------------------------------------------
def analyze_forensic_artifacts(image_source: Union[str, Path]) -> dict:
    """
    Run ELA + FFT forensic analysis on an image.

    Args:
        image_source: File path or parquet#row reference.

    Returns a dict with:
      - ela_mean_score: float — mean ELA pixel diff (higher = more editing)
      - ela_region_std: float — spatial variation of ELA (high = localized edits)
      - fft_peak_ratio: float — spectral peak prominence (high = GAN artifacts)
      - fft_grid_score: float — grid-like frequency artifacts (high = AI/GAN)
      - fft_high_freq_energy: float — fraction of high-frequency energy
      - artifact_trust_signal: float in [0, 1]
          (1.0 = looks unmanipulated; 0.0 = strong artifact signals)
      - flags: list of human-readable artifact flags raised
      - source: str — the image_source passed in
    """
    result = {
        "ela_mean_score": 0.0,
        "ela_region_std": 0.0,
        "fft_peak_ratio": 0.0,
        "fft_grid_score": 0.0,
        "fft_high_freq_energy": 0.0,
        "artifact_trust_signal": 0.5,
        "flags": [],
        "source": str(image_source),
    }

    try:
        pil_img = _load_pil(image_source)
    except Exception as e:
        result["flags"].append(f"Failed to load image: {e}")
        return result

    # --- Grayscale / B&W detection (OOD for this model) ---
    img_arr = np.array(pil_img, dtype=np.float32)
    channel_std = float(np.std(img_arr[:, :, 0] - img_arr[:, :, 1]))
    is_grayscale = channel_std < 2.0  # R≈G≈B across the whole image
    if is_grayscale:
        result["flags"].append(
            "⚠️ Image appears to be black-and-white or heavily desaturated. "
            "This model was trained on colour images — verdict reliability is REDUCED. "
            "B&W AI portraits (Midjourney/Flux) are a known false-negative."
        )
        trust = result["artifact_trust_signal"]  # will be set below

    # --- Detect if source is a lossless format (PNG, BMP, TIFF) ---
    _ext = Path(str(image_source)).suffix.lower()
    _is_lossless = _ext in (".png", ".bmp", ".tiff", ".tif", ".webp")

    # --- ELA ---
    try:
        _, ela_mean = _run_ela(pil_img, quality=75)
        ela_std = _ela_region_std(pil_img, quality=75)
        result["ela_mean_score"] = round(ela_mean, 4)
        result["ela_region_std"] = round(ela_std, 4)
    except Exception as e:
        result["flags"].append(f"ELA failed: {e}")

    # --- FFT ---
    try:
        fft_data = _run_fft(pil_img)
        result["fft_peak_ratio"] = fft_data["peak_ratio"]
        result["fft_grid_score"] = fft_data["grid_artifact_score"]
        result["fft_high_freq_energy"] = fft_data["fft_energy_high_freq"]
    except Exception as e:
        result["flags"].append(f"FFT failed: {e}")

    # --- Compute artifact trust signal ---
    trust = 0.7  # start slightly positive (unmanipulated is the base case)

    # ELA signals — use tighter thresholds for lossless formats (PNG/TIFF/BMP)
    # because lossless images have no inherent compression noise floor.
    ela_high_thresh  = 1.0 if _is_lossless else 8.0   # lossless: 1.0 is already suspicious
    ela_warn_thresh  = 0.5 if _is_lossless else 4.0

    if result["ela_mean_score"] > ela_high_thresh:
        trust -= 0.25
        result["flags"].append(
            f"⚠️ Elevated ELA score ({result['ela_mean_score']:.2f}) on a lossless image "
            f"— genuine lossless photos are near-zero; this suggests prior JPEG encoding "
            f"or pixel-level synthesis (AI generation / manipulation)."
            if _is_lossless else
            f"⚠️ High ELA mean score ({result['ela_mean_score']:.2f}) — possible re-editing or manipulation."
        )
    if result["ela_region_std"] > ela_warn_thresh:
        trust -= 0.15
        result["flags"].append(f"⚠️ High ELA spatial variation ({result['ela_region_std']:.2f}) — localized edits detected.")

    # FFT signals
    if result["fft_grid_score"] > 8.0:
        trust -= 0.20
        result["flags"].append(f"⚠️ FFT grid artifact score {result['fft_grid_score']:.2f} — possible GAN periodic pattern.")
    if result["fft_peak_ratio"] > 3.0:
        trust -= 0.10
        result["flags"].append(f"ℹ️ FFT peak ratio {result['fft_peak_ratio']:.2f} — elevated spectral peaks.")

    # B&W OOD penalty — model confidence is unreliable on greyscale
    if is_grayscale:
        trust -= 0.15

    # Combined AI-portrait suspicion signal:
    # Lossless PNG + no EXIF + elevated ELA + elevated FFT = strong AI indicator
    if (_is_lossless
            and result["ela_mean_score"] > 0.8
            and result["fft_peak_ratio"] > 3.0):
        trust -= 0.10
        result["flags"].append(
            "⚠️ Combined signal: lossless PNG with no camera EXIF, elevated ELA, "
            "and elevated FFT peaks — pattern consistent with AI-generated photorealistic portrait. "
            "Model verdict may be unreliable."
        )

    result["artifact_trust_signal"] = max(0.0, min(1.0, trust))

    if not result["flags"]:
        result["flags"].append("No strong artifact signals detected.")

    return result


# --------------------------------------------------------------------------
# CLI test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import pandas as pd

    test_manifest = Path(r"D:\demo\manifest_test.csv")
    if not test_manifest.exists():
        print(f"❌ Test manifest not found: {test_manifest}")
        sys.exit(1)

    df = pd.read_csv(test_manifest)
    print("=" * 70)
    print("   FORENSIC ARTIFACT ANALYSIS — Sample Results")
    print("=" * 70)

    # Test on CASIA Tp (manipulated) and AI-generated samples
    for label in ["manipulated", "ai_generated", "genuine"]:
        subset = df[(df["label"] == label) & (~df["filepath"].str.contains("#"))]
        if len(subset) == 0:
            print(f"\n[{label}] No non-parquet samples found.")
            continue
        row = subset.sample(1, random_state=99).iloc[0]
        print(f"\n[{label.upper()}] {Path(row['filepath']).name}")
        findings = analyze_forensic_artifacts(row["filepath"])
        print(f"  ELA mean score:      {findings['ela_mean_score']:.3f}")
        print(f"  ELA region std:      {findings['ela_region_std']:.3f}")
        print(f"  FFT peak ratio:      {findings['fft_peak_ratio']:.3f}")
        print(f"  FFT grid score:      {findings['fft_grid_score']:.3f}")
        print(f"  FFT HF energy:       {findings['fft_high_freq_energy']:.3f}")
        print(f"  artifact trust:      {findings['artifact_trust_signal']:.2f}")
        print(f"  flags:               {findings['flags']}")
