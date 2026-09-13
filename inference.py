"""
inference.py — End-to-End Dual-Domain Inference Pipeline for TRUEFRAME.
Combines:
  1. EfficientNet-B0 primary classification (positive domain)
  2. Inverted domain forward pass (negative domain) for signal validation
  3. Dual-Domain Grad-CAM heatmap generation and multiplicative fusion
  4. Quantitative multi-metric spatial agreement (IoU, SSIM, Pearson)
  5. Classical metadata (EXIF/XMP) and artifact (ELA/FFT) forensics
  6. Multi-signal Trust Score fusion with spatial agreement modifier

Single public API:
    analyze_image(image_path) -> dict
"""
import io
import sys
import json
from pathlib import Path
from typing import Union, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image

from gpu_check import require_cuda
from dataset import LABEL_TO_IDX, IDX_TO_LABEL
from train import build_model
from gradcam import (
    generate_dual_heatmaps,
    load_dual_image_tensors,
    fuse_gradcams,
    overlay_heatmap
)
from metadata_forensics import analyze_metadata
from artifact_forensics import analyze_forensic_artifacts
from trust_fusion import (
    compute_trust_score,
    classifier_to_trust_signal,
    FUSION_WEIGHTS,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

# --------------------------------------------------------------------------
# Lazy model loader (singleton)
# --------------------------------------------------------------------------
_MODEL = None
_DEVICE = None


def _get_model_and_device():
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = require_cuda()
        if not CHECKPOINT_PATH.exists():
            print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
            print("  Run train.py first to produce a trained model.")
            sys.exit(1)
        _MODEL = build_model(num_classes=2)
        _MODEL.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=_DEVICE))
        _MODEL.to(_DEVICE)
        _MODEL.eval()
        print(f"[OK] Model loaded from {CHECKPOINT_PATH}")
    return _MODEL, _DEVICE


# --------------------------------------------------------------------------
# Main inference function
# --------------------------------------------------------------------------
def analyze_image(image_path: Union[str, Path, Image.Image]) -> Dict[str, Any]:
    """
    Run full TRUEFRAME Dual-Domain analysis on a single image.

    Args:
        image_path: Absolute path to an image file, PIL Image, or "parquet_path#row_idx".

    Returns:
        {
          "verdict": "Genuine (Real)" | "AI-Generated",
          "confidence": float,                 # softmax probability of primary prediction
          "class_probabilities": dict,         # {label: probability} for normal image
          "inverted_prediction": {             # signal pass on inverted tensor
              "verdict": str,
              "confidence": float,
              "class_probabilities": dict
          },
          "trust_score": int,                  # 0–100 fused trust score
          "agreement_score": float,            # 0.0–1.0 composite spatial consensus
          "agreement_details": dict,           # {hotspot_iou, ssim_score, pearson_corr}
          "heatmap_fused": PIL.Image,          # Multiplicative fused heatmap
          "heatmap_normal": PIL.Image,         # Positive-domain Grad-CAM overlay
          "heatmap_inverted": PIL.Image,       # Negative-domain Grad-CAM overlay
          "heatmap_image": PIL.Image,          # Alias to heatmap_fused
          "metadata_findings": dict,           # from metadata_forensics
          "artifact_findings": dict,           # from artifact_forensics
          "fusion_weights_used": dict,
          "source": str,
        }
    """
    model, device = _get_model_and_device()
    image_str = str(image_path) if not isinstance(image_path, Image.Image) else "in-memory-pil"

    # ---- 1. Load Dual Tensors (Normal + Inverted) ----
    t_normal, pil_normal, t_inverted, pil_inverted = load_dual_image_tensors(image_path)
    t_normal = t_normal.to(device)
    t_inverted = t_inverted.to(device)

    # ---- 2. Primary Classification (Positive Domain) ----
    with torch.no_grad():
        logits_norm = model(t_normal.unsqueeze(0))
        probs_norm = F.softmax(logits_norm, dim=1).squeeze(0).cpu()

    pred_idx = int(probs_norm.argmax())
    pred_label_raw = IDX_TO_LABEL[pred_idx]
    confidence = float(probs_norm[pred_idx])

    verdict_map = {
        "genuine": "Genuine (Real)",
        "ai_generated": "AI-Generated",
    }
    verdict = verdict_map.get(pred_label_raw, "AI-Generated")

    class_probabilities = {
        IDX_TO_LABEL[i]: round(float(probs_norm[i]), 4) for i in range(2)
    }

    # ---- 3. Inverted Pass (Negative Domain) for Signal & Explainability ----
    with torch.no_grad():
        logits_inv = model(t_inverted.unsqueeze(0))
        probs_inv = F.softmax(logits_inv, dim=1).squeeze(0).cpu()

    inv_pred_idx = int(probs_inv.argmax())
    inv_label_raw = IDX_TO_LABEL[inv_pred_idx]
    inv_confidence = float(probs_inv[inv_pred_idx])
    inverted_prediction = {
        "verdict": verdict_map.get(inv_label_raw, "AI-Generated"),
        "confidence": round(inv_confidence, 4),
        "class_probabilities": {
            IDX_TO_LABEL[i]: round(float(probs_inv[i]), 4) for i in range(2)
        }
    }

    # ---- 4. Dual Grad-CAM & Heatmap Fusion ----
    try:
        dual_cam = generate_dual_heatmaps(image_path, model, device, target_class=pred_idx)
        heatmap_normal = dual_cam["heatmap_normal"]
        heatmap_inverted = dual_cam["heatmap_inverted"]
        heatmap_fused = dual_cam["heatmap_fused"]
        agreement_details = dual_cam["agreement_data"]
        agreement_score = agreement_details["agreement_score"]
    except Exception as e:
        print(f"  [WARN] Dual Grad-CAM generation failed: {e}")
        heatmap_normal = None
        heatmap_inverted = None
        heatmap_fused = None
        agreement_score = 1.0
        agreement_details = {
            "agreement_score": 1.0,
            "hotspot_iou": 1.0,
            "ssim_score": 1.0,
            "pearson_corr": 1.0,
            "pearson_norm": 1.0
        }

    # ---- 5. Metadata Forensics (CPU) ----
    try:
        meta_findings = analyze_metadata(image_path)
    except Exception as e:
        meta_findings = {"error": str(e), "metadata_trust_signal": 0.5}

    # ---- 6. Artifact Forensics (CPU: ELA + 2D FFT) ----
    try:
        artifact_findings = analyze_forensic_artifacts(image_path)
    except Exception as e:
        artifact_findings = {"error": str(e), "artifact_trust_signal": 0.5}

    # ---- 7. Multi-Signal Trust Fusion with Spatial Agreement & Inversion Cross-Validation ----
    cls_signal = classifier_to_trust_signal(pred_label_raw, confidence)
    inv_signal = classifier_to_trust_signal(inv_label_raw, inv_confidence)
    meta_signal = meta_findings.get("metadata_trust_signal", 0.5)
    art_signal = artifact_findings.get("artifact_trust_signal", 0.5)

    trust_score = compute_trust_score(
        classifier_confidence=cls_signal,
        metadata_signal=meta_signal,
        artifact_signal=art_signal,
        agreement_score=agreement_score,
        inverted_signal=inv_signal,
        weights=FUSION_WEIGHTS
    )

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probabilities,
        "inverted_prediction": inverted_prediction,
        "trust_score": trust_score,
        "agreement_score": agreement_score,
        "agreement_details": agreement_details,
        "heatmap_fused": heatmap_fused,
        "heatmap_normal": heatmap_normal,
        "heatmap_inverted": heatmap_inverted,
        "heatmap_image": heatmap_fused,  # standard alias
        "metadata_findings": meta_findings,
        "artifact_findings": artifact_findings,
        "fusion_weights_used": FUSION_WEIGHTS,
        "source": image_str,
    }


# --------------------------------------------------------------------------
# Pretty-print helper
# --------------------------------------------------------------------------
def print_analysis(result: dict):
    print("\n" + "=" * 70)
    print(f"  TRUEFRAME Dual-Domain Analysis: {Path(result['source']).name}")
    print("=" * 70)
    print(f"  Primary Verdict:       {result['verdict']}")
    print(f"  Confidence:            {result['confidence']*100:.1f}%")
    print(f"  Trust Score:           {result['trust_score']}/100")
    print(f"  Class Probs:           Genuine={result['class_probabilities']['genuine']*100:.1f}%  "
          f"AI={result['class_probabilities']['ai_generated']*100:.1f}%")
    
    inv = result.get("inverted_prediction", {})
    print(f"  Inverted Pass Verdict: {inv.get('verdict', '?')} ({inv.get('confidence', 0)*100:.1f}%)")

    aggr = result.get("agreement_details", {})
    print(f"  Dual-CAM Agreement:    {result['agreement_score']*100:.1f}%  "
          f"[IoU={aggr.get('hotspot_iou', 0):.2f}, SSIM={aggr.get('ssim_score', 0):.2f}, "
          f"Pearson={aggr.get('pearson_corr', 0):.2f}]")
    
    print(f"  Fused Heatmap:         {'Generated' if result['heatmap_fused'] else 'Failed'}")
    
    meta = result["metadata_findings"]
    print(f"  Metadata Trust:        {meta.get('metadata_trust_signal', '?'):.2f}  "
          f"| has_exif={meta.get('has_exif', '?')}  software={meta.get('software', 'None')}")
    
    art = result["artifact_findings"]
    print(f"  Artifact Trust:        {art.get('artifact_trust_signal', '?'):.2f}  "
          f"| ELA={art.get('ela_mean_score', '?'):.2f}  "
          f"FFT grid={art.get('fft_grid_score', '?'):.2f}")
    
    if meta.get("notes"):
        print(f"  Meta notes:            {meta['notes']}")
    if art.get("flags"):
        print(f"  Artifact flags:        {art['flags']}")
    print("=" * 70)


# --------------------------------------------------------------------------
# CLI Runner
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="TRUEFRAME Dual-Domain Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to image file to test")
    args = parser.parse_args()

    heatmap_dir = PROJECT_ROOT / "outputs" / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"❌ Error: File not found: {img_path}")
            sys.exit(1)
        print(f"\n🔍 Running TRUEFRAME Dual-Domain analysis on: {img_path}")
        result = analyze_image(str(img_path.resolve()))
        print_analysis(result)
        if result["heatmap_fused"] is not None:
            out_fused = heatmap_dir / f"{img_path.stem}_fused_heatmap.png"
            out_norm = heatmap_dir / f"{img_path.stem}_normal_heatmap.png"
            out_inv = heatmap_dir / f"{img_path.stem}_inverted_heatmap.png"
            result["heatmap_fused"].save(str(out_fused))
            result["heatmap_normal"].save(str(out_norm))
            result["heatmap_inverted"].save(str(out_inv))
            print(f"  🔥 Saved Fused Heatmap to:    `{out_fused}`")
            print(f"  🔥 Saved Normal Heatmap to:   `{out_norm}`")
            print(f"  🔥 Saved Inverted Heatmap to: `{out_inv}`\n")
    else:
        test_manifest = PROJECT_ROOT / "manifest_test.csv"
        if not test_manifest.exists():
            print(f"ERROR: Test manifest not found: {test_manifest}")
            sys.exit(1)

        df = pd.read_csv(test_manifest)
        print("\n=== TRUEFRAME Dual-Domain Inference Test ===")
        print(f"Test manifest: {len(df):,} samples\n")

        test_images = []
        for label in ["genuine", "ai_generated"]:
            subset = df[(df["label"] == label) & (~df["filepath"].str.contains("#"))]
            if len(subset) == 0:
                subset = df[df["label"] == label]
            sample = subset.sample(min(2, len(subset)), random_state=7)
            test_images.extend(sample["filepath"].tolist())

        print(f"Running inference on {len(test_images)} test samples...\n")

        for fp in test_images:
            try:
                result = analyze_image(fp)
                print_analysis(result)
                if result["heatmap_fused"] is not None:
                    fname = Path(fp).stem + "_fused_heatmap.png"
                    result["heatmap_fused"].save(str(heatmap_dir / fname))
            except Exception as e:
                print(f"\n[ERROR] Failed on {fp}: {e}")
