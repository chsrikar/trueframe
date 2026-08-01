"""
inference.py — End-to-end inference for TRUEFRAME pipeline.
Requires GPU (calls require_cuda()).

Single public API:
    analyze_image(image_path) -> dict
"""
import io
import sys
import json
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image

from gpu_check import require_cuda
from dataset import LABEL_TO_IDX, IDX_TO_LABEL, get_parquet_image_bytes
from train import build_model
from gradcam import generate_heatmap, load_image_tensor
from metadata_forensics import analyze_metadata
from artifact_forensics import analyze_forensic_artifacts
from trust_fusion import (
    compute_trust_score,
    classifier_to_trust_signal,
    FUSION_WEIGHTS,
)

sys.stdout.reconfigure(encoding='utf-8')

CHECKPOINT_PATH = Path(r"D:\demo\checkpoints\best_model.pth")

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
def analyze_image(image_path: Union[str, Path]) -> dict:
    """
    Run full TRUEFRAME analysis on a single image.

    Args:
        image_path: Absolute path to an image file, or "parquet_path#row_idx".

    Returns:
        {
          "verdict": "Genuine (Real)" | "AI-Generated",
          "confidence": float,          # softmax probability of predicted class
          "class_probabilities": dict,  # {label: probability} for 2 classes
          "trust_score": int,           # 0–100 fused trust score
          "heatmap_image": PIL.Image,   # Grad-CAM overlay
          "metadata_findings": dict,    # from metadata_forensics
          "artifact_findings": dict,    # from artifact_forensics
          "fusion_weights_used": dict,
          "source": str,
        }
    """
    model, device = _get_model_and_device()
    image_path = str(image_path)

    # ---- 1. Classifier inference ----
    tensor, pil_img = load_image_tensor(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(probs.argmax())
    pred_label_raw = IDX_TO_LABEL[pred_idx]           # 'genuine' / 'ai_generated'
    confidence = float(probs[pred_idx])

    verdict_map = {
        "genuine": "Genuine (Real)",
        "ai_generated": "AI-Generated",
    }
    verdict = verdict_map[pred_label_raw]

    class_probabilities = {
        IDX_TO_LABEL[i]: round(float(probs[i]), 4) for i in range(2)
    }

    # ---- 2. Grad-CAM heatmap ----
    try:
        heatmap = generate_heatmap(image_path, model, device)
    except Exception as e:
        heatmap = None
        print(f"  [WARN] Heatmap generation failed: {e}")

    # ---- 3. Metadata forensics (CPU) ----
    try:
        meta_findings = analyze_metadata(image_path)
    except Exception as e:
        meta_findings = {"error": str(e), "metadata_trust_signal": 0.5}

    # ---- 4. Artifact forensics (CPU) ----
    try:
        artifact_findings = analyze_forensic_artifacts(image_path)
    except Exception as e:
        artifact_findings = {"error": str(e), "artifact_trust_signal": 0.5}

    # ---- 5. Trust Fusion ----
    cls_signal = classifier_to_trust_signal(pred_label_raw, confidence)
    meta_signal = meta_findings.get("metadata_trust_signal", 0.5)
    art_signal = artifact_findings.get("artifact_trust_signal", 0.5)
    trust_score = compute_trust_score(cls_signal, meta_signal, art_signal, FUSION_WEIGHTS)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probabilities,
        "trust_score": trust_score,
        "heatmap_image": heatmap,
        "metadata_findings": meta_findings,
        "artifact_findings": artifact_findings,
        "fusion_weights_used": FUSION_WEIGHTS,
        "source": image_path,
    }


# --------------------------------------------------------------------------
# Pretty-print helper
# --------------------------------------------------------------------------
def print_analysis(result: dict):
    print("\n" + "=" * 65)
    print(f"  TRUEFRAME Analysis: {Path(result['source']).name}")
    print("=" * 65)
    print(f"  Verdict:        {result['verdict']}")
    print(f"  Confidence:     {result['confidence']*100:.1f}%")
    print(f"  Trust Score:    {result['trust_score']}/100")
    print(f"  Class probs:    Genuine={result['class_probabilities']['genuine']*100:.1f}%  "
          f"AI={result['class_probabilities']['ai_generated']*100:.1f}%")
    print(f"  Heatmap:        {'Generated' if result['heatmap_image'] else 'Failed'}")
    meta = result["metadata_findings"]
    print(f"  Metadata trust: {meta.get('metadata_trust_signal', '?'):.2f}  "
          f"| has_exif={meta.get('has_exif', '?')}  software={meta.get('software', 'None')}")
    art = result["artifact_findings"]
    print(f"  Artifact trust: {art.get('artifact_trust_signal', '?'):.2f}  "
          f"| ELA={art.get('ela_mean_score', '?'):.2f}  "
          f"FFT grid={art.get('fft_grid_score', '?'):.2f}")
    if meta.get("notes"):
        print(f"  Meta notes:     {meta['notes']}")
    if art.get("flags"):
        print(f"  Artifact flags: {art['flags']}")
    print("=" * 65)


# --------------------------------------------------------------------------
# CLI: test on real samples
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="TRUEFRAME Single Image Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to image file to test")
    args = parser.parse_args()

    heatmap_dir = Path(r"D:\demo\outputs\heatmaps")
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"❌ Error: File not found: {img_path}")
            sys.exit(1)
        print(f"\n🔍 Running TRUEFRAME analysis on: {img_path}")
        result = analyze_image(str(img_path.resolve()))
        print_analysis(result)
        if result["heatmap_image"] is not None:
            out_path = heatmap_dir / f"{img_path.stem}_heatmap.png"
            result["heatmap_image"].save(str(out_path))
            print(f"  🔥 Saved Grad-CAM heatmap to `{out_path}`\n")
    else:
        test_manifest = Path(r"D:\demo\manifest_test.csv")
        if not test_manifest.exists():
            print(f"ERROR: Test manifest not found: {test_manifest}")
            sys.exit(1)

        df = pd.read_csv(test_manifest)
        print("\n=== TRUEFRAME End-to-End Inference Test ===")
        print(f"Test manifest: {len(df):,} samples\n")

        test_images = []
        for label in ["genuine", "ai_generated", "manipulated"]:
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
                if result["heatmap_image"] is not None:
                    fname = Path(fp).stem + "_heatmap.png"
                    result["heatmap_image"].save(str(heatmap_dir / fname))
            except Exception as e:
                print(f"\n[ERROR] Failed on {fp}: {e}")
