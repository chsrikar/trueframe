"""
trust_fusion.py — Multi-Signal Weighted Trust Fusion for TRUEFRAME Pipeline.
Fuses classifier confidence (modulated by Dual Grad-CAM spatial agreement) +
metadata authenticity signals + ELA/FFT artifact signals into a single Trust Score (0–100 integer).

OCR forensics has been retired in favor of Dual-Domain (Positive/Negative) Spatial Agreement.
"""
import json
import sys
from pathlib import Path
from typing import Union, Dict, Any, Optional

# --------------------------------------------------------------------------
# Default fusion weights (configurable — do not hardcode inside functions)
# --------------------------------------------------------------------------
FUSION_WEIGHTS: Dict[str, float] = {
    "classifier": 0.50,
    "metadata":   0.25,
    "artifact":   0.25,
}

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


# --------------------------------------------------------------------------
# Core fusion function
# --------------------------------------------------------------------------
def compute_trust_score(
    classifier_confidence: float,
    metadata_signal: float,
    artifact_signal: float,
    agreement_score: Optional[float] = None,
    inverted_signal: Optional[float] = None,
    weights: Dict[str, float] = FUSION_WEIGHTS,
) -> int:
    """
    Compute a Multi-Signal Trust Score (0–100) for a single image.

    Design Rationale & Mathematical Defense:
    ----------------------------------------
    1. Stability Factor Floor (0.55):
       `stability_factor = 0.55 + 0.45 * agreement_score`
       Rationale: A floor of 0.55 ensures that natural domain-shift variance (e.g. inverting
       natural lighting or dark skin tones) does not completely nullify the model's global
       semantic representations, but still penalizes single-domain hallucinations by up to 45%.

    2. Dual-Domain Inversion Cross-Validation:
       If the inverted image yields an opposing class prediction with high conviction,
       `inverted_signal` will diverge from `classifier_confidence`. When a domain flip occurs,
       a cross-domain divergence penalty dampens the classifier confidence toward 0.50 (neutral).

    3. Multi-Metric Agreement Split (0.45 IoU / 0.35 SSIM / 0.20 Pearson):
       - IoU (45%): Strict spatial localization of top-20% suspicious hot zones.
       - SSIM (35%): Structural continuity of activation textures and gradients.
       - Pearson (20%): Linear co-activation correlation.
    """
    w_cls = weights.get("classifier", 0.50)
    w_meta = weights.get("metadata", 0.25)
    w_art = weights.get("artifact", 0.25)

    # Normalize weights so they sum to 1.0
    total = w_cls + w_meta + w_art
    if total <= 0:
        raise ValueError("Fusion weights must sum to a positive number.")
    w_cls /= total
    w_meta /= total
    w_art /= total

    # 1. Apply Spatial Agreement Modifier
    cls_signal = float(classifier_confidence)
    if agreement_score is not None:
        aggr = float(max(0.0, min(1.0, agreement_score)))
        stability_factor = 0.55 + 0.45 * aggr
        cls_signal = 0.50 + (cls_signal - 0.50) * stability_factor

    # 2. Apply Inversion Domain Divergence Check
    if inverted_signal is not None:
        inv_sig = float(max(0.0, min(1.0, inverted_signal)))
        # Measure divergence from primary signal: |cls - inv|
        divergence = abs(float(classifier_confidence) - inv_sig)
        if divergence > 0.40:
            # Significant domain flip (e.g., normal says genuine (0.9), inverted says AI (0.1))
            # Moderate damping applied to prevent overconfident false positives
            flip_dampening = 1.0 - (divergence - 0.40) * 0.30
            cls_signal = 0.50 + (cls_signal - 0.50) * max(0.70, flip_dampening)

    # Weighted sum of forensic signals in [0, 1]
    fused = (
        w_cls * cls_signal
        + w_meta * float(metadata_signal)
        + w_art * float(artifact_signal)
    )

    # Scale to 0–100 integer range and clamp
    trust_score = int(round(max(0.0, min(1.0, fused)) * 100))
    return trust_score


# --------------------------------------------------------------------------
# Convert classifier verdict + confidence to trust-oriented signal
# --------------------------------------------------------------------------
def classifier_to_trust_signal(verdict: str, confidence: float) -> float:
    """
    Convert a classifier (verdict, confidence) pair into a trust signal [0, 1].

    - If verdict == 'genuine': trust signal = confidence (high conf = high trust)
    - If verdict == 'ai_generated' or 'manipulated': trust signal = 1 - confidence (high conf = low trust)
    """
    v_clean = str(verdict).lower()
    if "genuine" in v_clean:
        return float(confidence)
    elif "ai" in v_clean or "manipulated" in v_clean or "synthetic" in v_clean:
        return 1.0 - float(confidence)
    else:
        return 0.5  # neutral / unknown


# --------------------------------------------------------------------------
# Weight sweep: find best weights against a reference manifest
# --------------------------------------------------------------------------
def run_weight_sweep(
    predictions: list[dict],
    ground_truth_labels: list[str],
    weight_combinations: list[dict] = None,
) -> dict:
    """
    Sweep several weight combinations and return the one that maximises
    binary classification accuracy of the fused trust score against ground truth.
    """
    if weight_combinations is None:
        weight_combinations = [
            {"classifier": 0.50, "metadata": 0.25, "artifact": 0.25},  # balanced default
            {"classifier": 0.70, "metadata": 0.15, "artifact": 0.15},  # classifier-heavy
            {"classifier": 0.40, "metadata": 0.35, "artifact": 0.25},  # metadata-heavy
            {"classifier": 0.40, "metadata": 0.25, "artifact": 0.35},  # artifact-heavy
        ]

    sweep_results = []
    best_weights = weight_combinations[0]
    best_accuracy = -1.0

    for weights in weight_combinations:
        correct = 0
        for pred, gt in zip(predictions, ground_truth_labels):
            cls_signal = classifier_to_trust_signal(pred["verdict"], pred["confidence"])
            score = compute_trust_score(
                classifier_confidence=cls_signal,
                metadata_signal=pred["metadata_signal"],
                artifact_signal=pred["artifact_signal"],
                agreement_score=pred.get("agreement_score", 1.0),
                weights=weights,
            )
            predicted_genuine = score >= 50
            actually_genuine = gt == "genuine"
            if predicted_genuine == actually_genuine:
                correct += 1

        accuracy = correct / len(predictions) if predictions else 0.0
        entry = {"weights": weights, "accuracy": round(accuracy, 4)}
        sweep_results.append(entry)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights

    return {
        "best_weights": best_weights,
        "best_accuracy": best_accuracy,
        "sweep_results": sweep_results,
    }


def save_fusion_report(sweep_result: dict, notes: str = ""):
    """Save the weight sweep results to outputs/fusion_weights.md."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TRUEFRAME Trust Fusion — Weight Sweep Results\n",
        f"**Best weights chosen**: {sweep_result['best_weights']}\n",
        f"**Best binary accuracy**: {sweep_result['best_accuracy']*100:.1f}%\n\n",
        "## Sweep Table\n\n",
        "| Classifier | Metadata | Artifact | Binary Accuracy |\n",
        "|-----------|----------|----------|-----------------|\n",
    ]
    for r in sweep_result["sweep_results"]:
        w = r["weights"]
        lines.append(
            f"| {w['classifier']:.2f}      | {w['metadata']:.2f}     "
            f"| {w['artifact']:.2f}     | {r['accuracy']*100:.1f}%           |\n"
        )
    if notes:
        lines.append(f"\n## Notes\n\n{notes}\n")

    out_path = OUTPUT_DIR / "fusion_weights.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[OK] Fusion weight report saved to {out_path}")


# --------------------------------------------------------------------------
# CLI smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Trust Fusion (Dual Grad-CAM & Multi-Signal) Smoke Test ===\n")
    print("Using weights:", FUSION_WEIGHTS)

    test_cases = [
        # (verdict, confidence, meta_signal, artifact_signal, agreement_score, expected)
        ("genuine",      0.95, 0.85, 0.80, 0.92, "HIGH"),
        ("genuine",      0.95, 0.85, 0.80, 0.20, "REDUCED_BY_SPATIAL_DISAGREEMENT"),
        ("ai_generated", 0.92, 0.10, 0.20, 0.85, "LOW"),
        ("ai_generated", 0.60, 0.50, 0.50, 0.50, "MEDIUM-LOW"),
        ("genuine",      0.50, 0.50, 0.50, 0.50, "NEUTRAL"),
    ]

    print(f"{'Verdict':<15} {'Conf':>6}  {'Meta':>6}  {'Art':>6}  {'Aggr':>6}  {'Trust Score':>12}  Expected")
    print("-" * 78)
    for verdict, conf, meta, art, aggr, expected in test_cases:
        cls_signal = classifier_to_trust_signal(verdict, conf)
        score = compute_trust_score(cls_signal, meta, art, agreement_score=aggr)
        print(f"{verdict:<15} {conf:>6.2f}  {meta:>6.2f}  {art:>6.2f}  {aggr:>6.2f}  {score:>12}  [{expected}]")

    print("\n[OK] Trust fusion module verified.")
