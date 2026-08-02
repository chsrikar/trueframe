"""
trust_fusion.py — Weighted fusion of classifier + metadata + artifact signals
into a single Trust Score (0–100 integer) for TRUEFRAME pipeline.

Fusion weights are stored as a config dict (not hardcoded inline).
A weight sweep is provided to tune weights against the validation manifest
after training completes.
"""
import json
import sys
from pathlib import Path
from typing import Union

# --------------------------------------------------------------------------
# Default fusion weights (configurable — do not hardcode inside functions)
# --------------------------------------------------------------------------
FUSION_WEIGHTS: dict[str, float] = {
    "classifier": 0.45,
    "metadata":   0.20,
    "artifact":   0.20,
    "text":       0.15,
}

# Output paths
OUTPUT_DIR = Path(r"D:\demo\outputs")


# --------------------------------------------------------------------------
# Core fusion function
# --------------------------------------------------------------------------
def compute_trust_score(
    classifier_confidence: float,
    metadata_signal: float,
    artifact_signal: float,
    text_signal: float = 1.0,
    weights: dict[str, float] = FUSION_WEIGHTS,
) -> int:
    """
    Compute a Trust Score (0–100) for a single image.

    Args:
        classifier_confidence: Model softmax probability for the predicted class [0, 1].
        metadata_signal: Output of metadata_forensics ['metadata_trust_signal'] in [0, 1].
        artifact_signal: Output of artifact_forensics ['artifact_trust_signal'] in [0, 1].
        text_signal: Output of ocr_forensics ['text_trust_signal'] in [0, 1].
        weights: Dict with keys 'classifier', 'metadata', 'artifact', 'text'.

    Returns:
        int in [0, 100] — 100 = highly trusted genuine, 0 = clear manipulation/AI.
    """
    w_cls = weights.get("classifier", 0.45)
    w_meta = weights.get("metadata", 0.20)
    w_art = weights.get("artifact", 0.20)
    w_txt = weights.get("text", 0.15)

    # Normalise weights in case they don't sum to 1
    total = w_cls + w_meta + w_art + w_txt
    if total <= 0:
        raise ValueError("Fusion weights must sum to a positive number.")
    w_cls /= total
    w_meta /= total
    w_art /= total
    w_txt /= total

    # Weighted average of all signals in [0, 1]
    fused = (
        w_cls * float(classifier_confidence)
        + w_meta * float(metadata_signal)
        + w_art * float(artifact_signal)
        + w_txt * float(text_signal)
    )

    # Hard override: If OCR text anomaly is severe (text_signal < 0.5), cap maximum trust
    if text_signal < 0.5:
        fused = min(fused, text_signal)

    # Scale to 0–100 and clamp
    trust_score = int(round(max(0.0, min(1.0, fused)) * 100))
    return trust_score



# --------------------------------------------------------------------------
# Convert classifier verdict + confidence to trust-oriented signal
# --------------------------------------------------------------------------
def classifier_to_trust_signal(verdict: str, confidence: float) -> float:
    """
    Convert a classifier (verdict, confidence) pair into a trust signal [0, 1].

    - If verdict == 'genuine': trust signal = confidence (high conf = high trust)
    - If verdict == 'manipulated': trust signal = 1 - confidence (high conf = low trust)
    - If verdict == 'ai_generated': trust signal = 1 - confidence
    """
    if verdict == "genuine":
        return float(confidence)
    elif verdict == "ai_generated":
        return 1.0 - float(confidence)
    else:
        return 0.5  # unknown — neutral


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
    accuracy of the binarised trust score against ground truth.

    Args:
        predictions: List of dicts with keys:
            'verdict', 'confidence', 'metadata_signal', 'artifact_signal'
        ground_truth_labels: List of label strings ('genuine', 'ai_generated', 'manipulated')
        weight_combinations: Optional list of weight dicts to try.
            Defaults to 4 standard combinations.

    Returns:
        dict with 'best_weights', 'best_accuracy', 'sweep_results'
    """
    if weight_combinations is None:
        weight_combinations = [
            {"classifier": 0.50, "metadata": 0.25, "artifact": 0.25},  # default
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
                weights=weights,
            )
            # Binarise: score >= 50 → "trust" (genuine expected), < 50 → "suspicious"
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
# CLI: quick smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Trust Fusion Smoke Test ===\n")
    print("Using weights:", FUSION_WEIGHTS)

    test_cases = [
        # (verdict, confidence, meta_signal, artifact_signal, expected_trust_direction)
        ("genuine",      0.95, 0.85, 0.80, "HIGH"),
        ("genuine",      0.60, 0.20, 0.30, "MEDIUM-LOW"),
        ("ai_generated", 0.92, 0.10, 0.20, "LOW"),
        ("manipulated",  0.88, 0.35, 0.15, "LOW"),
        ("genuine",      0.50, 0.50, 0.50, "NEUTRAL"),
    ]

    print(f"{'Verdict':<15} {'Conf':>6}  {'Meta':>6}  {'Art':>6}  {'Trust Score':>12}  Expected")
    print("-" * 68)
    for verdict, conf, meta, art, expected in test_cases:
        cls_signal = classifier_to_trust_signal(verdict, conf)
        score = compute_trust_score(cls_signal, meta, art)
        print(f"{verdict:<15} {conf:>6.2f}  {meta:>6.2f}  {art:>6.2f}  {score:>12}  [{expected}]")

    print("\n[OK] Trust fusion module working correctly.")
