"""
evaluate.py — Final evaluation on the LOCKED test set for TRUEFRAME pipeline.
Requires GPU (calls require_cuda()).
NEVER called until training is fully complete and a best_model.pth exists.

Outputs:
  outputs/metrics.json          — Full metrics report
  outputs/confusion_matrix.png  — Plotted 3x3 confusion matrix
  outputs/error_analysis.md     — 10–15 worst misclassifications
"""
import torch_compat  # noqa: F401
import io
import sys
import json
import random
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from gpu_check import require_cuda
from dataset import TrueframeDataset, LABEL_TO_IDX, IDX_TO_LABEL
from train import build_model
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT   = Path(__file__).resolve().parent
MANIFEST_DIR   = Path(os.environ.get("TRUEFRAME_MANIFEST_DIR", PROJECT_ROOT)).expanduser().resolve()
TEST_MANIFEST  = MANIFEST_DIR / "manifest_test.csv"
CHECKPOINT_DIR = Path(os.environ.get("TRUEFRAME_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")).expanduser().resolve()
CHECKPOINT     = CHECKPOINT_DIR / "best_model.pth"
OUTPUT_DIR     = Path(os.environ.get("TRUEFRAME_OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()
LABEL_NAMES    = ["genuine", "ai_generated"]


# --------------------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------------------
def evaluate_test_set(batch_size: int = 32, num_workers: int = 0) -> dict:
    """Run full test-set evaluation. Returns metrics dict."""
    device = require_cuda(allow_cpu=True)

    if not TEST_MANIFEST.exists():
        print(f"ERROR: Test manifest not found: {TEST_MANIFEST}")
        sys.exit(1)
    if not CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT}")
        print("  Run train.py first.")
        sys.exit(1)

    print(f"Loading model from {CHECKPOINT}...")
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    test_df = pd.read_csv(TEST_MANIFEST)
    train_manifest = MANIFEST_DIR / "manifest_train.csv"
    val_manifest = MANIFEST_DIR / "manifest_val.csv"
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("Train/validation manifests are needed to verify the held-out generator split.")
    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    required = {"label", "generator", "source"}
    if not required.issubset(test_df.columns) or not required.issubset(train_df.columns) or not required.issubset(val_df.columns):
        raise ValueError("Manifests lack generator metadata. Regenerate all manifests with build_manifest.py.")
    train_generators = set(train_df.loc[train_df.label == "ai_generated", "generator"].astype(str))
    val_generators = set(val_df.loc[val_df.label == "ai_generated", "generator"].astype(str))
    test_generators = set(test_df.loc[test_df.label == "ai_generated", "generator"].astype(str))
    if train_generators & test_generators or train_generators & val_generators or val_generators & test_generators:
        raise ValueError("Synthetic generator groups overlap across splits. Rebuild manifests before evaluation.")
    print(f"Test manifest: {len(test_df):,} samples")
    print(f"Class distribution:\n{test_df['label'].value_counts().to_string()}\n")

    test_dataset = TrueframeDataset(test_df, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_preds    = []
    all_targets  = []
    all_probs    = []

    print("Running inference on test set...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.tolist())

            if (i + 1) % 50 == 0:
                print(f"  Processed {(i+1)*batch_size:,} / {len(test_df):,} samples...")

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs   = np.array(all_probs)

    return all_preds, all_targets, all_probs, test_df


# --------------------------------------------------------------------------
# Metrics computation
# --------------------------------------------------------------------------
def compute_metrics(preds, targets, probs, test_df=None):
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, roc_auc_score,
    )

    acc = float(accuracy_score(targets, preds))

    report = classification_report(
        targets, preds,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Per-class ROC-AUC (One-vs-Rest)
    n_classes = probs.shape[1]
    roc_auc = {}
    for i, name in enumerate(LABEL_NAMES):
        try:
            binary_targets = (targets == i).astype(int)
            auc = float(roc_auc_score(binary_targets, probs[:, i]))
            roc_auc[name] = round(auc, 4)
        except Exception:
            roc_auc[name] = None

    cm = confusion_matrix(targets, preds, labels=[0, 1]).tolist()

    confidence = probs.max(axis=1)
    correct = (preds == targets).astype(float)
    ece = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))

    result = {
        "accuracy": round(acc, 4),
        "classification_report": report,
        "roc_auc_ovr": roc_auc,
        "brier_score": round(float(np.mean(np.sum((probs - np.eye(probs.shape[1])[targets]) ** 2, axis=1))), 4),
        "expected_calibration_error_10_bins": round(ece, 4),
        "confidence_is_calibrated": False,
        "confusion_matrix": cm,
        "confusion_matrix_labels": LABEL_NAMES,
    }
    if test_df is not None:
        for group_col in ("source", "generator"):
            if group_col not in test_df.columns:
                continue
            result[f"metrics_by_{group_col}"] = {}
            values = test_df[group_col].fillna("unknown").astype(str).to_numpy()
            for group in sorted(set(values)):
                mask = values == group
                group_report = classification_report(
                    targets[mask], preds[mask], labels=[0, 1], target_names=LABEL_NAMES,
                    output_dict=True, zero_division=0,
                )
                result[f"metrics_by_{group_col}"][group] = {
                    "count": int(mask.sum()),
                    "accuracy": round(float(accuracy_score(targets[mask], preds[mask])), 4),
                    "macro_f1": round(float(group_report["macro avg"]["f1-score"]), 4),
                    "classification_report": group_report,
                }
    return result


def source_only_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Measure how much label can be predicted from dataset source alone."""
    source_labels = train_df.groupby("source")["label"].agg(lambda labels: labels.value_counts().index[0]).to_dict()
    global_label = train_df["label"].value_counts().index[0]
    predictions = np.array([
        LABEL_TO_IDX.get(source_labels.get(str(source), global_label), LABEL_TO_IDX[global_label])
        for source in test_df["source"].fillna("unknown")
    ])
    targets = np.array([LABEL_TO_IDX[label] for label in test_df["label"]])
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": round(float(accuracy_score(targets, predictions)), 4),
        "macro_f1": round(float(f1_score(targets, predictions, average="macro", zero_division=0)), 4),
        "source_to_training_majority_label": source_labels,
        "warning": "High performance means class and dataset source are confounded; it is not model performance.",
    }


# --------------------------------------------------------------------------
# Plot confusion matrix
# --------------------------------------------------------------------------
def plot_confusion_matrix(cm: list, labels: list, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm_arr = np.array(cm)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm_arr, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("TRUEFRAME — Confusion Matrix (Test Set)", fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        print(f"[OK] Confusion matrix saved to {out_path}")
    except Exception as e:
        print(f"[WARN] Could not plot confusion matrix (missing matplotlib/seaborn?): {e}")
        print("       Install: pip install matplotlib seaborn")


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------
def write_error_analysis(preds, targets, probs, test_df, out_path: Path, n: int = 15):
    """Write a markdown error analysis of the worst misclassifications."""
    errors = np.where(preds != targets)[0]
    if len(errors) == 0:
        out_path.write_text("# Error Analysis\n\nNo misclassifications found!\n")
        return

    # Sort by confidence in wrong prediction (highest confidence = worst errors)
    error_confs = probs[errors, preds[errors]]
    sorted_idx  = np.argsort(-error_confs)[:n]
    worst       = errors[sorted_idx]

    lines = [
        "# TRUEFRAME — Error Analysis (Worst Misclassifications)\n\n",
        f"Total misclassified: {len(errors):,} / {len(targets):,} "
        f"({len(errors)/len(targets)*100:.1f}%)\n\n",
        "## Top Worst Errors\n\n",
        "| # | True Label | Predicted | Confidence | Source | File |\n",
        "|---|-----------|-----------|------------|--------|------|\n",
    ]

    confusion_patterns = {}
    for idx in errors:
        true_lbl = IDX_TO_LABEL[int(targets[idx])]
        pred_lbl = IDX_TO_LABEL[int(preds[idx])]
        key = f"{true_lbl} → {pred_lbl}"
        confusion_patterns[key] = confusion_patterns.get(key, 0) + 1
    for rank, idx in enumerate(worst):
        true_lbl = IDX_TO_LABEL[int(targets[idx])]
        pred_lbl = IDX_TO_LABEL[int(preds[idx])]
        conf     = probs[idx, preds[idx]] * 100
        row      = test_df.iloc[idx]
        source   = row.get("source", "?")
        fname    = str(row["filepath"])[-50:]

        lines.append(
            f"| {rank+1} | {true_lbl} | {pred_lbl} | {conf:.1f}% | {source} | ...{fname} |\n"
        )


    # Pattern summary
    lines.append("\n## All Error Confusion Patterns\n\n")
    for pattern, count in sorted(confusion_patterns.items(), key=lambda x: -x[1]):
        lines.append(f"- **{pattern}**: {count} cases\n")

    lines.append("\n## Notes\n\n")
    lines.append("- This report lists observed errors only; it does not infer causes from filenames or confidence.\n")
    lines.append("- Confidence is raw softmax output and has not been calibrated.\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] Error analysis saved to {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("   TRUEFRAME — Final Test-Set Evaluation")
    print("=" * 70 + "\n")

    preds, targets, probs, test_df = evaluate_test_set(batch_size=32, num_workers=0)
    test_df = test_df.reset_index(drop=True)
    metrics = compute_metrics(preds, targets, probs, test_df)
    train_df = pd.read_csv(MANIFEST_DIR / "manifest_train.csv")
    metrics["source_only_majority_baseline"] = source_only_baseline(train_df, test_df)
    metrics["evaluation_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["test_manifest_sha256"] = hashlib.sha256(TEST_MANIFEST.read_bytes()).hexdigest()
    metrics["train_manifest_sha256"] = hashlib.sha256((MANIFEST_DIR / "manifest_train.csv").read_bytes()).hexdigest()
    metrics["validation_manifest_sha256"] = hashlib.sha256((MANIFEST_DIR / "manifest_val.csv").read_bytes()).hexdigest()
    metrics["checkpoint_sha256"] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    metrics["test_sample_count"] = int(len(test_df))
    metrics["evaluation_protocol"] = "See split_protocol.json; results are limited to listed datasets and generators."

    # Save metrics JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved to {metrics_path}")

    # Print summary
    print("\n--- Test Set Metrics ---")
    print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1:    {metrics['classification_report']['macro avg']['f1-score']*100:.2f}%")
    print(f"  ROC-AUC (OvR):")
    for label, auc in metrics["roc_auc_ovr"].items():
        print(f"    {label:<15}: {auc}")
    print(f"\n  Per-class metrics:")
    for label in LABEL_NAMES:
        r = metrics["classification_report"][label]
        print(f"    {label:<15}: P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  support={r['support']}")

    # Plot confusion matrix
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], LABEL_NAMES, cm_path)

    # Error analysis
    error_path = OUTPUT_DIR / "error_analysis.md"
    write_error_analysis(preds, targets, probs, test_df, error_path)

    print("\n[DONE] Evaluation complete.")
