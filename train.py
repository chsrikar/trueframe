import torch_compat  # noqa: F401
import os
import sys
import json
import hashlib
import time
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from dataset import TrueframeDataset, get_weighted_sampler, LABEL_TO_IDX, IDX_TO_LABEL
from gpu_check import require_cuda

# Ensure UTF-8 output encoding for terminal
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("TRUEFRAME_OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()
CHECKPOINT_DIR = Path(os.environ.get("TRUEFRAME_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(num_classes=2):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )
    return model

def freeze_backbone(model):
    """Freeze all features (Phase 1)."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_top_blocks(model, num_blocks=2):
    """Unfreeze top backbone blocks (Phase 2)."""
    # Unfreeze all classifier params
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Unfreeze features from last N blocks
    total_blocks = len(model.features)
    unfreeze_start = max(0, total_blocks - num_blocks)
    
    for i, block in enumerate(model.features):
        requires_grad = (i >= unfreeze_start)
        for param in block.parameters():
            param.requires_grad = requires_grad

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp=True, mixup_alpha=0.3):
    model.train()
    # Frozen backbone blocks must also keep BatchNorm statistics fixed.
    for block in model.features:
        if not any(p.requires_grad for p in block.parameters()):
            block.eval()
    running_loss = 0.0
    total_samples = 0
    expected_correct = 0.0
    
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)
        
        optimizer.zero_grad()

        if mixup_alpha > 0 and batch_size > 1:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            index = torch.randperm(batch_size, device=device)

            lam_tensor = torch.full((batch_size, 1, 1, 1), lam, device=device)
            mixed_images = lam_tensor * images + (1.0 - lam_tensor) * images[index]
            targets_a, targets_b = targets, targets[index]
            lam_vec = lam_tensor.view(-1)

            weight = criterion.weight if hasattr(criterion, 'weight') else None
            label_smoothing = criterion.label_smoothing if hasattr(criterion, 'label_smoothing') else 0.0

            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(mixed_images)
                    loss_a = torch.nn.functional.cross_entropy(outputs, targets_a, weight=weight, label_smoothing=label_smoothing, reduction='none')
                    loss_b = torch.nn.functional.cross_entropy(outputs, targets_b, weight=weight, label_smoothing=label_smoothing, reduction='none')
                    loss = (lam_vec * loss_a + (1.0 - lam_vec) * loss_b).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(mixed_images)
                loss_a = torch.nn.functional.cross_entropy(outputs, targets_a, weight=weight, label_smoothing=label_smoothing, reduction='none')
                loss_b = torch.nn.functional.cross_entropy(outputs, targets_b, weight=weight, label_smoothing=label_smoothing, reduction='none')
                loss = (lam_vec * loss_a + (1.0 - lam_vec) * loss_b).mean()
                loss.backward()
                optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            expected_correct += (
                lam * (preds == targets_a).float().sum().item()
                + (1.0 - lam) * (preds == targets_b).float().sum().item()
            )
        else:
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            expected_correct += (preds == targets).float().sum().item()
            
        running_loss += loss.item() * batch_size
        total_samples += batch_size
    epoch_loss = running_loss / max(total_samples, 1)
    # This expected accuracy reflects both MixUp labels; it is not hard-label F1.
    expected_accuracy = expected_correct / max(total_samples, 1)
    return epoch_loss, expected_accuracy, None

def evaluate(model, dataloader, criterion, device, use_amp=True):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)
            
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    precision, recall, _, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    per_class_metrics = {}
    for idx, name in IDX_TO_LABEL.items():
        per_class_metrics[name] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx])
        }
        
    return epoch_loss, epoch_acc, epoch_f1, per_class_metrics

def generate_model_card(best_metrics, train_df, val_df, hyperparams):
    demo_mode = os.environ.get("TRUEFRAME_DEMO_MODE", "0") == "1"
    sources = train_df.groupby(["label", "source"]).size().to_dict()
    generators = sorted(train_df.loc[train_df.label == "ai_generated", "generator"].astype(str).unique()) if "generator" in train_df else []
    if demo_mode:
        card_content = f"""# TRUEFRAME Closed-Set Demo Model Card

## Demo-only warning
This checkpoint was trained and selected using the same small set of `gimg/` and `aiimg/` examples. It is intended only to replay those known demo examples. It has no held-out validation or test result, and its scores must not be presented as evidence of general prediction accuracy.

## Model
- Architecture: torchvision EfficientNet-B0 with ImageNet weights and a two-layer MLP head (1280 → 256 GELU → 2).
- Labels: `genuine` (0), `ai_generated` (1).
- Inputs: RGB images resized/cropped to 256 × 256 and ImageNet-normalized.
- Training and selection images: {len(train_df)}; validation reuses those same images.
- Label counts by folder: `{sources}`.

Confidence is raw softmax output and is not calibrated. This is a memorization demo, not an authenticity detector.
"""
    else:
        card_content = f"""# TRUEFRAME Model Card

## Intended use
Research prototype for binary classification of the specific genuine and AI-generated face images in the declared manifests. It is not a general deepfake or image manipulation detector and should not be used as proof of authenticity.

## Model
- Architecture: torchvision EfficientNet-B0 with ImageNet weights and a two-layer MLP head (1280 → 256 GELU → 2).
- Labels: `genuine` (0), `ai_generated` (1).
- Training updates the classifier head first, then the top five EfficientNet feature blocks.
- Inputs: RGB images resized/cropped to 256 × 256 and ImageNet-normalized.

## Data and split
- Training images: {len(train_df):,}; validation images: {len(val_df):,}.
- Training counts by label and source: `{sources}`.
- Synthetic generator families present in training: {', '.join(generators) or 'not recorded'}.
- Split manifest: `split_protocol.json`. The synthetic validation and test generators are disjoint from training.
- Limitation: the AI class comes from one dataset family; real-image identity disjointness is not established. Results do not establish cross-dataset, identity, or universal generalization.
- Manipulated/deepfake images are not a target class in this experiment.

## Training configuration
- Optimizer: AdamW; phase 1 LR {hyperparams['phase1_lr']}; phase 2 LR {hyperparams['phase2_lr']}.
- Batch size: {hyperparams['batch_size']}; MixUp alpha: {hyperparams['mixup_alpha']}.
- Seed: {hyperparams['seed']}; phase 1 epochs: {hyperparams['phase1_epochs']}; phase 2 epochs: {hyperparams['phase2_epochs']}.
- Selection metric: validation macro F1.

## Validation result
- Best validation loss: {best_metrics['val_loss']:.4f}
- Best validation accuracy: {best_metrics['val_acc']:.4f}
- Best validation macro F1: {best_metrics['val_f1']:.4f}

This card reports validation only. Use `evaluate.py` on the held-out test manifest for test metrics. Confidence is raw softmax output and is not calibrated.
"""
    card_path = OUTPUT_DIR / "model_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"\n✅ Auto-generated Model Card written to `{card_path}`")

def _print_sanity_predictions(model, val_dataset, device, n=5):
    """Print stratified val-set predictions — at least 1 sample per class.

    Guarantees all three classes (genuine, ai_generated, manipulated) appear
    in the output so the sanity check is representative. Extra slots (n > 3)
    are filled with random samples from the full dataset.
    """
    import random as _random
    import torch.nn.functional as F

    # --- Build per-class index buckets ---
    class_buckets = {idx: [] for idx in IDX_TO_LABEL}
    for i in range(len(val_dataset)):
        _, label = val_dataset.get_label(i)   # fast label-only lookup
        class_buckets[label].append(i)

    # --- Stratified pick: 1 per class first, then random fill ---
    picked = []
    for cls_idx in sorted(class_buckets):
        if class_buckets[cls_idx]:
            picked.append(_random.choice(class_buckets[cls_idx]))

    remaining_slots = max(0, n - len(picked))
    all_indices = list(range(len(val_dataset)))
    extra = _random.sample(
        [i for i in all_indices if i not in set(picked)],
        min(remaining_slots, len(all_indices) - len(picked))
    )
    indices = picked + extra

    model.eval()
    print("\n" + "─" * 65)
    print("  SANITY CHECK — Val predictions after Epoch 1 (stratified)")
    print("─" * 65)
    print(f"  {'#':<4} {'True':<15} {'Pred':<15} {'Conf':>6}  {'OK?'}")
    print("─" * 65)
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, true_idx = val_dataset[idx]
            logits = model(img.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            pred_idx = int(probs.argmax())
            conf = float(probs[pred_idx]) * 100
            ok = "✅" if pred_idx == true_idx else "❌"
            print(f"  {i+1:<4} {IDX_TO_LABEL[true_idx]:<15} {IDX_TO_LABEL[pred_idx]:<15} {conf:>5.1f}%  {ok}")
    print("─" * 65 + "\n")
    model.train()

def train_pipeline(args):
    set_seed(args.seed)

    # --- GPU enforcement: exits immediately if no CUDA GPU found ---
    device = require_cuda(allow_cpu=args.cpu)
    if device.type == 'cpu':
        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(cpu_cores)
        print(f"   CPU multi-threading enabled: {cpu_cores} threads")
    print(f"   Seed: {args.seed}")
    
    train_csv = Path(args.train_manifest)
    val_csv = Path(args.val_manifest)
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    manifest_hashes = {
        "train": hashlib.sha256(train_csv.read_bytes()).hexdigest(),
        "validation": hashlib.sha256(val_csv.read_bytes()).hexdigest(),
    }
    training_config = {
        'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs,
        'phase1_lr': args.phase1_lr,
        'phase2_lr': args.phase2_lr,
        'batch_size': args.batch_size,
        'mixup_alpha': args.mixup_alpha,
        'seed': args.seed,
    }
    
    print(f"Loaded train set ({len(train_df):,} samples) and val set ({len(val_df):,} samples).")
    
    train_dataset = TrueframeDataset(train_df, is_train=True)
    val_dataset = TrueframeDataset(val_df, is_train=False)
    
    sampler, class_weights_dict = get_weighted_sampler(train_df)
    
    num_workers = args.num_workers
    _prefetch = 2 if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=_prefetch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,   # Same batch size for val to fit 4GB VRAM safely
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False,     # Avoid keeping persistent workers open for validation loader
        prefetch_factor=_prefetch
    )
    
    model = build_model(num_classes=2).to(device)
    
    if args.use_loss_weights:
        class_weights = torch.tensor([class_weights_dict[i] for i in range(2)], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    start_epoch = 1
    best_val_f1 = 0.0
    best_metrics = {}
    training_log = []
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_scaler_state = None
    
    best_model_path = CHECKPOINT_DIR / "best_model.pth"
    latest_ckpt_path = CHECKPOINT_DIR / "latest_checkpoint.pth"
    
    if args.resume and latest_ckpt_path.exists():
        print(f"🔄 Resuming training from checkpoint `{latest_ckpt_path}`...")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("manifest_hashes") != manifest_hashes:
            raise ValueError("Checkpoint was created from different/missing manifests. Start a fresh run or use its original manifests.")
        if ckpt.get("training_config") != training_config:
            raise ValueError("Checkpoint training configuration differs from this run. Resume with the original settings or start a fresh run.")
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        best_metrics = ckpt.get('best_metrics', {})
        training_log = ckpt.get('training_log', [])
        resume_optimizer_state = ckpt.get('optimizer_state_dict')
        resume_scheduler_state = ckpt.get('scheduler_state_dict')
        resume_scaler_state = ckpt.get('scaler_state_dict')
        if ckpt.get('python_rng_state') is not None:
            random.setstate(ckpt['python_rng_state'])
        if ckpt.get('numpy_rng_state') is not None:
            np.random.set_state(ckpt['numpy_rng_state'])
        if ckpt.get('torch_rng_state') is not None:
            torch.set_rng_state(ckpt['torch_rng_state'])
        if torch.cuda.is_available() and ckpt.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        if resume_scaler_state is not None:
            scaler.load_state_dict(resume_scaler_state)
        print(f"Resumed at epoch {start_epoch} (Best Val F1 so far: {best_val_f1:.4f})")
        
    print("\n" + "="*80)
    print("                      STARTING TWO-PHASE TRAINING")
    print("="*80)
    
    phase1_epochs = args.phase1_epochs
    phase2_epochs = args.phase2_epochs
    total_epochs = phase1_epochs + phase2_epochs
    
    patience = args.patience
    no_improve_counter = 0
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.phase1_lr, weight_decay=1e-4)
    scheduler = None
    resume_states_loaded = False
    
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()
        
        # Phase transition logic
        if epoch <= phase1_epochs:
            current_phase = 1
            if epoch == start_epoch or epoch == 1:
                print(f"\n--- Phase 1: Training Classifier Head Only (Epochs 1-{phase1_epochs}, LR={args.phase1_lr}) ---")
                freeze_backbone(model)
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.phase1_lr, weight_decay=1e-4)
                if resume_optimizer_state is not None and not resume_states_loaded:
                    optimizer.load_state_dict(resume_optimizer_state)
                    resume_states_loaded = True
        else:
            current_phase = 2
            if epoch == phase1_epochs + 1 or (start_epoch > phase1_epochs and epoch == start_epoch):
                print(f"\n--- Phase 2: Joint Fine-Tuning Top 5 Backbone Blocks (Epochs {phase1_epochs+1}-{total_epochs}, LR={args.phase2_lr}) ---")
                unfreeze_top_blocks(model, num_blocks=5)
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.phase2_lr, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=1e-6)
                if resume_optimizer_state is not None and not resume_states_loaded:
                    optimizer.load_state_dict(resume_optimizer_state)
                    if resume_scheduler_state is not None:
                        scheduler.load_state_dict(resume_scheduler_state)
                    resume_states_loaded = True

        train_loss, train_acc, _ = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=True, mixup_alpha=args.mixup_alpha)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        val_loss, val_acc, val_f1, per_class = evaluate(model, val_loader, criterion, device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # --- Sanity check after first epoch of Phase 1 ---
        if epoch == 1:
            _print_sanity_predictions(model, val_dataset, device, n=5)
        
        if scheduler:
            scheduler.step()
            
        elapsed = time.time() - epoch_start_time
        
        log_entry = {
            'epoch': epoch,
            'phase': current_phase,
            'train_loss': float(train_loss),
            'train_expected_accuracy': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_f1': float(val_f1),
            'per_class': per_class,
            'elapsed_sec': float(elapsed)
        }
        training_log.append(log_entry)
        
        # Write log to JSON
        with open(OUTPUT_DIR / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)
            
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] (P{current_phase}) | "
              f"Train Loss: {train_loss:.4f} Expected Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% F1: {val_f1:.4f} | ({elapsed:.1f}s)")

        stop_after_epoch = False
        # Check for best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1, 'per_class': per_class}
            torch.save(model.state_dict(), best_model_path)
            print(f"  ⭐ Saved new best model checkpoint to `{best_model_path}` (Val F1: {val_f1:.4f})")
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if current_phase == 2 and no_improve_counter >= patience:
                print(f"\n✋ Early stopping triggered after {patience} epochs without validation improvement.")
                stop_after_epoch = True

        # Save checkpoint after updating best metrics so resume state is complete.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict(),
            'best_val_f1': best_val_f1,
            'best_metrics': best_metrics,
            'manifest_hashes': manifest_hashes,
            'training_config': training_config,
            'training_log': training_log,
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, latest_ckpt_path)
        if stop_after_epoch:
            break

    print("\n" + "="*80)
    print("                      TRAINING COMPLETE")
    print(f"Best Validation Macro F1: {best_val_f1:.4f}")
    print("="*80)

    # Generate Model Card
    hyperparams = {
        'phase1_lr': args.phase1_lr,
        'phase2_lr': args.phase2_lr,
        'batch_size': args.batch_size,
        'mixup_alpha': args.mixup_alpha,
        'seed': args.seed,
        'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs,
    }
    if best_metrics:
        generate_model_card(best_metrics, train_df, val_df, hyperparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRUEFRAME Training Pipeline")
    parser.add_argument("--train-manifest", type=str, default=str(PROJECT_ROOT / "manifest_train.csv"))
    parser.add_argument("--val-manifest", type=str, default=str(PROJECT_ROOT / "manifest_val.csv"))
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per step. Colab T4 typically supports 32 at 256px; lower if needed.")
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-epochs", type=int, default=14)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--mixup-alpha", type=float, default=0.3,
                        help="Mixup alpha (default 0.3, set 0 to disable). Skips manipulated class.")
    parser.add_argument("--use-loss-weights", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Allow running on CPU")
    default_workers = 0 if sys.platform == "win32" else 4
    parser.add_argument("--num-workers", type=int, default=default_workers,
                        help="DataLoader workers (default: 0 on Windows to avoid shared memory error 1455, 4 on Linux)")

    args = parser.parse_args()
    train_pipeline(args)
