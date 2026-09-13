"""
gradcam.py — Dual-Domain Grad-CAM & Attention Fusion Engine for TRUEFRAME.
Computes activation heatmaps in both original (positive) and inverted (negative) domains
via EfficientNet-B0 (model.features[-1]), fuses the spatial maps, and calculates a quantitative
multi-metric spatial agreement score (IoU, SSIM, Pearson Correlation).
Requires GPU (calls require_cuda()).
"""
import io
import os
import random
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from dataset import LABEL_TO_IDX, IDX_TO_LABEL
except ImportError:
    LABEL_TO_IDX = {"genuine": 0, "ai_generated": 1}
    IDX_TO_LABEL = {0: "genuine", 1: "ai_generated"}
from invert import _invert_array

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "heatmaps"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

try:
    import torchvision.transforms as T
    _INFER_TRANSFORM = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except ImportError:
    _INFER_TRANSFORM = None


def load_image_tensor(image_source: Union[str, Path, Image.Image, torch.Tensor]) -> tuple[torch.Tensor, Image.Image]:
    """Load an image from a path, PIL Image, or parquet#row reference and return (tensor, pil_image)."""
    if isinstance(image_source, torch.Tensor):
        return image_source, None

    if isinstance(image_source, Image.Image):
        pil_img = image_source.convert("RGB")
    else:
        path_str = str(image_source)
        if "#" in path_str:
            from dataset import get_parquet_image_bytes
            parquet_path, idx_str = path_str.split("#")
            img_bytes = get_parquet_image_bytes(parquet_path, int(idx_str))
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            pil_img = Image.open(path_str).convert("RGB")

    tensor = _INFER_TRANSFORM(pil_img)
    return tensor, pil_img


def load_dual_image_tensors(
    image_source: Union[str, Path, Image.Image]
) -> Tuple[torch.Tensor, Image.Image, torch.Tensor, Image.Image]:
    """
    Load image and generate both normal (positive) and inverted (negative) versions.
    Reuses _invert_array from invert.py to avoid code duplication.
    
    Returns:
        (tensor_normal, pil_normal, tensor_inverted, pil_inverted)
    """
    tensor_normal, pil_normal = load_image_tensor(image_source)

    # Invert using invert.py's vectorised NumPy math (preserving image mode)
    np_normal = np.array(pil_normal)
    np_inverted = _invert_array(np_normal)
    pil_inverted = Image.fromarray(np_inverted, mode=pil_normal.mode)
    tensor_inverted = _INFER_TRANSFORM(pil_inverted)

    return tensor_normal, pil_normal, tensor_inverted, pil_inverted


# --------------------------------------------------------------------------
# Grad-CAM Implementation
# --------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM implementation hooking the final conv block of EfficientNet-B0.
    The target layer is model.features[-1] (MBConv block output).
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []

        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2-D numpy heatmap [H, W] normalized to [0, 1].
        If target_class is None, uses the predicted class.
        """
        self.model.eval()
        x = input_tensor if input_tensor.dim() == 4 else input_tensor.unsqueeze(0)

        with torch.enable_grad():
            logits = self.model(x)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            score = logits[0, target_class]
            self.model.zero_grad()
            score.backward()

        # Global average pool the gradients over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # [H, W]
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


# --------------------------------------------------------------------------
# Map Alignment & Resizing Helper
# --------------------------------------------------------------------------
def align_cam_map(cam: np.ndarray, target_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Explicitly resizes and normalizes an activation map to target_shape (H, W)."""
    if cam.shape == target_shape:
        return np.clip(cam, 0.0, 1.0)
    
    pil_cam = Image.fromarray((np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8))
    resized = pil_cam.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    aligned = np.array(resized, dtype=np.float32) / 255.0
    return aligned


# --------------------------------------------------------------------------
# Multi-Metric Heatmap Agreement & Fusion
# --------------------------------------------------------------------------
def fuse_gradcams(
    cam_normal: np.ndarray,
    cam_inverted: np.ndarray,
    target_shape: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fuse positive-domain and negative-domain Grad-CAM activation maps into a single
    robust fused heatmap, and compute quantitative multi-metric spatial agreement.

    Fusion Hypothesis & Design Rationale:
    --------------------------------------
    1. Multiplicative Fusion (cam_normal * cam_inverted):
       Regions where BOTH normal and inverted passes trigger high activation represent
       structural, luminance-invariant synthesis artifacts (e.g. boundary seams, unnatural skin blending).
       Multiplication naturally suppresses single-domain noise while amplifying consensus regions.
    
    2. Agreement Metrics (all bounded in [0.0, 1.0]):
       - Hotspot IoU (Intersection-over-Union): Measures overlap of the top 20% most activated
         regions (percentile 80). Highly interpretable for forensic verification: "Do both domains
         localize the exact same suspicious zone?"
       - SSIM (Structural Similarity Index): Measures spatial gradient, variance, and texture
         structural agreement across the continuous heatmap space.
       - Pearson Correlation: Linear co-activation correlation between flattened maps, mapped to [0, 1].

    3. Composite Agreement Score:
       agreement_score = 0.45 * IoU + 0.35 * SSIM + 0.20 * Normalized_Pearson
       Range: [0.0, 1.0] where 1.0 indicates perfect spatial consistency across domains.

    Returns:
        fused_cam: 2D numpy array [H, W] normalized to [0.0, 1.0].
        metrics: Dict containing agreement_score, hotspot_iou, ssim_score, pearson_corr.
    """
    # 1. Align and enforce uniform spatial grid
    map_a = align_cam_map(cam_normal, target_shape)
    map_b = align_cam_map(cam_inverted, target_shape)

    # 2. Multiplicative consensus fusion
    fused_raw = map_a * map_b
    f_min, f_max = fused_raw.min(), fused_raw.max()
    if f_max > f_min:
        fused_cam = (fused_raw - f_min) / (f_max - f_min + 1e-8)
    else:
        fused_cam = (map_a + map_b) / 2.0

    # 3. Metric A: Hotspot IoU (Top 20% activation threshold)
    th_a = np.percentile(map_a, 80)
    th_b = np.percentile(map_b, 80)
    mask_a = map_a >= th_a
    mask_b = map_b >= th_b
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    hotspot_iou = float(intersection / (union + 1e-8))

    # 4. Metric B: Structural Similarity Index (SSIM)
    mu_a = float(np.mean(map_a))
    mu_b = float(np.mean(map_b))
    var_a = float(np.var(map_a))
    var_b = float(np.var(map_b))
    cov_ab = float(np.mean((map_a - mu_a) * (map_b - mu_b)))
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    ssim_raw = ((2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2) + 1e-8
    )
    ssim_score = float(np.clip(ssim_raw, 0.0, 1.0))

    # 5. Metric C: Pearson Linear Correlation Coefficient
    flat_a = map_a.flatten()
    flat_b = map_b.flatten()
    std_a = np.std(flat_a)
    std_b = np.std(flat_b)
    if std_a > 1e-8 and std_b > 1e-8:
        pearson_raw = float(np.corrcoef(flat_a, flat_b)[0, 1])
    else:
        pearson_raw = 0.0
    # Map [-1, 1] to [0, 1] for unified scoring
    pearson_norm = float(np.clip((pearson_raw + 1.0) / 2.0, 0.0, 1.0))

    # 6. Composite Agreement Score (Bounded in [0.0, 1.0])
    composite_agreement = float(
        np.clip(
            0.45 * hotspot_iou + 0.35 * ssim_score + 0.20 * pearson_norm,
            0.0,
            1.0,
        )
    )

    metrics = {
        "agreement_score": round(composite_agreement, 4),
        "hotspot_iou": round(hotspot_iou, 4),
        "ssim_score": round(ssim_score, 4),
        "pearson_corr": round(pearson_raw, 4),
        "pearson_norm": round(pearson_norm, 4),
    }

    return fused_cam, metrics


# --------------------------------------------------------------------------
# Overlay helper
# --------------------------------------------------------------------------
def overlay_heatmap(
    pil_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> Image.Image:
    """Blend an activation heatmap with the base PIL image."""
    import matplotlib
    matplotlib.use("Agg")
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        import matplotlib.cm as mplcm
        cmap = mplcm.get_cmap(colormap)

    cam_resized = np.array(
        Image.fromarray((np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)).resize(
            pil_img.size, Image.BILINEAR
        )
    ) / 255.0

    heatmap_rgb = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    blended = Image.blend(pil_img.convert("RGB"), heatmap_pil, alpha=alpha)
    return blended


# --------------------------------------------------------------------------
# Dual Grad-CAM Pipeline Runner
# --------------------------------------------------------------------------
def generate_dual_heatmaps(
    image_source: Union[str, Path, Image.Image],
    model: nn.Module,
    device: torch.device,
    target_class: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Complete dual-domain Grad-CAM generation:
    1. Loads normal and inverted tensors via invert.py logic
    2. Runs hook-based Grad-CAM on both passes (shared target layer)
    3. Fuses activation maps and computes agreement metrics
    4. Generates visual overlays for Normal, Inverted, and Fused heatmaps

    Returns dict with:
        - heatmap_normal: PIL.Image
        - heatmap_inverted: PIL.Image
        - heatmap_fused: PIL.Image
        - cam_normal_raw: 2D numpy
        - cam_inverted_raw: 2D numpy
        - cam_fused_raw: 2D numpy
        - agreement_data: Dict of agreement metrics
        - pil_normal: Base image
        - pil_inverted: Inverted image
    """
    t_normal, pil_normal, t_inverted, pil_inverted = load_dual_image_tensors(image_source)
    t_normal = t_normal.to(device)
    t_inverted = t_inverted.to(device)

    target_layer = model.features[-1]
    cam_gen = GradCAM(model, target_layer)

    # Pass 1: Normal image Grad-CAM
    cam_normal = cam_gen.generate(t_normal, target_class=target_class)

    # Pass 2: Inverted image Grad-CAM
    cam_inverted = cam_gen.generate(t_inverted, target_class=target_class)

    cam_gen.remove_hooks()

    # Align & Fuse
    cam_fused, agreement_data = fuse_gradcams(cam_normal, cam_inverted)

    # Generate visual overlays
    overlay_norm = overlay_heatmap(pil_normal, cam_normal)
    overlay_inv = overlay_heatmap(pil_inverted, cam_inverted)
    overlay_fused = overlay_heatmap(pil_normal, cam_fused)

    return {
        "heatmap_normal": overlay_norm,
        "heatmap_inverted": overlay_inv,
        "heatmap_fused": overlay_fused,
        "cam_normal_raw": cam_normal,
        "cam_inverted_raw": cam_inverted,
        "cam_fused_raw": cam_fused,
        "agreement_data": agreement_data,
        "pil_normal": pil_normal,
        "pil_inverted": pil_inverted,
    }


# Backwards compatibility alias for single heatmap request
def generate_heatmap(
    image_source: Union[str, Path, torch.Tensor],
    model: nn.Module,
    device: torch.device,
    target_class: Optional[int] = None,
) -> Image.Image:
    res = generate_dual_heatmaps(image_source, model, device, target_class=target_class)
    return res["heatmap_fused"]


if __name__ == "__main__":
    device = require_cuda()
    print("=== Dual Grad-CAM & Fusion Smoke Test ===")
    model = build_model(num_classes=2)
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    test_img = Image.new("RGB", (256, 256), color=(120, 150, 200))
    res = generate_dual_heatmaps(test_img, model, device)
    print("Agreement Metrics:", res["agreement_data"])
    print("[OK] Dual Grad-CAM module initialized successfully.")
