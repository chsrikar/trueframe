"""
gradcam.py — Grad-CAM heatmap generation for TRUEFRAME pipeline.
Targets the final conv block of EfficientNet-B0.
Requires GPU (calls require_cuda()).
"""
import io
import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from gpu_check import require_cuda
from dataset import LABEL_TO_IDX, IDX_TO_LABEL
from train import build_model

OUTPUT_DIR = Path(r"D:\demo\outputs\heatmaps")
CHECKPOINT_PATH = Path(r"D:\demo\checkpoints\best_model.pth")

# --------------------------------------------------------------------------
# Transform for inference (no augmentation)
# --------------------------------------------------------------------------
_INFER_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_tensor(image_source: Union[str, Path, torch.Tensor]) -> tuple[torch.Tensor, Image.Image]:
    """Load an image from a path (or parquet#row reference) and return (tensor, pil_image)."""
    if isinstance(image_source, torch.Tensor):
        return image_source, None

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


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM implementation hooking the final conv block of EfficientNet-B0.
    The target layer is model.features[-1] (MBConv block output).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []

        self._hooks.append(
            target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradients)
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
        Returns a 2-D numpy heatmap [H, W] in [0, 1].
        If target_class is None, uses the predicted class.
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)  # [1, C, H, W]

        with torch.enable_grad():
            logits = self.model(input_tensor)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            score = logits[0, target_class]
            self.model.zero_grad()
            score.backward()

        # Global average pool the gradients over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # [H, W]
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


# --------------------------------------------------------------------------
# Score-CAM (gradient-free fallback)
# --------------------------------------------------------------------------
class ScoreCAM:
    """
    Gradient-free Score-CAM implementation.
    Slower than Grad-CAM but doesn't require backward pass.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self._hook = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def remove_hook(self):
        self._hook.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                 device: torch.device = torch.device("cuda")) -> np.ndarray:
        self.model.eval()
        x = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            base_logits = self.model(x)
            if target_class is None:
                target_class = int(base_logits.argmax(dim=1).item())
            _ = self.activations  # trigger hook

        acts = self.activations  # [1, C, H, W]
        B, C, H, W = acts.shape
        _, _, iH, iW = x.shape

        scores = []
        for c in range(C):
            channel = acts[0, c].cpu().numpy()
            # Normalise channel map to [0, 1]
            if channel.max() > channel.min():
                channel = (channel - channel.min()) / (channel.max() - channel.min())
            mask = torch.tensor(channel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, size=(iH, iW), mode="bilinear", align_corners=False).to(device)
            masked = x * mask
            with torch.no_grad():
                out = self.model(masked)
                score = float(F.softmax(out, dim=1)[0, target_class].cpu())
            scores.append(score)

        scores_t = torch.tensor(scores, dtype=torch.float32).view(1, C, 1, 1).to(device)
        cam = (scores_t * acts).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


# --------------------------------------------------------------------------
# Overlay helper
# --------------------------------------------------------------------------
def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray,
                    alpha: float = 0.5, colormap: str = "jet") -> Image.Image:
    """Blend the CAM heatmap with the original image."""
    import matplotlib
    matplotlib.use("Agg")
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        import matplotlib.cm as mplcm
        cmap = mplcm.get_cmap(colormap)
    heatmap_rgb = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb).resize(pil_img.size, Image.BILINEAR)

    blended = Image.blend(pil_img.convert("RGB"), heatmap_pil, alpha=alpha)
    return blended


# --------------------------------------------------------------------------
# Main generate_heatmap API
# --------------------------------------------------------------------------
def generate_heatmap(
    image_source: Union[str, Path, torch.Tensor],
    model: nn.Module,
    device: torch.device,
    target_class: Optional[int] = None,
    use_score_cam: bool = False,
) -> Image.Image:
    """
    Generate a Grad-CAM (or Score-CAM) heatmap overlay for an image.

    Args:
        image_source: File path, parquet#row reference, or pre-loaded tensor.
        model: Loaded EfficientNet-B0 TRUEFRAME model (on device).
        device: torch.device("cuda")
        target_class: Class index to explain (None = predicted class).
        use_score_cam: Use gradient-free Score-CAM instead of Grad-CAM.

    Returns:
        PIL.Image — original image blended with the heatmap overlay.
    """
    tensor, pil_img = load_image_tensor(image_source)
    tensor = tensor.to(device)

    target_layer = model.features[-1]

    if use_score_cam:
        cam_gen = ScoreCAM(model, target_layer)
        cam = cam_gen.generate(tensor, target_class=target_class, device=device)
        cam_gen.remove_hook()
    else:
        cam_gen = GradCAM(model, target_layer)
        cam = cam_gen.generate(tensor, target_class=target_class)
        cam_gen.remove_hooks()

    if pil_img is None:
        # Convert tensor back to PIL for overlay
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm = tensor.cpu() * std + mean
        pil_img = T.ToPILImage()(denorm.clamp(0, 1))

    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
        pil_img.size, Image.BILINEAR
    )) / 255.0

    return overlay_heatmap(pil_img, cam_resized)


# --------------------------------------------------------------------------
# CLI: test on real samples from manifest_test.csv
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import pandas as pd

    device = require_cuda()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}. Train the model first.")
        sys.exit(1)

    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = build_model(num_classes=3)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    test_df = pd.read_csv(r"D:\demo\manifest_test.csv")

    # Sample 2 images per class
    print("\nGenerating heatmaps for 2 samples per class...")
    for label_str in ["genuine", "ai_generated"]:
        subset = test_df[test_df["label"] == label_str]
        if len(subset) == 0:
            print(f"  ⚠️ No {label_str} samples in test set.")
            continue
        samples = subset.sample(min(2, len(subset)), random_state=42)

        for i, (_, row) in enumerate(samples.iterrows()):
            fp = row["filepath"]
            try:
                heatmap = generate_heatmap(fp, model, device)
                out_path = OUTPUT_DIR / f"{label_str}_{i+1}_gradcam.png"
                heatmap.save(str(out_path))
                print(f"  ✅ Saved: {out_path.name}  [{label_str}]")
            except Exception as e:
                print(f"  ❌ Failed on {fp}: {e}")

    print(f"\nAll heatmaps saved to: {OUTPUT_DIR}")
