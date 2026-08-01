"""
gpu_check.py — CUDA enforcement utility for TRUEFRAME pipeline.
Every model-touching script must call require_cuda() at startup.
"""
import sys
import torch

sys.stdout.reconfigure(encoding='utf-8')


def require_cuda() -> "torch.device":
    """
    Assert CUDA GPU is available and return torch.device('cuda').
    Exits with a clear error message if no GPU is found — the pipeline
    must never silently fall back to CPU training.
    """
    if not torch.cuda.is_available():
        print(
            "\n" + "=" * 70 + "\n"
            "ERROR: CUDA GPU not detected.\n"
            "This pipeline requires GPU training (RTX 3050) and will NOT\n"
            "proceed on CPU — it would take 10–20× longer and is not supported.\n\n"
            "Possible fixes:\n"
            "  1. Ensure you installed the CUDA build of PyTorch:\n"
            "       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
            "  2. Check that your NVIDIA drivers are installed (run: nvidia-smi)\n"
            "  3. Confirm the GPU is not disabled in Device Manager\n"
            + "=" * 70
        )
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[OK] GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"     CUDA version:   {torch.version.cuda}")
    print(f"     PyTorch:        {torch.__version__}")
    return device


if __name__ == "__main__":
    device = require_cuda()
    print(f"\nDevice ready: {device}")
