"""
gpu_check.py — CUDA enforcement utility for TRUEFRAME pipeline.
Every model-touching script must call require_cuda() at startup.
"""
import sys
import torch

sys.stdout.reconfigure(encoding='utf-8')


def require_cuda(allow_cpu: bool = False) -> "torch.device":
    """
    Assert CUDA GPU is available and return torch.device('cuda').
    Supports optional CPU fallback if allow_cpu=True or --cpu is passed.
    """
    if not torch.cuda.is_available():
        if allow_cpu or "--cpu" in sys.argv:
            print("\n[INFO] CUDA GPU not detected. Proceeding on CPU as requested/fallback.")
            return torch.device("cpu")
        print(
            "\n" + "=" * 70 + "\n"
            "ERROR: CUDA GPU not detected.\n"
            "This pipeline was configured for GPU training (e.g. RTX 3050).\n"
            "To run on CPU, pass the --cpu flag.\n\n"
            "Possible fixes:\n"
            "  1. To run on CPU: pass the --cpu argument\n"
            "  2. If using GPU: ensure NVIDIA drivers are installed (run: nvidia-smi)\n"
            "  3. Ensure the CUDA build of PyTorch is installed\n"
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
