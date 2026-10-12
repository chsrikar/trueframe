"""
torch_compat.py — Compatibility shim for PyTorch and Torchvision.
Resolves 'RuntimeError: operator torchvision::nms does not exist' on Python 3.14 / PyTorch 2.14+
when torchvision C++ operators are not registered in the Torch dispatcher.
"""
import torch

try:
    # torchvision._meta_registrations attempts to register fake kernels for C++ ops
    # (nms, qnms, roi_align, etc.) that do not exist in the Torch dispatcher when C++
    # extensions are omitted on Python 3.14. A no-op decorator bypasses this harmlessly.
    torch.library.register_fake = lambda name, *args, **kwargs: (lambda fn: fn)
except Exception:
    pass
