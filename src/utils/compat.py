"""Compatibility helpers for third-party libraries."""

from __future__ import annotations

import sys
import types


def ensure_torchvision_functional_tensor() -> None:
    """Provide a shim for torchvision.transforms.functional_tensor if missing.

    basicsr/gfpgan import torchvision.transforms.functional_tensor.rgb_to_grayscale,
    which was removed in newer torchvision versions. We expose the function from
    torchvision.transforms.functional under the legacy module path.
    """
    try:
        import torchvision.transforms.functional_tensor as _  # noqa: F401
        return
    except Exception:
        pass

    try:
        import torchvision.transforms.functional as F
    except Exception:
        return

    module = types.ModuleType("torchvision.transforms.functional_tensor")
    if hasattr(F, "rgb_to_grayscale"):
        module.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = module
