"""Super-resolution utilities using Real-ESRGAN.

This module provides image upscaling capabilities for producing
high-resolution output suitable for 4K displays and social media.

Models are auto-downloaded on first use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch

from utils.compat import ensure_torchvision_functional_tensor
from utils.model_download import ensure_manifest_model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
ESRGAN_MODELS = {
    "x2": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
    },
    "x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
    },
    "anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
    },
}

UpscaleModel = Literal["x2", "x4", "anime"]


def _ensure_model(filename: str, url: str) -> Path:
    """Download model if not present."""
    return ensure_manifest_model(
        filename=filename,
        fallback_url=url,
        label=filename,
        models_dir=MODELS_DIR,
    )


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SuperResolver:
    """Real-ESRGAN-based super-resolution upscaler.
    
    Example:
        resolver = SuperResolver(model="x2")
        upscaled = resolver.upscale(image)
    """
    
    def __init__(
        self,
        model: UpscaleModel = "x2",
        device: Optional[str] = None,
        tile_size: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
    ) -> None:
        """Initialize the super-resolver.
        
        Args:
            model: Which model to use ("x2", "x4", or "anime")
            device: Device to use ("cpu", "cuda", "mps", or None for auto)
            tile_size: Tile size for processing (0 = no tiling)
            tile_pad: Padding between tiles
            pre_pad: Pre-padding to avoid border artifacts
        """
        self.model_name = model
        self.device = torch.device(device) if device else _get_device()
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self._upsampler = None
        self._initialized = False
        
    @property
    def scale(self) -> int:
        """Get the upscale factor for current model."""
        return ESRGAN_MODELS[self.model_name]["scale"]
    
    def _lazy_init(self) -> None:
        """Lazily initialize the model on first use."""
        if self._initialized:
            return
        
        ensure_torchvision_functional_tensor()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "Real-ESRGAN is required for super-resolution. "
                "Install with: pip install realesrgan basicsr"
            )
        
        config = ESRGAN_MODELS[self.model_name]
        model_path = _ensure_model(config["filename"], config["url"])
        
        # Build the network architecture
        if self.model_name == "anime":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=config["num_block"],
                num_grow_ch=32,
                scale=config["scale"],
            )
        else:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=config["num_block"],
                num_grow_ch=32,
                scale=config["scale"],
            )
        
        # MPS doesn't support half precision well
        use_half = self.device.type == "cuda"
        
        # Determine tile size based on device memory
        tile = self.tile_size
        if tile == 0:
            if self.device.type == "mps":
                tile = 256  # Conservative for MPS
            elif self.device.type == "cuda":
                tile = 400
            # CPU doesn't need tiling
        
        self._upsampler = RealESRGANer(
            scale=config["scale"],
            model_path=str(model_path),
            model=model,
            tile=tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=use_half,
            device=self.device,
        )
        
        self._initialized = True
        logger.info(f"Initialized Real-ESRGAN ({self.model_name}) on {self.device}")
    
    def upscale(
        self,
        image: np.ndarray,
        outscale: Optional[float] = None,
    ) -> np.ndarray:
        """Upscale an image.
        
        Args:
            image: Input image (BGR, uint8)
            outscale: Final output scale. If None, uses model's native scale.
                     Can be used to get non-integer scales (e.g., 1.5x with x2 model)
                     
        Returns:
            Upscaled image (BGR, uint8)
        """
        self._lazy_init()
        
        output, _ = self._upsampler.enhance(image, outscale=outscale)
        return output


def apply_super_resolution(
    image: np.ndarray,
    scale: int = 2,
    device: Optional[str] = None,
) -> np.ndarray:
    """Convenience function for one-off super-resolution.
    
    Args:
        image: Input image (BGR, uint8)
        scale: Target scale (2 or 4)
        device: Device to use (None for auto-detect)
        
    Returns:
        Upscaled image (BGR, uint8)
    """
    model = "x2" if scale <= 2 else "x4"
    resolver = SuperResolver(model=model, device=device)
    return resolver.upscale(image, outscale=scale)


def intelligent_upscale(
    image: np.ndarray,
    target_min_dimension: int = 1080,
    max_scale: int = 4,
    device: Optional[str] = None,
) -> np.ndarray:
    """Intelligently upscale to meet minimum dimension requirement.
    
    Args:
        image: Input image (BGR, uint8)
        target_min_dimension: Minimum size for shortest side
        max_scale: Maximum upscale factor allowed
        device: Device to use
        
    Returns:
        Upscaled image (BGR, uint8)
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    
    if min_dim >= target_min_dimension:
        logger.info(f"Image already meets target ({min_dim}px >= {target_min_dimension}px)")
        return image
    
    required_scale = target_min_dimension / min_dim
    
    if required_scale > max_scale:
        logger.warning(
            f"Required scale {required_scale:.2f}x exceeds max {max_scale}x. "
            f"Capping at {max_scale}x."
        )
        required_scale = max_scale
    
    # Use appropriate model
    if required_scale <= 2.0:
        model = "x2"
    else:
        model = "x4"
    
    resolver = SuperResolver(model=model, device=device)
    return resolver.upscale(image, outscale=required_scale)
