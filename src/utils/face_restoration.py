"""Face restoration utilities using GFPGAN and CodeFormer.

This module provides a unified interface for face restoration models
that can restore fine facial details, remove artifacts, and enhance
skin texture in averaged faces.

Models are auto-downloaded on first use to the models/ directory.
"""

from __future__ import annotations

import logging
import os
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

# Model download URLs
GFPGAN_V14_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
CODEFORMER_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
DETECTION_URL = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
PARSING_URL = "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

RestoreMethod = Literal["gfpgan", "codeformer"]


def _ensure_model(filename: str, url: str) -> Path:
    """Download model if not present."""
    return ensure_manifest_model(
        filename=filename,
        fallback_url=url,
        label=filename,
        models_dir=MODELS_DIR,
    )


def _get_device() -> torch.device:
    """Get the best available device, optimized for M3 Pro."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FaceRestorer:
    """Unified interface for GFPGAN and CodeFormer face restoration.
    
    Example:
        restorer = FaceRestorer(method="gfpgan")
        restored = restorer.restore(image)
    """
    
    def __init__(
        self,
        method: RestoreMethod = "gfpgan",
        device: Optional[str] = None,
        bg_upsampler: bool = False,
    ) -> None:
        """Initialize the face restorer.
        
        Args:
            method: Which model to use ("gfpgan" or "codeformer")
            device: Device to use ("cpu", "cuda", "mps", or None for auto)
            bg_upsampler: Whether to upscale background (adds Real-ESRGAN)
        """
        self.method = method
        self.device = torch.device(device) if device else _get_device()
        self.bg_upsampler = bg_upsampler
        self._restorer = None
        self._initialized = False
        
    def _lazy_init(self) -> None:
        """Lazily initialize the model on first use."""
        if self._initialized:
            return
            
        if self.method == "gfpgan":
            self._init_gfpgan()
        else:
            self._init_codeformer()
            
        self._initialized = True
    
    def _init_gfpgan(self) -> None:
        """Initialize GFPGAN v1.4 model."""
        ensure_torchvision_functional_tensor()
        try:
            from gfpgan import GFPGANer
        except ImportError:
            raise ImportError(
                "GFPGAN is required for face restoration. "
                "Install with: pip install gfpgan"
            )
        
        # Ensure required models are downloaded
        model_path = _ensure_model("GFPGANv1.4.pth", GFPGAN_V14_URL)
        _ensure_model("detection_Resnet50_Final.pth", DETECTION_URL)
        _ensure_model("parsing_parsenet.pth", PARSING_URL)
        
        # Set facexlib model root
        os.environ.setdefault("FACEXLIB_MODELROOT", str(MODELS_DIR))
        
        bg_upsampler_instance = None
        if self.bg_upsampler:
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                esrgan_model_path = MODELS_DIR / "RealESRGAN_x2plus.pth"
                if esrgan_model_path.exists():
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                    num_block=23, num_grow_ch=32, scale=2)
                    bg_upsampler_instance = RealESRGANer(
                        scale=2,
                        model_path=str(esrgan_model_path),
                        model=model,
                        tile=400,
                        tile_pad=10,
                        pre_pad=0,
                        half=self.device.type == "cuda",
                    )
            except ImportError:
                logger.warning("Real-ESRGAN not available for background upsampling")
        
        self._restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler_instance,
            device=self.device,
        )
        logger.info(f"Initialized GFPGAN on {self.device}")
    
    def _init_codeformer(self) -> None:
        """Initialize CodeFormer model."""
        ensure_torchvision_functional_tensor()
        try:
            from codeformer import CodeFormer as CF
            from codeformer.basicsr.utils.download_util import load_file_from_url
        except ImportError:
            # Fallback to manual loading
            pass
        
        model_path = _ensure_model("codeformer.pth", CODEFORMER_URL)
        _ensure_model("detection_Resnet50_Final.pth", DETECTION_URL)
        _ensure_model("parsing_parsenet.pth", PARSING_URL)
        
        os.environ.setdefault("FACEXLIB_MODELROOT", str(MODELS_DIR))
        
        # Try to use facexlib for face detection
        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        except ImportError:
            raise ImportError(
                "facexlib is required for CodeFormer. "
                "Install with: pip install facexlib"
            )
        
        self._face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
        )
        
        # Load CodeFormer weights
        from basicsr.utils.registry import ARCH_REGISTRY
        self._net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location="cpu")
        self._net.load_state_dict(checkpoint["params_ema"])
        self._net.eval()
        
        logger.info(f"Initialized CodeFormer on {self.device}")
    
    def restore(
        self,
        image: np.ndarray,
        fidelity: float = 0.7,
        only_center_face: bool = True,
        paste_back: bool = True,
    ) -> np.ndarray:
        """Restore faces in the image.
        
        Args:
            image: Input image (BGR, uint8)
            fidelity: Balance between quality and identity preservation (0-1)
                     Higher = more faithful to input, lower = more enhancement
            only_center_face: Only process the center/largest face
            paste_back: Paste restored face back into original image
            
        Returns:
            Restored image (BGR, uint8)
        """
        self._lazy_init()
        
        if self.method == "gfpgan":
            return self._restore_gfpgan(image, only_center_face, paste_back)
        else:
            return self._restore_codeformer(image, fidelity, only_center_face, paste_back)
    
    def _restore_gfpgan(
        self,
        image: np.ndarray,
        only_center_face: bool,
        paste_back: bool,
    ) -> np.ndarray:
        """Restore using GFPGAN."""
        _, _, output = self._restorer.enhance(
            image,
            has_aligned=False,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )
        return output
    
    def _restore_codeformer(
        self,
        image: np.ndarray,
        fidelity: float,
        only_center_face: bool,
        paste_back: bool,
    ) -> np.ndarray:
        """Restore using CodeFormer."""
        self._face_helper.clean_all()
        self._face_helper.read_image(image)
        
        # Detect faces
        num_det_faces = self._face_helper.get_face_landmarks_5(
            only_center_face=only_center_face,
            resize=640,
            eye_dist_threshold=5,
        )
        
        if num_det_faces == 0:
            logger.warning("No faces detected, returning original image")
            return image
        
        # Align and warp faces
        self._face_helper.align_warp_face()
        
        # Restore each face
        for idx, cropped_face in enumerate(self._face_helper.cropped_faces):
            cropped_face_t = self._img2tensor(cropped_face).to(self.device)
            
            with torch.no_grad():
                output = self._net(cropped_face_t, w=fidelity, adain=True)[0]
                restored_face = self._tensor2img(output)
            
            self._face_helper.add_restored_face(restored_face)
        
        if paste_back:
            self._face_helper.get_inverse_affine(None)
            restored_img = self._face_helper.paste_faces_to_input_image()
        else:
            restored_img = self._face_helper.restored_faces[0]
        
        return restored_img
    
    @staticmethod
    def _img2tensor(img: np.ndarray) -> torch.Tensor:
        """Convert BGR uint8 image to normalized tensor."""
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return img
    
    @staticmethod  
    def _tensor2img(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to BGR uint8 image."""
        output = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
        output = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def apply_face_restoration(
    image: np.ndarray,
    method: RestoreMethod = "gfpgan",
    fidelity: float = 0.7,
    device: Optional[str] = None,
) -> np.ndarray:
    """Convenience function for one-off face restoration.
    
    Args:
        image: Input image (BGR, uint8)
        method: Which model to use ("gfpgan" or "codeformer")
        fidelity: For CodeFormer, balance quality vs identity (0-1)
        device: Device to use (None for auto-detect)
        
    Returns:
        Restored image (BGR, uint8)
    """
    restorer = FaceRestorer(method=method, device=device)
    return restorer.restore(image, fidelity=fidelity)
