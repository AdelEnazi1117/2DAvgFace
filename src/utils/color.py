"""Color normalization helpers for face averaging."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def compute_lab_stats(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    stats_mean = []
    stats_std = []
    mask_bool = mask > 0.05
    if not np.any(mask_bool):
        mask_bool = None
    for c in range(3):
        channel = lab[:, :, c]
        if mask_bool is None:
            m = channel.mean()
            s = channel.std()
        else:
            m = channel[mask_bool].mean()
            s = channel[mask_bool].std()
        stats_mean.append(m)
        stats_std.append(max(s, 1e-6))
    return np.array(stats_mean, dtype=np.float32), np.array(stats_std, dtype=np.float32)


def apply_lab_transfer(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    luminance_strength: float = 0.5,  # Gentler transfer on L channel
    color_strength: float = 0.8,  # Stronger on A/B channels
) -> np.ndarray:
    """Apply LAB color transfer with control over luminance and color channels.
    
    Args:
        image_bgr: Source image (uint8 BGR)
        mask: Face mask (float32 0-1)
        target_mean: Target LAB means
        target_std: Target LAB stds
        luminance_strength: How much to match L channel (0=none, 1=full)
        color_strength: How much to match A/B channels (0=none, 1=full)
        
    Returns:
        Color-transferred image (float32 0-1)
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask_bool = mask > 0.05
    if not np.any(mask_bool):
        mask_bool = None

    for c in range(3):
        channel = lab[:, :, c]
        if mask_bool is None:
            m = channel.mean()
            s = channel.std()
        else:
            m = channel[mask_bool].mean()
            s = channel[mask_bool].std()
        s = max(s, 1e-6)
        
        # Apply different strength for luminance vs color
        if c == 0:  # L channel
            strength = luminance_strength
        else:  # A and B channels
            strength = color_strength
        
        # Blend between original and transferred
        transferred = (channel - m) * (target_std[c] / s) + target_mean[c]
        channel = channel * (1 - strength) + transferred * strength
        lab[:, :, c] = channel

    lab = np.clip(lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
