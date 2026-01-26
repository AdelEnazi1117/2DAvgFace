"""Advanced blending utilities for face averaging.

Implements multiple blending strategies:
- Weighted mean blending (baseline)
- Weighted median blending (sharper results)
- Laplacian pyramid blending (best quality, preserves details)
- Per-pixel quality weighting based on local sharpness
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def compute_local_sharpness(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Compute per-pixel sharpness map using Laplacian variance.
    
    Higher values indicate sharper/more detailed regions.
    
    Args:
        image: Input image (float32 or uint8)
        kernel_size: Window size for local variance computation
        
    Returns:
        Sharpness map (height, width) normalized to 0-1
    """
    if image.dtype == np.float32:
        img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = image
    
    # Convert to grayscale if color
    if img_u8.ndim == 3:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_u8
    
    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    
    # Local variance using box filter
    lap_sq = laplacian ** 2
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    local_var = cv2.boxFilter(lap_sq, -1, (k, k))
    
    # Normalize to 0-1
    max_var = local_var.max()
    if max_var > 0:
        local_var = local_var / max_var
    
    return local_var


def blend_mean(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    weights: Optional[List[float]] = None,
    use_local_sharpness: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard weighted mean blending.
    
    Args:
        images: List of aligned face images (float32, 0-1)
        masks: List of per-image masks (float32, 0-1)
        weights: Optional per-image quality weights
        use_local_sharpness: Weight pixels by local sharpness
        
    Returns:
        Tuple of (blended_image, accumulated_mask)
    """
    if not images:
        raise ValueError("No images to blend")
    
    h, w = images[0].shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float32)
    wsum = np.zeros((h, w, 3), dtype=np.float32)
    mask_acc = np.zeros((h, w), dtype=np.float32)
    
    if weights is None:
        weights = [1.0] * len(images)
    
    for img, mask, weight in zip(images, masks, weights):
        # Compute pixel-level weight
        if use_local_sharpness:
            sharpness = compute_local_sharpness(img)
            pixel_weight = mask * float(weight) * (0.5 + 0.5 * sharpness)
        else:
            pixel_weight = mask * float(weight)
        
        weight_map = pixel_weight[:, :, None]
        acc += img * weight_map
        wsum += weight_map
        mask_acc += mask
    
    output = acc / np.maximum(wsum, 1e-6)
    output = np.clip(output, 0.0, 1.0)
    mask_final = np.clip(mask_acc / max(1, len(images)), 0.0, 1.0)
    
    return output, mask_final


def blend_median(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted median blending for sharper results.
    
    Median is more robust to outliers and produces sharper results
    than mean blending, especially when input images have varying quality.
    
    Args:
        images: List of aligned face images (float32, 0-1)
        masks: List of per-image masks (float32, 0-1)
        weights: Optional per-image quality weights (not used for median)
        
    Returns:
        Tuple of (blended_image, accumulated_mask)
    """
    if not images:
        raise ValueError("No images to blend")
    
    h, w = images[0].shape[:2]
    
    # Stack all images
    stack = np.stack(images, axis=0)  # (N, H, W, 3)
    mask_stack = np.stack(masks, axis=0)  # (N, H, W)
    
    # Compute per-pixel median
    output = np.median(stack, axis=0)
    
    # Accumulate masks
    mask_acc = mask_stack.mean(axis=0)
    
    return output.astype(np.float32), mask_acc


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Gaussian pyramid for an image."""
    pyramid = [image.copy()]
    current = image
    
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
    return pyramid


def build_laplacian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Laplacian pyramid for an image."""
    gaussian = build_gaussian_pyramid(image, levels)
    laplacian = []
    
    for i in range(levels - 1):
        h, w = gaussian[i].shape[:2]
        expanded = cv2.pyrUp(gaussian[i + 1], dstsize=(w, h))
        lap = gaussian[i] - expanded
        laplacian.append(lap)
    
    # Top level is just the smallest Gaussian
    laplacian.append(gaussian[-1])
    
    return laplacian


def reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid."""
    current = pyramid[-1]
    
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[:2]
        expanded = cv2.pyrUp(current, dstsize=(w, h))
        current = expanded + pyramid[i]
    
    return current


def blend_laplacian_pyramid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    weights: Optional[List[float]] = None,
    levels: int = 5,
    high_freq_damping: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-resolution blending using Laplacian pyramids.
    
    This is the highest-quality blending method. It blends different
    frequency bands separately, which preserves fine details while
    creating smooth color transitions.
    
    The high_freq_damping parameter reduces contribution of high-frequency
    details from averaged faces, reducing blur while preserving structure.
    
    Args:
        images: List of aligned face images (float32, 0-1)
        masks: List of per-image masks (float32, 0-1)
        weights: Optional per-image quality weights
        levels: Number of pyramid levels
        high_freq_damping: Damping factor for high-freq bands (0-1)
        
    Returns:
        Tuple of (blended_image, accumulated_mask)
    """
    if not images:
        raise ValueError("No images to blend")
    
    h, w = images[0].shape[:2]
    
    # Limit levels based on image size
    max_levels = int(np.log2(min(h, w))) - 2
    levels = min(levels, max(2, max_levels))
    
    if weights is None:
        weights = [1.0] * len(images)
    
    weight_sum = sum(weights)
    norm_weights = [w / weight_sum for w in weights]
    
    # Build Laplacian pyramids for all images
    image_pyramids = [build_laplacian_pyramid(img, levels) for img in images]
    mask_pyramids = [build_gaussian_pyramid(m[:, :, None] if m.ndim == 2 else m, levels) for m in masks]
    
    # Blend each level separately
    blended_pyramid = []
    
    for level_idx in range(levels):
        level_sum = np.zeros_like(image_pyramids[0][level_idx])
        weight_sum_map = np.zeros_like(level_sum)
        
        for img_idx, (img_pyr, mask_pyr) in enumerate(zip(image_pyramids, mask_pyramids)):
            # Weight for this image at this level
            img_weight = norm_weights[img_idx]
            
            # Apply high-frequency damping for lower levels (higher freq)
            if level_idx < levels - 2:
                # Lower levels = higher frequencies
                freq_factor = high_freq_damping ** (levels - 2 - level_idx)
            else:
                freq_factor = 1.0
            
            mask = mask_pyr[level_idx]
            if mask.ndim == 2:
                mask = mask[:, :, None]
            
            weight_map = mask * img_weight * freq_factor
            level_sum += img_pyr[level_idx] * weight_map
            weight_sum_map += weight_map
        
        blended_level = level_sum / np.maximum(weight_sum_map, 1e-6)
        blended_pyramid.append(blended_level)
    
    # Reconstruct from blended pyramid
    output = reconstruct_from_laplacian(blended_pyramid)
    output = np.clip(output, 0.0, 1.0)
    
    # Accumulate masks
    mask_acc = np.mean(np.stack(masks, axis=0), axis=0)
    
    return output, mask_acc


def blend_images(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    weights: Optional[List[float]] = None,
    method: str = "median",
    use_local_sharpness: bool = True,
    pyramid_levels: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unified blending interface.
    
    Args:
        images: List of warped face images (float32, 0-1)
        masks: List of per-image masks (float32, 0-1)
        weights: Optional per-image quality weights
        method: One of "mean", "median", "pyramid"
        use_local_sharpness: For mean blending, weight by local sharpness
        pyramid_levels: For pyramid blending, number of levels
        
    Returns:
        Tuple of (blended_image, final_mask)
    """
    if method == "mean":
        return blend_mean(images, masks, weights, use_local_sharpness)
    elif method == "median":
        return blend_median(images, masks, weights)
    elif method == "pyramid":
        return blend_laplacian_pyramid(images, masks, weights, pyramid_levels)
    else:
        raise ValueError(f"Unknown blend method: {method}")
