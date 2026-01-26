"""Skin texture and image enhancement utilities.

This module provides functions to enhance facial skin texture,
reduce the "plastic AI look", and apply professional-grade
post-processing for photorealistic results.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


def frequency_separate(
    image: np.ndarray,
    radius: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate image into low and high frequency components.
    
    This is the foundation of professional skin retouching workflows,
    allowing independent manipulation of texture (high freq) and
    color/tone (low freq).
    
    Args:
        image: Input image (BGR, float32 0-1 or uint8)
        radius: Gaussian blur radius for low-pass filter
        
    Returns:
        Tuple of (low_freq, high_freq) images
    """
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()
    
    # Low frequency: blurred version
    sigma = max(radius, 0.1)
    low_freq = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # High frequency: difference from original
    # Using linear light blend mode equivalent
    high_freq = img - low_freq + 0.5
    
    return low_freq, high_freq


def frequency_merge(
    low_freq: np.ndarray,
    high_freq: np.ndarray,
) -> np.ndarray:
    """Merge frequency-separated components back together.
    
    Args:
        low_freq: Low frequency (color/tone) component
        high_freq: High frequency (texture) component
        
    Returns:
        Merged image (same type as inputs)
    """
    merged = low_freq + (high_freq - 0.5)
    return np.clip(merged, 0.0, 1.0)


def add_skin_grain(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    amount: float = 0.02,
    roughness: float = 0.5,
) -> np.ndarray:
    """Add subtle skin-like grain to reduce the plastic AI look.
    
    This adds noise at frequencies matching natural skin pore patterns,
    creating a more organic, photographic appearance.
    
    Args:
        image: Input image (uint8 BGR)
        mask: Optional face mask (grayscale, 0-1) to limit grain to face
        amount: Grain intensity (0-0.1 recommended)
        roughness: Grain size/roughness (0=fine, 1=coarse)
        
    Returns:
        Image with grain applied (uint8 BGR)
    """
    if amount <= 0:
        return image
    
    h, w = image.shape[:2]
    
    # Generate multi-scale noise for natural look
    scales = [1.0, 0.5, 0.25]
    weights = [0.6, 0.3, 0.1]
    
    combined_noise = np.zeros((h, w), dtype=np.float32)
    
    for scale, weight in zip(scales, weights):
        sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
        noise = np.random.randn(sh, sw).astype(np.float32)
        
        if scale < 1.0:
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        
        combined_noise += noise * weight
    
    # Apply roughness (blur for smoother grain)
    if roughness < 1.0:
        blur_radius = int((1.0 - roughness) * 3) * 2 + 1
        combined_noise = cv2.GaussianBlur(combined_noise, (blur_radius, blur_radius), 0)
    
    # Normalize
    combined_noise = combined_noise / np.std(combined_noise)
    
    # Convert to color noise
    color_noise = np.stack([combined_noise] * 3, axis=-1)
    
    # Apply mask if provided
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        color_noise = color_noise * mask
    
    # Apply to image
    img_float = image.astype(np.float32) / 255.0
    result = img_float + color_noise * amount
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    return result


def enhance_skin_texture(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
    grain_amount: float = 0.015,
    detail_boost: float = 0.3,
) -> np.ndarray:
    """Comprehensive skin texture enhancement.
    
    Combines multiple techniques to create natural, photographic skin:
    1. High-frequency detail enhancement
    2. Subtle grain addition for organic feel
    3. Local contrast enhancement for pore visibility
    
    Args:
        image: Input image (uint8 BGR)
        mask: Optional face mask (0-1 float) for targeted enhancement
        strength: Overall enhancement strength (0-1)
        grain_amount: Amount of skin grain to add
        detail_boost: High-frequency detail boost factor
        
    Returns:
        Enhanced image (uint8 BGR)
    """
    if strength <= 0:
        return image
    
    img_float = image.astype(np.float32) / 255.0
    
    # Step 1: Frequency separation
    low_freq, high_freq = frequency_separate(image, radius=4.0)
    
    # Step 2: Boost high-frequency details
    if detail_boost > 0:
        # Enhance contrast in high-freq band
        detail_center = 0.5
        high_freq_enhanced = detail_center + (high_freq - detail_center) * (1.0 + detail_boost * strength)
        high_freq_enhanced = np.clip(high_freq_enhanced, 0.0, 1.0)
    else:
        high_freq_enhanced = high_freq
    
    # Step 3: Merge back
    result = frequency_merge(low_freq, high_freq_enhanced)
    result = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # Step 4: Add skin grain
    if grain_amount > 0:
        result = add_skin_grain(result, mask, amount=grain_amount * strength)
    
    # Step 5: Blend with original based on mask and strength
    if mask is not None:
        if mask.ndim == 2:
            mask_3d = mask[:, :, np.newaxis]
        else:
            mask_3d = mask
        
        blend_mask = mask_3d * strength
        result = (
            image.astype(np.float32) * (1.0 - blend_mask) +
            result.astype(np.float32) * blend_mask
        ).astype(np.uint8)
    
    return result


def smart_sharpen(
    image: np.ndarray,
    amount: float = 0.35,
    radius: float = 1.2,
    threshold: int = 3,
) -> np.ndarray:
    """Smart unsharp mask with threshold to protect smooth areas.
    
    Unlike basic unsharp mask, this avoids sharpening areas
    with very low contrast (like smooth skin), reducing artifacts.
    
    Args:
        image: Input image (uint8)
        amount: Sharpening strength (0-1+)
        radius: Blur radius for mask
        threshold: Minimum difference to sharpen (in uint8 levels)
        
    Returns:
        Sharpened image (uint8)
    """
    if amount <= 0:
        return image
    
    # Create Gaussian blur
    sigma = max(radius, 0.1)
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # Calculate difference (high-frequency)
    diff = cv2.subtract(image, blurred)
    
    # Create threshold mask to protect smooth areas
    if threshold > 0:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
        _, thresh_mask = cv2.threshold(
            cv2.convertScaleAbs(diff_gray),
            threshold,
            255,
            cv2.THRESH_BINARY,
        )
        thresh_mask = cv2.GaussianBlur(thresh_mask, (3, 3), 0)
        thresh_mask = thresh_mask[:, :, np.newaxis] / 255.0 if image.ndim == 3 else thresh_mask / 255.0
        diff = (diff.astype(np.float32) * thresh_mask).astype(np.uint8)
    
    # Apply sharpening
    sharpened = cv2.addWeighted(image, 1.0, diff, amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def local_contrast_enhance(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Enhance local contrast using CLAHE.
    
    Improves micro-contrast and brings out subtle details
    without affecting global brightness.
    
    Args:
        image: Input image (uint8 BGR)
        clip_limit: Contrast limiting parameter
        tile_size: Size of grid for local histogram
        
    Returns:
        Enhanced image (uint8 BGR)
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def auto_levels(
    image: np.ndarray,
    clip_percent: float = 0.3,
    protect_shadows: bool = True,
    protect_highlights: bool = True,
    strength: float = 0.6,
) -> np.ndarray:
    """Auto-adjust levels for optimal dynamic range with face-friendly defaults.

    Uses LAB color space to adjust luminance without color shifts,
    and includes protection for shadows (eyes, eyebrows) and highlights
    to avoid over-darkening or blowing out facial features.

    Args:
        image: Input image (uint8 BGR)
        clip_percent: Percentage of pixels to clip at each end (default 0.3%)
        protect_shadows: Preserve dark areas like eyes (default True)
        protect_highlights: Preserve bright areas (default True)
        strength: Blending strength with original (0-1, default 0.6)

    Returns:
        Adjusted image (uint8 BGR)
    """
    # Convert to LAB for luminance-only adjustments
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    l_original = l_channel.copy()

    # Find histogram percentiles for luminance
    low = np.percentile(l_channel, clip_percent)
    high = np.percentile(l_channel, 100 - clip_percent)

    if high <= low:
        return image

    # Gentle mid-tone focused stretch
    # Instead of full histogram stretch, focus on expanding mid-tones
    mid_point = (low + high) / 2
    target_mid = 127.5

    # Calculate stretch that centers mid-tones but doesn't crush shadows/highlights
    l_stretched = (l_channel - low) * 255.0 / (high - low)
    l_stretched = np.clip(l_stretched, 0, 255)

    # Shadow protection: prevent dark areas from getting darker
    # Simple rule: if a pixel is dark (L < threshold), it can only get brighter, not darker
    if protect_shadows:
        shadow_threshold = 80  # Protect pixels below this L value
        is_shadow = l_original < shadow_threshold

        # For shadow pixels: only allow brightening (take max of stretched and original)
        l_stretched = np.where(
            is_shadow,
            np.maximum(l_stretched, l_original),  # Can only brighten
            l_stretched  # Normal stretch for non-shadows
        )

    # Highlight protection: prevent bright areas from blowing out
    if protect_highlights:
        highlight_threshold = 200  # Protect pixels above this L value
        is_highlight = l_original > highlight_threshold

        # For highlight pixels: only allow darkening (take min of stretched and original)
        l_stretched = np.where(
            is_highlight,
            np.minimum(l_stretched, l_original),  # Can only darken
            l_stretched  # Normal stretch for non-highlights
        )

    # Apply overall strength blending with original
    l_final = l_original * (1 - strength) + l_stretched * strength
    l_final = np.clip(l_final, 0, 255).astype(np.uint8)

    # Gentle saturation boost to compensate for any perceived flatness
    a_channel = lab[:, :, 1].astype(np.float32)
    b_channel = lab[:, :, 2].astype(np.float32)

    # Very gentle color enhancement (5% boost)
    sat_boost = 1.05
    a_enhanced = 128 + (a_channel - 128) * sat_boost
    b_enhanced = 128 + (b_channel - 128) * sat_boost

    # Rebuild LAB and convert back
    lab_result = np.stack([
        l_final,
        np.clip(a_enhanced, 0, 255).astype(np.uint8),
        np.clip(b_enhanced, 0, 255).astype(np.uint8),
    ], axis=-1)

    return cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)


def color_grade(
    image: np.ndarray,
    preset: str = "natural",
) -> np.ndarray:
    """Apply professional color grading preset.
    
    Args:
        image: Input image (uint8 BGR)
        preset: One of "natural", "warm", "cool", "cinematic", "vibrant"
        
    Returns:
        Color-graded image (uint8 BGR)
    """
    presets = {
        "natural": {
            "saturation": 1.05,
            "contrast": 1.05,
            "warmth": 0,
        },
        "warm": {
            "saturation": 1.1,
            "contrast": 1.0,
            "warmth": 10,  # Add to R, subtract from B
        },
        "cool": {
            "saturation": 0.95,
            "contrast": 1.05,
            "warmth": -10,
        },
        "cinematic": {
            "saturation": 0.9,
            "contrast": 1.15,
            "warmth": -5,
            "teal_orange": True,
        },
        "vibrant": {
            "saturation": 1.2,
            "contrast": 1.1,
            "warmth": 5,
        },
    }
    
    if preset not in presets:
        preset = "natural"
    
    params = presets[preset]
    result = image.copy().astype(np.float32)
    
    # Apply warmth (white balance shift)
    warmth = params.get("warmth", 0)
    if warmth != 0:
        result[:, :, 2] = np.clip(result[:, :, 2] + warmth, 0, 255)  # R
        result[:, :, 0] = np.clip(result[:, :, 0] - warmth, 0, 255)  # B
    
    # Convert to HSV for saturation
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params.get("saturation", 1.0), 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Apply contrast
    contrast = params.get("contrast", 1.0)
    if contrast != 1.0:
        mean = np.mean(result)
        result = mean + (result - mean) * contrast
        result = np.clip(result, 0, 255)
    
    # Cinematic teal-orange look (shadows teal, highlights orange)
    if params.get("teal_orange"):
        lab = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        # Shift shadows toward teal (neg A, neg B), highlights toward orange (pos A, pos B)
        l_normalized = lab[:, :, 0] / 255.0
        shadow_mask = 1.0 - l_normalized
        highlight_mask = l_normalized
        
        lab[:, :, 1] = lab[:, :, 1] - shadow_mask * 5 + highlight_mask * 5  # A channel
        lab[:, :, 2] = lab[:, :, 2] - shadow_mask * 10 + highlight_mask * 5  # B channel
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    
    return np.clip(result, 0, 255).astype(np.uint8)
