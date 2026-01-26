"""Mask utilities for face averaging.

Enhanced with:
- Edge-aware feathering using guided/bilateral filter
- Distance transform for soft falloff
- Higher resolution face parsing support (768, 1024)
- Improved mask edge quality
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import mediapipe as mp


def edge_aware_feather(
    mask: np.ndarray,
    guide_image: np.ndarray,
    feather_size: int = 31,
    eps: float = 0.01,
) -> np.ndarray:
    """Edge-preserving mask feathering using guided filter.
    
    Unlike Gaussian blur which blurs uniformly, guided filter
    preserves edges from the guide image, resulting in cleaner
    transitions at face boundaries.
    
    Args:
        mask: Binary or soft mask (float32, 0-1)
        guide_image: Image to guide edge preservation (uint8 BGR)
        feather_size: Filter radius
        eps: Regularization (higher = more smoothing)
        
    Returns:
        Edge-aware feathered mask (float32, 0-1)
    """
    if guide_image.dtype != np.uint8:
        guide = (np.clip(guide_image, 0, 1) * 255).astype(np.uint8)
    else:
        guide = guide_image
    
    # Convert to grayscale for guidance
    if guide.ndim == 3:
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
    else:
        guide_gray = guide
    
    # Normalize guide to 0-1 for guided filter
    guide_norm = guide_gray.astype(np.float32) / 255.0
    
    # Apply guided filter
    radius = feather_size // 2
    
    # Guided filter implementation
    mean_I = cv2.boxFilter(guide_norm, -1, (radius, radius))
    mean_p = cv2.boxFilter(mask, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(guide_norm * mask, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(guide_norm * guide_norm, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    
    q = mean_a * guide_norm + mean_b
    
    return np.clip(q, 0.0, 1.0)


def distance_based_falloff(
    mask: np.ndarray,
    falloff_distance: int = 20,
    inner_distance: int = 5,
) -> np.ndarray:
    """Apply distance-based soft falloff to mask edges.
    
    Creates a smooth transition from 1.0 inside the mask
    to 0.0 outside, with controllable falloff distance.
    
    Args:
        mask: Binary mask (0 or 1)
        falloff_distance: Distance (pixels) for falloff outside edge
        inner_distance: Distance (pixels) to keep solid inside
        
    Returns:
        Soft mask with distance-based falloff (float32, 0-1)
    """
    # Binarize mask
    binary = (mask > 0.5).astype(np.uint8)
    
    # Compute distance transform from mask edge
    dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    
    # Create falloff: 1.0 inside, smooth transition at edge
    result = np.zeros_like(mask, dtype=np.float32)
    
    # Inside: fully solid beyond inner_distance
    inside_mask = dist_inside > inner_distance
    result[inside_mask] = 1.0
    
    # Transition zone inside
    transition_inside = (dist_inside > 0) & (dist_inside <= inner_distance)
    result[transition_inside] = dist_inside[transition_inside] / inner_distance
    
    # Outside: falloff
    if falloff_distance > 0:
        falloff_mask = (dist_outside > 0) & (dist_outside < falloff_distance)
        result[falloff_mask] = 1.0 - (dist_outside[falloff_mask] / falloff_distance)
    
    return np.clip(result, 0.0, 1.0)


def build_face_mask(oval_points: np.ndarray, width: int, height: int, expand: float = 1.0, feather: int = 31) -> np.ndarray:
    pts = oval_points.copy()
    center = pts.mean(axis=0)
    pts = (pts - center) * expand + center

    hull = cv2.convexHull(pts.astype(np.float32))
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1.0)

    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0.0, 1.0)

    return mask


def get_segmentation_weights(segmenter, image_bgr: np.ndarray, weights: Dict[int, float], feather: int) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = segmenter.segment(mp_image)
    if result.category_mask is None:
        return None
    mask = result.category_mask.numpy_view().astype(np.uint8)

    weight_map = np.zeros_like(mask, dtype=np.float32)
    for label, value in weights.items():
        weight_map[mask == label] = value

    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        weight_map = cv2.GaussianBlur(weight_map, (k, k), 0)
        weight_map = np.clip(weight_map, 0.0, 1.0)
    return weight_map


def get_face_parsing_weights(mode: str = "balanced") -> Dict[int, float]:
    """Get face parsing weights for different averaging modes.
    
    Label mapping from face-parsing (0 = background):
    0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
    6: eye_g (glasses), 7: l_ear, 8: r_ear, 9: ear_r, 10: nose,
    11: mouth, 12: u_lip, 13: l_lip, 14: neck, 15: neck_l,
    16: cloth, 17: hair, 18: hat
    
    Args:
        mode: "balanced" (include some hair/ears), "face_only" (core features only),
              or "strict" (only skin, eyes, nose, lips)
    
    Returns:
        Dictionary of class_id -> weight (0-1)
    """
    if mode == "strict":
        # STRICT: Only core facial features, no hair/ears/neck
        return {
            0: 0.0,   # background
            1: 1.0,   # skin - full weight
            2: 0.9,   # l_brow
            3: 0.9,   # r_brow
            4: 1.0,   # l_eye - full weight
            5: 1.0,   # r_eye - full weight
            6: 0.0,   # glasses - exclude
            7: 0.0,   # l_ear - exclude
            8: 0.0,   # r_ear - exclude
            9: 0.0,   # ear_r - exclude
            10: 1.0,  # nose - full weight
            11: 0.9,  # mouth
            12: 1.0,  # u_lip - full weight
            13: 1.0,  # l_lip - full weight
            14: 0.0,  # neck - exclude
            15: 0.0,  # neck_l - exclude
            16: 0.0,  # cloth - exclude
            17: 0.0,  # hair - exclude
            18: 0.0,  # hat - exclude
        }
    elif mode == "face_only":
        # FACE_ONLY: Core features + minimal ears, no hair
        return {
            0: 0.0,   # background
            1: 1.0,   # skin
            2: 0.85,  # l_brow
            3: 0.85,  # r_brow
            4: 1.0,   # l_eye
            5: 1.0,   # r_eye
            6: 0.0,   # glasses - exclude
            7: 0.2,   # l_ear - minimal
            8: 0.2,   # r_ear - minimal
            9: 0.1,   # ear_r
            10: 1.0,  # nose
            11: 0.9,  # mouth
            12: 0.95, # u_lip
            13: 0.95, # l_lip
            14: 0.1,  # neck - minimal
            15: 0.0,  # neck_l
            16: 0.0,  # cloth
            17: 0.0,  # hair - exclude
            18: 0.0,  # hat - exclude
        }
    else:
        # BALANCED: Default - includes some hair for natural look
        return {
            0: 0.0,   # background
            1: 1.0,   # skin
            2: 0.8,   # l_brow
            3: 0.8,   # r_brow
            4: 0.95,  # l_eye
            5: 0.95,  # r_eye
            6: 0.0,   # glasses - exclude
            7: 0.3,   # l_ear
            8: 0.3,   # r_ear
            9: 0.2,   # ear_r
            10: 0.95, # nose
            11: 0.85, # mouth
            12: 0.9,  # u_lip
            13: 0.9,  # l_lip
            14: 0.3,  # neck
            15: 0.1,  # neck_l
            16: 0.0,  # cloth
            17: 0.3,  # hair - reduced from 0.6
            18: 0.0,  # hat
        }



@torch.no_grad()
def get_face_parsing_mask(
    model,
    device,
    image_bgr: np.ndarray,
    input_size: int = 768,  # Increased from 512
    weights: Optional[Dict[int, float]] = None,
    feather: int = 21,  # Reduced from 31
    use_edge_aware: bool = True,  # New: use guided filter
) -> np.ndarray:
    """Generate face parsing mask with optional edge-aware feathering.
    
    Enhanced version with:
    - Higher default resolution (768 vs 512) for finer detail
    - Edge-aware feathering option for sharper boundaries
    - Reduced default feather size for less blur
    
    Args:
        model: BiSeNet face parsing model
        device: Torch device
        image_bgr: Input image (uint8 BGR)
        input_size: Parsing network input resolution
        weights: Per-class weight dictionary
        feather: Feathering size in pixels
        use_edge_aware: Use guided filter instead of Gaussian
        
    Returns:
        Weight map (float32, 0-1)
    """
    if weights is None:
        weights = get_face_parsing_weights()
    
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.to(device)

    output = model(tensor)[0]
    pred = output.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)

    weight_map = np.zeros((height, width), dtype=np.float32)
    for label, value in weights.items():
        if value <= 0:
            continue
        weight_map[pred == label] = value

    if feather > 0:
        if use_edge_aware:
            # Use edge-aware feathering for sharper boundaries
            weight_map = edge_aware_feather(weight_map, image_bgr, feather)
        else:
            # Fallback to Gaussian blur
            k = feather if feather % 2 == 1 else feather + 1
            weight_map = cv2.GaussianBlur(weight_map, (k, k), 0)
        weight_map = np.clip(weight_map, 0.0, 1.0)
    
    return weight_map

