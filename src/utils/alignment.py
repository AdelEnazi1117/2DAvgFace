"""Alignment utilities for face averaging.

Enhanced with:
- Full Procrustes alignment using all 478 MediaPipe landmarks
- Weighted landmark importance (eyes > nose > mouth > contour)
- Pose estimation and filtering for extreme angles
- Subpixel landmark refinement
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Dict

import cv2
import numpy as np
import mediapipe as mp


# Landmark region indices for MediaPipe 478-point face mesh
# These weights prioritize more stable/important facial features
LANDMARK_WEIGHTS: Dict[str, Tuple[List[int], float]] = {
    # Eyes are most stable for alignment
    "left_eye": (list(range(33, 42)) + list(range(133, 145)), 1.5),
    "right_eye": (list(range(263, 272)) + list(range(362, 374)), 1.5),
    # Eye corners (canthi) - extremely stable
    "eye_corners": ([33, 133, 362, 263], 2.0),
    # Nose bridge and tip
    "nose": ([1, 2, 4, 5, 6, 19, 94, 168, 197, 195], 1.3),
    # Mouth corners and center
    "mouth": ([61, 291, 0, 17, 78, 308, 80, 310, 13, 14, 82, 312], 1.0),
    # Eyebrows
    "eyebrows": (list(range(46, 53)) + list(range(276, 283)), 0.8),
    # Face contour (less stable but helps with shape)
    "contour": (list(range(10, 11)) + list(range(338, 339)) + [152, 234, 454], 0.5),
}


def get_landmark_weights(num_landmarks: int = 478) -> np.ndarray:
    """Build per-landmark weight array based on region importance."""
    weights = np.ones(num_landmarks, dtype=np.float32) * 0.7  # Default weight
    
    for region_name, (indices, weight) in LANDMARK_WEIGHTS.items():
        for idx in indices:
            if idx < num_landmarks:
                weights[idx] = max(weights[idx], weight)
    
    return weights


def estimate_pose(landmarks: np.ndarray) -> Tuple[float, float, float]:
    """Estimate face pose (yaw, pitch, roll) from 2D landmarks.
    
    Uses geometric relationships between key facial landmarks to estimate
    3D pose angles from 2D projections.
    
    Returns:
        Tuple of (yaw, pitch, roll) in degrees
    """
    if landmarks.shape[0] < 468:
        return 0.0, 0.0, 0.0
    
    # Key landmarks for pose estimation
    nose_tip = landmarks[4]       # Nose tip
    nose_bridge = landmarks[6]    # Between eyes
    left_eye = landmarks[33]      # Left eye outer corner
    right_eye = landmarks[263]    # Right eye outer corner
    chin = landmarks[152]         # Chin center
    
    # Compute eye center
    eye_center = (left_eye + right_eye) / 2.0
    
    # Inter-pupillary distance (for normalization)
    ipd = np.linalg.norm(right_eye - left_eye)
    if ipd < 1e-6:
        return 0.0, 0.0, 0.0
    
    # YAW: Horizontal deviation of nose from eye line center
    # If nose is left of center, yaw is negative (looking right)
    nose_deviation = nose_tip[0] - eye_center[0]
    yaw = np.arctan2(nose_deviation, ipd * 0.5) * 180 / np.pi
    
    # PITCH: Vertical relationship between nose bridge and nose tip
    # Relative to face height
    face_height = np.linalg.norm(chin - nose_bridge)
    if face_height < 1e-6:
        pitch = 0.0
    else:
        vertical_offset = (nose_tip[1] - eye_center[1]) / face_height
        # Normalize expected vertical offset
        pitch = (vertical_offset - 0.35) * 90  # 0.35 is typical frontal ratio
    
    # ROLL: Angle of eye line from horizontal
    eye_vector = right_eye - left_eye
    roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
    
    return float(yaw), float(pitch), float(roll)


def filter_by_pose(
    images: List[np.ndarray],
    points: List[np.ndarray],
    max_yaw: float = 30.0,
    max_pitch: float = 25.0,
    max_roll: float = 20.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Filter out faces with extreme poses.
    
    Returns:
        Tuple of (filtered_images, filtered_points, kept_indices)
    """
    filtered_images = []
    filtered_points = []
    kept_indices = []
    
    for i, (img, pts) in enumerate(zip(images, points)):
        yaw, pitch, roll = estimate_pose(pts)
        
        if abs(yaw) <= max_yaw and abs(pitch) <= max_pitch and abs(roll) <= max_roll:
            filtered_images.append(img)
            filtered_points.append(pts)
            kept_indices.append(i)
    
    return filtered_images, filtered_points, kept_indices


def procrustes_align(
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute optimal similarity transform (scale + rotation + translation)
    using weighted Procrustes analysis on all landmarks.
    
    Args:
        src_landmarks: Source landmark positions (N, 2)
        dst_landmarks: Target landmark positions (N, 2)
        weights: Optional per-landmark weights (N,)
        
    Returns:
        2x3 affine transformation matrix
    """
    src = np.asarray(src_landmarks, dtype=np.float64)
    dst = np.asarray(dst_landmarks, dtype=np.float64)
    
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        w = w / w.sum()  # Normalize weights
    else:
        w = np.ones((src.shape[0], 1), dtype=np.float64) / src.shape[0]
    
    # Weighted centroids
    src_centroid = (src * w).sum(axis=0)
    dst_centroid = (dst * w).sum(axis=0)
    
    # Center the points
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    
    # Weighted covariance matrix
    H = (src_centered * w).T @ dst_centered
    
    # SVD for optimal rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    src_var = ((src_centered ** 2) * w).sum()
    scale = S.sum() / max(src_var, 1e-10)
    
    # Compute translation
    t = dst_centroid - scale * (R @ src_centroid)
    
    # Build 2x3 affine matrix
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:2, :2] = scale * R
    matrix[:, 2] = t
    
    return matrix.astype(np.float32)


def indices_from_connections(connections: Sequence) -> List[int]:
    idx = set()
    for conn in connections:
        if hasattr(conn, "start") and hasattr(conn, "end"):
            idx.add(conn.start)
            idx.add(conn.end)
        else:
            a, b = conn
            idx.add(a)
            idx.add(b)
    return sorted(idx)


def get_face_landmarks(face_landmarker, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = face_landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    h, w = image_bgr.shape[:2]
    points = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
    return points


def compute_crop_rect(points: np.ndarray, image_shape: Tuple[int, int, int], scale: float) -> Optional[Tuple[int, int, int, int]]:
    h, w = image_shape[:2]
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    size = max(x_max - x_min, y_max - y_min) * scale

    x1 = int(max(0, cx - size / 2))
    y1 = int(max(0, cy - size / 2))
    x2 = int(min(w, cx + size / 2))
    y2 = int(min(h, cy + size / 2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def get_feature_centers(points: np.ndarray, left_eye_idx: Sequence[int], right_eye_idx: Sequence[int], lips_idx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_eye = points[list(left_eye_idx)].mean(axis=0)
    right_eye = points[list(right_eye_idx)].mean(axis=0)
    mouth = points[list(lips_idx)].mean(axis=0)
    return left_eye, right_eye, mouth


def get_align_transform(
    points: np.ndarray,
    left_eye_idx: Sequence[int],
    right_eye_idx: Sequence[int],
    lips_idx: Sequence[int],
    width: int,
    height: int,
) -> np.ndarray:
    left_eye, right_eye, mouth = get_feature_centers(points, left_eye_idx, right_eye_idx, lips_idx)
    src = np.float32([left_eye, right_eye, mouth])
    dst = np.float32(
        [
            (0.35 * width, 0.38 * height),
            (0.65 * width, 0.38 * height),
            (0.50 * width, 0.70 * height),
        ]
    )
    return cv2.getAffineTransform(src, dst)


def apply_transform(image: np.ndarray, points: np.ndarray, matrix: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    width, height = size
    warped = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    points = cv2.transform(np.expand_dims(points, axis=0), matrix)[0]
    return warped, points


def refine_similarity_alignment(
    images: List[np.ndarray],
    points: List[np.ndarray],
    width: int,
    height: int,
    iterations: int = 5,
    use_weighted_procrustes: bool = True,
    convergence_threshold: float = 0.5,
    interpolation: int = cv2.INTER_CUBIC,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Refine alignment using all landmarks with weighted Procrustes.
    
    Enhanced version with:
    - Weighted Procrustes using landmark importance (eyes > nose > contour)
    - Early stopping when mean alignment error converges
    - Higher quality interpolation (cubic by default)
    
    Args:
        images: List of aligned face images
        points: List of landmark arrays (N, 2)
        width: Output width
        height: Output height
        iterations: Maximum refinement iterations (default: 5)
        use_weighted_procrustes: Use weighted Procrustes alignment
        convergence_threshold: Stop if mean error change < this (pixels)
        interpolation: OpenCV interpolation flag
        
    Returns:
        Tuple of (refined_images, refined_points)
    """
    if not images or not points:
        return images, points

    refined_images = list(images)
    refined_points = [pts.copy() for pts in points]
    
    # Get landmark weights if using weighted Procrustes
    landmark_weights = None
    if use_weighted_procrustes and len(points[0]) >= 468:
        landmark_weights = get_landmark_weights(len(points[0]))
    
    prev_mean_error = float('inf')

    for iteration in range(max(1, iterations)):
        # Compute weighted mean shape
        pts_stack = np.stack(refined_points, axis=0)
        mean_shape = pts_stack.mean(axis=0)
        
        # Compute current mean alignment error
        errors = np.linalg.norm(pts_stack - mean_shape, axis=2).mean(axis=1)
        current_mean_error = errors.mean()
        
        # Check for convergence
        if abs(prev_mean_error - current_mean_error) < convergence_threshold:
            break
        prev_mean_error = current_mean_error
        
        next_images: List[np.ndarray] = []
        next_points: List[np.ndarray] = []
        
        for img, pts in zip(refined_images, refined_points):
            # Use weighted Procrustes or fallback to cv2 estimator
            if use_weighted_procrustes and landmark_weights is not None:
                matrix = procrustes_align(pts, mean_shape, landmark_weights)
            else:
                matrix, _ = cv2.estimateAffinePartial2D(pts, mean_shape, method=cv2.LMEDS)
                if matrix is None:
                    next_images.append(img)
                    next_points.append(pts)
                    continue
            
            warped = cv2.warpAffine(
                img,
                matrix,
                (width, height),
                flags=interpolation,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            pts_warped = cv2.transform(np.expand_dims(pts, axis=0), matrix)[0]
            next_images.append(warped)
            next_points.append(pts_warped)
        
        refined_images = next_images
        refined_points = next_points

    return refined_images, refined_points
