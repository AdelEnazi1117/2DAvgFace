"""Quality scoring helpers for face averaging."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def face_bbox_from_points(points: np.ndarray) -> Tuple[float, float, float, float]:
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())
    return x_min, y_min, x_max, y_max


def laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_score(
    image_bgr: np.ndarray,
    points: np.ndarray,
    blur_ref: float = 500.0,
    size_ref: float = 0.12,
) -> float:
    """Return a 0-1 quality score based on blur + face size."""
    h, w = image_bgr.shape[:2]
    x_min, y_min, x_max, y_max = face_bbox_from_points(points)
    area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_ratio = area / max(1.0, float(h * w))

    blur = laplacian_variance(image_bgr)
    blur_score = np.clip(blur / max(blur_ref, 1.0), 0.0, 1.0)
    size_score = np.clip(area_ratio / max(size_ref, 1e-6), 0.0, 1.0)

    return float(0.6 * blur_score + 0.4 * size_score)


def landmark_error_score(
    points: np.ndarray,
    mean_shape: np.ndarray,
    ref_error: float,
) -> float:
    """Return a 0-1 score based on landmark deviation from mean."""
    diff = points - mean_shape
    err = float(np.linalg.norm(diff.reshape(-1)))
    denom = max(ref_error, 1e-6)
    score = np.exp(-err / denom)
    return float(np.clip(score, 0.0, 1.0))
