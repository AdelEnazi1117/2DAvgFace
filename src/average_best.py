#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import torch

from face_parsing_model import BiSeNet
from utils.compat import ensure_torchvision_functional_tensor
from utils.model_download import ensure_manifest_model, ensure_model_file
from utils import (
    indices_from_connections,
    get_face_landmarks,
    compute_crop_rect,
    get_align_transform,
    apply_transform,
    refine_similarity_alignment,
    filter_by_pose,
    estimate_pose,
    add_boundary_points,
    calculate_delaunay_triangles,
    warp_image_delaunay,
    warp_image_tps,
    build_face_mask,
    get_segmentation_weights,
    get_face_parsing_weights,
    get_face_parsing_mask,
    blend_images,
    quality_score,
    landmark_error_score,
    compute_lab_stats,
    apply_lab_transfer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACES_DIR = PROJECT_ROOT / "data" / "faces"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
DEFAULT_SEG_MODEL = PROJECT_ROOT / "models" / "selfie_multiclass_256x256.tflite"
DEFAULT_PARSING_WEIGHTS = PROJECT_ROOT / "models" / "face_parsing_resnet34.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runtime" / "outputs"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
SEG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
PARSING_WEIGHTS_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.pt"


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Best-in-class 2D face averaging (dense landmarks + TPS + face parsing)."
    )
    parser.add_argument("--faces-dir", default=str(DEFAULT_FACES_DIR))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--seg-model", default=str(DEFAULT_SEG_MODEL))
    parser.add_argument("--parsing-weights", default=str(DEFAULT_PARSING_WEIGHTS))
    parser.add_argument("--parsing-backbone", choices=["resnet18", "resnet34"], default="resnet34")
    parser.add_argument("--parsing-size", type=int, default=768)  # Increased from 512
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default="result_best")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-auto-crop", action="store_true")
    parser.add_argument("--crop-scale", type=float, default=1.6)
    parser.add_argument("--mask-expand", type=float, default=1.08)
    parser.add_argument("--feather", type=int, default=31)
    parser.add_argument("--outlier-z", type=float, default=3.0)
    parser.add_argument("--quality-min", type=float, default=0.2)
    parser.add_argument("--quality-blur-ref", type=float, default=500.0)
    parser.add_argument("--quality-size-ref", type=float, default=0.12)
    parser.add_argument("--align-iters", type=int, default=5)  # Increased from 2
    parser.add_argument("--warp-method", choices=["tps", "delaunay"], default="tps")
    parser.add_argument("--warp-grid", type=int, default=4)  # Finer grid (was 8)
    parser.add_argument("--tps-smooth", type=float, default=0.0)
    parser.add_argument("--no-face-parsing", action="store_true")
    parser.add_argument("--no-segmentation", action="store_true")
    parser.add_argument("--hair-weight", type=float, default=None, help="Override hair weight (0-1)")
    parser.add_argument("--body-weight", type=float, default=None, help="Override body/neck weight (0-1)")
    parser.add_argument("--clothes-weight", type=float, default=None, help="Override clothes weight (0-1)")
    parser.add_argument("--accessory-weight", type=float, default=None, help="Override accessory/hat weight (0-1)")
    parser.add_argument("--no-color-transfer", action="store_true")
    parser.add_argument("--background", choices=["gray", "white", "black"], default="gray")
    parser.add_argument("--export-scale", type=float, default=1.0)
    parser.add_argument("--sharpen", type=float, default=0.35)
    parser.add_argument("--sharpen-radius", type=float, default=1.2)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true")
    
    # NEW: Blending mode
    parser.add_argument(
        "--blend-mode",
        choices=["mean", "median", "pyramid"],
        default="median",
        help="Blending method: mean (legacy), median (sharper), pyramid (best quality)"
    )
    
    # NEW: Pose filtering
    parser.add_argument("--max-pose", type=float, default=30.0, help="Reject faces with pose > degrees")
    parser.add_argument("--no-pose-filter", action="store_true", help="Disable pose filtering")
    
    # NEW: Face focus mode - controls which regions are averaged
    parser.add_argument(
        "--face-focus",
        choices=["balanced", "face_only", "strict"],
        default="face_only",
        help="Face region focus: balanced (some hair), face_only (no hair/glasses), strict (core features only)"
    )
    
    # NEW: Edge-aware feathering
    parser.add_argument("--edge-feather", action="store_true", default=True, help="Use edge-aware feathering")
    parser.add_argument("--no-edge-feather", action="store_true", help="Disable edge-aware feathering")
    
    # NEW: Interpolation quality
    parser.add_argument(
        "--interpolation",
        choices=["linear", "cubic", "lanczos"],
        default="cubic",
        help="Warp interpolation: linear (fast), cubic (balanced), lanczos (best)"
    )
    
    # Quality presets (overrides individual settings)
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "max"],
        default="max",
        help="Quality preset: fast (legacy settings), balanced (enhanced), max (all enhancements, default)"
    )
    
    # Enhancement options
    parser.add_argument("--no-enhance", action="store_true", help="Disable all post-processing enhancements")
    parser.add_argument("--restore", action="store_true", help="Enable face restoration (GFPGAN)")
    parser.add_argument(
        "--restore-method",
        choices=["gfpgan", "codeformer"],
        default="gfpgan",
        help="Face restoration method (default: gfpgan)"
    )
    parser.add_argument("--restore-fidelity", type=float, default=0.7, help="Restoration fidelity 0-1 (default: 0.7)")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2, 4], help="Super-resolution scale (1=off, 2=2x, 4=4x)")
    parser.add_argument("--skin-enhance", action="store_true", help="Enhance skin texture")
    parser.add_argument("--skin-strength", type=float, default=0.5, help="Skin enhancement strength 0-1 (default: 0.5)")
    parser.add_argument(
        "--color-grade",
        choices=["none", "natural", "warm", "cool", "cinematic", "vibrant"],
        default="none",
        help="Apply color grading preset"
    )
    parser.add_argument("--auto-levels", action="store_true", default=True, help="Auto-adjust image levels (enabled by default)")
    parser.add_argument("--no-auto-levels", action="store_true", help="Disable auto-levels adjustment")
    
    return parser.parse_args()


def build_background(name: str, width: int, height: int) -> np.ndarray:
    if name == "white":
        val = 1.0
    elif name == "black":
        val = 0.0
    else:
        val = 0.5
    return np.full((height, width, 3), val, dtype=np.float32)


def ensure_model(model_path: Path, url: str) -> None:
    models_dir = PROJECT_ROOT / "models"
    if model_path.parent == models_dir:
        ensure_manifest_model(
            filename=model_path.name,
            fallback_url=url,
            label=model_path.name,
            models_dir=model_path.parent,
        )
        return

    ensure_model_file(
        filename=model_path.name,
        url=url,
        expected_sha256=None,
        label=model_path.name,
        models_dir=model_path.parent,
    )


def unsharp_mask(image: np.ndarray, amount: float = 0.35, radius: float = 1.2) -> np.ndarray:
    if amount <= 0:
        return image
    sigma = max(radius, 0.1)
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    
    # Determine edge-aware feathering setting
    use_edge_feather = args.edge_feather and not args.no_edge_feather
    
    # Map interpolation string to OpenCV constant
    interp_map = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interpolation = interp_map.get(args.interpolation, cv2.INTER_CUBIC)
    
    # Apply quality presets (override individual settings)
    if args.quality == "fast":
        # Legacy settings for speed
        args.no_enhance = True
        args.warp_grid = 8
        args.parsing_size = 512
        args.align_iters = 2
        args.feather = 19
        args.sharpen = 0.25
        args.blend_mode = "mean"
        args.no_pose_filter = True
        use_edge_feather = False
        interpolation = cv2.INTER_LINEAR
    elif args.quality == "balanced":
        # New enhanced defaults (already set in parse_args)
        args.restore = True
        args.upscale = 1
        args.skin_enhance = False
        args.auto_levels = True  # Enable auto-levels for better contrast
        args.feather = 31
        args.sharpen = 0.35
    elif args.quality == "max":
        # Maximum quality settings
        args.restore = True
        args.upscale = 2
        args.skin_enhance = True
        args.auto_levels = True
        args.warp_grid = 2  # Finest grid
        args.parsing_size = 1024
        args.blend_mode = "pyramid"  # Best blending
        args.face_focus = "face_only"  # Focus on features
        interpolation = cv2.INTER_LANCZOS4
        args.feather = 41
        args.sharpen = 0.45
        if args.color_grade == "none":
            args.color_grade = "natural"

    preset_sizes = {
        "fast": (900, 1200),
        "balanced": (1200, 1600),
        "max": (1600, 2200),
    }
    if args.width is None or args.height is None:
        preset_width, preset_height = preset_sizes.get(args.quality, (1600, 2200))
        if args.width is None:
            args.width = preset_width
        if args.height is None:
            args.height = preset_height

    if not args.no_enhance:
        ensure_torchvision_functional_tensor()
        missing = set()

        def require(module: str, package: str) -> None:
            try:
                __import__(module)
            except ImportError:
                missing.add(package)

        if args.restore:
            if args.restore_method == "codeformer":
                require("codeformer", "codeformer")
                require("basicsr", "basicsr")
                require("facexlib", "facexlib")
            else:
                require("gfpgan", "gfpgan")
                require("facexlib", "facexlib")
        if args.upscale > 1:
            require("realesrgan", "realesrgan")
            require("basicsr", "basicsr")

        if missing:
            missing_list = ", ".join(sorted(missing))
            log(f"Missing enhancement libraries: {missing_list}")
            log("Update the environment with: conda env update -f environment.yml --prune")
            return 1
    
    faces_dir = Path(args.faces_dir)
    model_path = Path(args.model)
    seg_model_path = Path(args.seg_model)
    parsing_weights_path = Path(args.parsing_weights)
    output_dir = Path(args.output_dir)

    if not faces_dir.exists():
        faces_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"Created {faces_dir.resolve()}\n"
            "Add face images (.jpg/.jpeg/.png) to this folder and re-run the script."
        )
        return 1

    image_paths = [
        p
        for p in faces_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not image_paths:
        log(f"No images found in {faces_dir.resolve()}. Add .jpg/.jpeg/.png files and re-run.")
        return 1

    ensure_model(model_path, MODEL_URL)
    if not args.no_segmentation:
        ensure_model(seg_model_path, SEG_MODEL_URL)
    if not args.no_face_parsing:
        ensure_model(parsing_weights_path, PARSING_WEIGHTS_URL)

    face_connections = mp.tasks.vision.FaceLandmarksConnections
    left_eye_idx = indices_from_connections(face_connections.FACE_LANDMARKS_LEFT_EYE)
    right_eye_idx = indices_from_connections(face_connections.FACE_LANDMARKS_RIGHT_EYE)
    lips_idx = indices_from_connections(face_connections.FACE_LANDMARKS_LIPS)
    oval_idx = indices_from_connections(face_connections.FACE_LANDMARKS_FACE_OVAL)

    width, height = args.width, args.height

    aligned_images = []
    aligned_points = []
    quality_scores = []

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        log("Detecting landmarks and aligning...")
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                log(f"Skipping unreadable image: {path.name}")
                continue

            points = get_face_landmarks(face_landmarker, img)
            if points is None:
                log(f"No face found in {path.name}")
                continue

            quality = quality_score(img, points, args.quality_blur_ref, args.quality_size_ref)

            if not args.no_auto_crop:
                rect = compute_crop_rect(points[oval_idx], img.shape, args.crop_scale)
                if rect is not None:
                    x1, y1, x2, y2 = rect
                    img = img[y1:y2, x1:x2]
                    points = points.copy()
                    points[:, 0] -= x1
                    points[:, 1] -= y1

            matrix = get_align_transform(points, left_eye_idx, right_eye_idx, lips_idx, width, height)
            warped, warped_points = apply_transform(img, points, matrix, (width, height))

            warped = warped.astype(np.float32) / 255.0
            aligned_images.append(warped)
            aligned_points.append(warped_points)
            quality_scores.append(quality)
            log(f"Aligned {path.name} (quality {quality:.3f})")

    if not aligned_images:
        log("No faces found in any images.")
        return 1

    aligned_images, aligned_points = refine_similarity_alignment(
        aligned_images,
        aligned_points,
        width,
        height,
        iterations=max(1, args.align_iters),
    )

    points_stack = np.stack(aligned_points, axis=0)
    mean_shape = points_stack.mean(axis=0)
    errors = np.linalg.norm((points_stack - mean_shape).reshape(points_stack.shape[0], -1), axis=1)
    ref_error = np.median(errors) + 1e-6

    quality_scores = [
        0.7 * q + 0.3 * landmark_error_score(pts, mean_shape, ref_error)
        for q, pts in zip(quality_scores, aligned_points)
    ]

    z = (errors - errors.mean()) / (errors.std() + 1e-6)
    keep_mask = (z <= args.outlier_z) & (np.array(quality_scores) >= args.quality_min)

    if np.any(~keep_mask) and keep_mask.sum() >= 2:
        aligned_images = [img for img, k in zip(aligned_images, keep_mask) if k]
        aligned_points = [pts for pts, k in zip(aligned_points, keep_mask) if k]
        quality_scores = [q for q, k in zip(quality_scores, keep_mask) if k]
        log(f"Filtered outliers: {np.sum(~keep_mask)} removed")
    elif keep_mask.sum() < 2:
        order = np.argsort(quality_scores)[::-1]
        top_k = min(len(order), 2)
        top = order[:top_k]
        aligned_images = [aligned_images[i] for i in top]
        aligned_points = [aligned_points[i] for i in top]
        quality_scores = [quality_scores[i] for i in top]
        log("Outlier filter too aggressive; using top-quality images instead")

    base_points = np.stack(aligned_points, axis=0)
    mean_shape = base_points.mean(axis=0)

    points_ext_list = [add_boundary_points(p, width, height) for p in aligned_points]
    mean_points_ext = add_boundary_points(mean_shape, width, height)

    rect = (0, 0, width, height)
    triangles = calculate_delaunay_triangles(rect, mean_points_ext)

    fallback_mask = build_face_mask(mean_shape[oval_idx], width, height, args.mask_expand, args.feather)

    segmenter = None
    if not args.no_segmentation:
        seg_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(seg_model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(seg_options)

    parsing_model = None
    parsing_device = None
    if not args.no_face_parsing:
        parsing_device = get_device()
        parsing_model = BiSeNet(num_classes=19, backbone_name=args.parsing_backbone)
        parsing_state = torch.load(parsing_weights_path, map_location="cpu")
        parsing_model.load_state_dict(parsing_state, strict=False)
        parsing_model.to(parsing_device)
        parsing_model.eval()

    # Get parsing weights based on face focus mode
    parsing_weights = get_face_parsing_weights(mode=args.face_focus)
    # Allow CLI overrides for specific regions
    if args.hair_weight is not None:
        parsing_weights[17] = max(0.0, min(args.hair_weight, 1.0))
    if args.clothes_weight is not None:
        parsing_weights[16] = max(0.0, min(args.clothes_weight, 1.0))
    if args.accessory_weight is not None:
        parsing_weights[18] = max(0.0, min(args.accessory_weight, 1.0))
    if args.body_weight is not None:
        parsing_weights[14] = max(0.0, min(args.body_weight, 1.0))
        parsing_weights[15] = max(0.0, min(args.body_weight, 1.0))

    # Build seg_weights from parsing_weights for consistency with face_focus mode
    # Segmentation labels: 0=bg, 1=hair, 2=body, 3=face-skin, 4=cloth, 5=accessory
    # Mapping: seg 1->parsing 17 (hair), seg 2->parsing 14 (neck), seg 4->parsing 16 (cloth), seg 5->parsing 18 (hat)
    seg_weights = {
        0: 0.0,  # background
        1: parsing_weights.get(17, 0.3),  # hair
        2: parsing_weights.get(14, 0.2),  # body/neck
        3: 1.0,  # face-skin - always full weight
        4: parsing_weights.get(16, 0.0),  # cloth
        5: parsing_weights.get(18, 0.0),  # accessory/hat
    }

    try:
        log(f"Warping {len(aligned_images)} faces using {args.warp_method.upper()}...")
        ref_idx = int(np.argmax(quality_scores))
        target_mean = None
        target_std = None

        def warp_face(img, pts):
            if args.warp_method == "tps":
                try:
                    return warp_image_tps(
                        img,
                        pts,
                        mean_points_ext,
                        (width, height),
                        grid_step=args.warp_grid,
                        smooth=args.tps_smooth,
                    )
                except Exception as exc:
                    log(f"TPS warp failed ({exc}); falling back to Delaunay")
            return warp_image_delaunay(img, pts, mean_points_ext, triangles, (width, height))

        # Prepare reference stats
        ref_warped = warp_face(aligned_images[ref_idx], points_ext_list[ref_idx])
        ref_u8 = (np.clip(ref_warped, 0.0, 1.0) * 255.0).astype(np.uint8)
        ref_mask = None
        if parsing_model is not None:
            ref_mask = get_face_parsing_mask(
                parsing_model,
                parsing_device,
                ref_u8,
                args.parsing_size,
                parsing_weights,
                args.feather,
            )
        if ref_mask is None and segmenter is not None:
            ref_mask = get_segmentation_weights(segmenter, ref_u8, seg_weights, args.feather)
        if ref_mask is None:
            ref_mask = fallback_mask
        target_mean, target_std = compute_lab_stats(ref_u8, ref_mask)

        acc = np.zeros((height, width, 3), dtype=np.float32)
        wsum = np.zeros((height, width, 3), dtype=np.float32)
        mask_acc = np.zeros((height, width), dtype=np.float32)

        for idx, (img, pts, weight) in enumerate(zip(aligned_images, points_ext_list, quality_scores), start=1):
            warped = warp_face(img, pts)
            warped_u8 = (np.clip(warped, 0.0, 1.0) * 255.0).astype(np.uint8)

            mask = None
            if parsing_model is not None:
                mask = get_face_parsing_mask(
                    parsing_model,
                    parsing_device,
                    warped_u8,
                    args.parsing_size,
                    parsing_weights,
                    args.feather,
                )

            if mask is None and segmenter is not None:
                mask = get_segmentation_weights(segmenter, warped_u8, seg_weights, args.feather)

            if mask is None:
                mask = fallback_mask

            if not args.no_color_transfer and target_mean is not None:
                warped = apply_lab_transfer(warped_u8, mask, target_mean, target_std)

            weight_map = mask * float(weight)
            acc += warped * weight_map[:, :, None]
            wsum += weight_map[:, :, None]
            mask_acc += mask
            log(f"Warped image {idx}/{len(aligned_images)}")
    finally:
        if segmenter is not None:
            segmenter.close()

    output = acc / np.maximum(wsum, 1e-6)
    output = np.clip(output, 0.0, 1.0)

    mask_final = np.clip(mask_acc / max(1, len(aligned_images)), 0.0, 1.0)
    background = build_background(args.background, width, height)
    output = output * mask_final[:, :, None] + background * (1.0 - mask_final[:, :, None])
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    if args.export_scale and args.export_scale != 1.0:
        out_w = int(width * args.export_scale)
        out_h = int(height * args.export_scale)
        output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    output = unsharp_mask(output, amount=args.sharpen, radius=args.sharpen_radius)

    # ========================================
    # POST-PROCESSING ENHANCEMENTS
    # ========================================
    if not args.no_enhance:
        # Auto-levels (enabled by default for better contrast)
        if args.auto_levels and not getattr(args, 'no_auto_levels', False):
            from utils.enhancement import auto_levels
            log("Applying auto-levels...")
            output = auto_levels(output)
        
        # Face restoration (GFPGAN/CodeFormer)
        if args.restore:
            try:
                from utils.face_restoration import apply_face_restoration
                log(f"Restoring face with {args.restore_method.upper()}...")
                output = apply_face_restoration(
                    output,
                    method=args.restore_method,
                    fidelity=args.restore_fidelity,
                )
            except ImportError as e:
                log(f"Face restoration unavailable: {e}")
                log("Install with: pip install gfpgan facexlib")
            except Exception as e:
                log(f"Face restoration failed: {e}")
        
        # Super-resolution
        if args.upscale > 1:
            try:
                from utils.super_resolution import apply_super_resolution
                log(f"Upscaling {args.upscale}x with Real-ESRGAN...")
                output = apply_super_resolution(output, scale=args.upscale)
            except ImportError as e:
                log(f"Super-resolution unavailable: {e}")
                log("Install with: pip install realesrgan basicsr")
            except Exception as e:
                log(f"Super-resolution failed: {e}")
        
        # Skin texture enhancement
        if args.skin_enhance:
            from utils.enhancement import enhance_skin_texture
            log("Enhancing skin texture...")
            # Create approximate face mask for skin enhancement
            h, w = output.shape[:2]
            skin_mask = np.ones((h, w), dtype=np.float32)
            output = enhance_skin_texture(
                output,
                mask=skin_mask,
                strength=args.skin_strength,
            )
        
        # Color grading
        if args.color_grade != "none":
            from utils.enhancement import color_grade
            log(f"Applying {args.color_grade} color grade...")
            output = color_grade(output, preset=args.color_grade)
    
    log(f"Final output size: {output.shape[1]}x{output.shape[0]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.no_timestamp:
        filename = f"{args.output_name}.jpg"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.output_name}_{stamp}.jpg"
    out_path = output_dir / filename
    cv2.imwrite(str(out_path), output)
    log(f"Saved: {out_path.resolve()}")

    if args.show:
        cv2.imshow("Average Face (Best)", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
