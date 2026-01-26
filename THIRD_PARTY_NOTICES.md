# Third-Party Notices

This project uses external datasets, models, and libraries. Licenses remain with their
respective owners. Model download URLs and hashes are tracked in `models/manifest.json`.

## Dataset
- Human Faces dataset by Ashwin Gupta (Kaggle) — CC0: Public Domain (per Kaggle listing).
  Source: https://www.kaggle.com/datasets/ashwingupta3012/human-faces

## Models / Weights
- MediaPipe Face Landmarker model (`face_landmarker.task`) — Apache-2.0.
  Source: https://github.com/google/mediapipe
- MediaPipe Selfie Segmentation model (`selfie_multiclass_256x256.tflite`) — Apache-2.0.
  Source: https://github.com/google/mediapipe
- Face parsing weights (`face_parsing_resnet34.pt`) from yakhyo/face-parsing — MIT License.
  Source: https://github.com/yakhyo/face-parsing
- GFPGAN weights (`GFPGANv1.4.pth`) — Apache-2.0.
  Source: https://github.com/TencentARC/GFPGAN
- Real-ESRGAN weights (e.g. `RealESRGAN_x2plus.pth`) — BSD-3-Clause.
  Source: https://github.com/xinntao/Real-ESRGAN
- facexlib weights (`detection_Resnet50_Final.pth`, `parsing_parsenet.pth`) — MIT License.
  Source: https://github.com/xinntao/facexlib
- CodeFormer weights (`codeformer.pth`, optional) — NTU S-Lab License 1.0.
  Source: https://github.com/sczhou/CodeFormer

## Fonts (loaded at runtime)
- Fraunces, Space Grotesk via Google Fonts. See Google Fonts for license details.
