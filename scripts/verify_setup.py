#!/usr/bin/env python3
"""
Verify that required dependencies are installed.
Run this AFTER activating the avgface conda environment.
"""
import sys
import types


def ensure_torchvision_functional_tensor():
    try:
        import torchvision.transforms.functional_tensor as _  # noqa: F401
        return
    except Exception:
        pass

    try:
        import torchvision.transforms.functional as F
    except Exception:
        return

    module = types.ModuleType("torchvision.transforms.functional_tensor")
    if hasattr(F, "rgb_to_grayscale"):
        module.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = module


def try_import(module):
    try:
        __import__(module)
        return True, None
    except Exception as exc:
        return False, exc

def check_imports():
    missing = []

    # Core dependencies
    required = {
        'cv2': 'opencv',
        'numpy': 'numpy',
        'mediapipe': 'mediapipe',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'scipy': 'scipy',
        'yaml': 'pyyaml',
        'PIL': 'pillow',
        'flask': 'flask',
    }

    # Enhancement libraries (required for default max-quality preset)
    enhancements = {
        'gfpgan': 'gfpgan',
        'realesrgan': 'realesrgan',
        'basicsr': 'basicsr',
        'facexlib': 'facexlib',
    }

    print("Checking core dependencies...")
    for module, package in required.items():
        ok, err = try_import(module)
        if ok:
            print(f"  [OK] {package}")
        else:
            print(f"  [MISSING] {package}")
            missing.append(package)

    print("\nChecking enhancement libraries...")
    ensure_torchvision_functional_tensor()
    for module, package in enhancements.items():
        ok, err = try_import(module)
        if ok:
            print(f"  [OK] {package}")
        else:
            print(f"  [MISSING] {package} ({type(err).__name__}: {err})")
            missing.append(package)

    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo fix, run:")
        print("  conda env update -f environment.yml --prune")
        return False

    print("\nAll dependencies installed.")
    print("\n✓✓✓✓✓ Setup verified successfully! ✓✓✓✓✓")
    return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
