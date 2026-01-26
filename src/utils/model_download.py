"""Model download helpers with checksum verification."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "models" / "manifest.json"


def _load_manifest() -> Dict[str, Dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except Exception as exc:
        logger.warning(f"Failed to read model manifest: {exc}")
        return {}
    return data.get("models", {}) if isinstance(data, dict) else {}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_checksum(path: Path, expected: str) -> bool:
    actual = _sha256(path)
    return actual.lower() == expected.lower()


def ensure_model_file(
    filename: str,
    url: str,
    expected_sha256: Optional[str] = None,
    label: Optional[str] = None,
    models_dir: Optional[Path] = None,
) -> Path:
    """Ensure a model file exists and matches checksum when provided.

    If a file already exists and the checksum mismatches, we warn but continue
    to avoid breaking existing setups. If we download a file and the checksum
    mismatches, we delete it and raise.
    """
    models_dir = models_dir or (PROJECT_ROOT / "models")
    model_path = models_dir / filename

    if model_path.exists():
        if expected_sha256 and not _verify_checksum(model_path, expected_sha256):
            logger.warning(
                "Checksum mismatch for %s. Expected %s, got %s. "
                "Continuing with existing file.",
                filename,
                expected_sha256,
                _sha256(model_path),
            )
        return model_path

    models_dir.mkdir(parents=True, exist_ok=True)
    display = label or filename
    logger.info("Downloading %s from %s...", display, url)
    urlretrieve(url, model_path)

    if expected_sha256 and not _verify_checksum(model_path, expected_sha256):
        actual = _sha256(model_path)
        try:
            model_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Checksum mismatch for {filename}. Expected {expected_sha256}, got {actual}."
        )
    return model_path


def ensure_manifest_model(
    filename: str,
    fallback_url: Optional[str] = None,
    fallback_sha256: Optional[str] = None,
    label: Optional[str] = None,
    models_dir: Optional[Path] = None,
) -> Path:
    """Ensure model using manifest entry when available."""
    manifest = _load_manifest()
    entry = manifest.get(filename, {}) if isinstance(manifest, dict) else {}

    url = entry.get("url") or fallback_url
    if not url:
        raise ValueError(f"Missing download URL for {filename}")

    expected_sha256 = entry.get("sha256") or fallback_sha256
    display = label or entry.get("source") or filename
    return ensure_model_file(
        filename=filename,
        url=url,
        expected_sha256=expected_sha256,
        label=display,
        models_dir=models_dir,
    )
