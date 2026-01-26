"""Shared app configuration (change project name here)."""
from __future__ import annotations

import os

PROJECT_NAME = os.getenv("AVGFACE_NAME", "AvgFace")
PROJECT_TAGLINE = os.getenv(
    "AVGFACE_TAGLINE",
    "2D face averaging made simple.",
)
PROJECT_DESCRIPTION = os.getenv(
    "AVGFACE_DESCRIPTION",
    "Upload a set of portraits and generate a clean, balanced average in minutes.",
)

DEFAULT_HOST = os.getenv("AVGFACE_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("AVGFACE_PORT", "5000"))
