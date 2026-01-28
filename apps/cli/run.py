#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.config import PROJECT_NAME, DEFAULT_HOST, DEFAULT_PORT
AVERAGE_SCRIPT = PROJECT_ROOT / "src" / "average_best.py"
WEB_SCRIPT = PROJECT_ROOT / "apps" / "web" / "app.py"
DEFAULT_FACES_DIR = PROJECT_ROOT / "data" / "faces"
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples" / "faces_example"

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
QUALITY_DEFAULT_SIZES = {
    "fast": (900, 1200),
    "balanced": (1200, 1600),
    "max": (1600, 2200),
}


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("Size must be in WIDTHxHEIGHT format, e.g. 1200x1600.")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    return width, height


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len([p for p in folder.iterdir() if p.suffix.lower() in ALLOWED_EXTS])


def build_run_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("run", help="Run the 2D averaging pipeline")
    parser.add_argument(
        "--faces",
        type=str,
        help="Folder of input images (jpg/png). Default: data/faces",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Use images from data/examples/faces_example",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "max"],
        default="max",
        help="Quality preset (default: max)",
    )
    parser.add_argument(
        "--size",
        type=str,
        help="Output size as WIDTHxHEIGHT (overrides preset size).",
    )
    parser.add_argument(
        "--background",
        choices=["gray", "white", "black"],
        default="gray",
        help="Background color (default: gray)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a preview window after processing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: runtime/outputs)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output filename without extension.",
    )
    return parser


def build_web_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("web", help="Start the web interface")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--debug", action="store_true")
    return parser


def resolve_faces_dir(args: argparse.Namespace) -> Path:
    if args.examples:
        return EXAMPLES_DIR
    if args.faces:
        return Path(args.faces)
    return DEFAULT_FACES_DIR


def run_subprocess(cmd: list[str], *, cwd: Path, on_interrupt_msg: str | None = None) -> int:
    process = subprocess.Popen(cmd, cwd=str(cwd))
    try:
        return process.wait()
    except KeyboardInterrupt:
        if on_interrupt_msg:
            print(f"\n{on_interrupt_msg}")

        # Forward Ctrl+C to child, then fall back to terminate/kill if needed.
        try:
            process.send_signal(signal.SIGINT)
        except Exception:
            pass

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                process.wait()

        return 130


def run_pipeline(args: argparse.Namespace, extra: list[str]) -> int:
    faces_dir = resolve_faces_dir(args)
    image_count = count_images(faces_dir)
    if image_count < 2:
        location = "example set" if args.examples else "data/faces"
        print(f"No usable images found in {faces_dir}.")
        print(f"Add at least 2 jpg/png files to {location} and try again.")
        return 1

    cmd = [
        sys.executable,
        str(AVERAGE_SCRIPT),
        "--faces-dir",
        str(faces_dir),
        "--background",
        args.background,
    ]

    if args.quality:
        cmd.extend(["--quality", args.quality])

    if args.size:
        width, height = parse_size(args.size)
        cmd.extend(["--width", str(width), "--height", str(height)])
    else:
        width, height = QUALITY_DEFAULT_SIZES.get(args.quality, (900, 1200))
        cmd.extend(["--width", str(width), "--height", str(height)])

    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.output_name:
        cmd.extend(["--output-name", args.output_name])
    if args.show:
        cmd.append("--show")

    cmd.extend(extra)
    return run_subprocess(cmd, cwd=PROJECT_ROOT, on_interrupt_msg="Processing canceled.")


def run_web(args: argparse.Namespace, extra: list[str]) -> int:
    cmd = [
        sys.executable,
        str(WEB_SCRIPT),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.debug:
        cmd.append("--debug")
    cmd.extend(extra)
    return run_subprocess(cmd, cwd=PROJECT_ROOT, on_interrupt_msg="Server stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="avgface",
        description=f"{PROJECT_NAME} command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command")
    build_run_parser(subparsers)
    build_web_parser(subparsers)

    args, extra = parser.parse_known_args()
    command = args.command or "run"

    if command == "web":
        return run_web(args, extra)
    return run_pipeline(args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
