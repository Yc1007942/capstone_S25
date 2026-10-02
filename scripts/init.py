"""Inventory images and optionally sample video frames into a reviewable manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data import (
    DEFAULT_DEFINITIONS,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    VIDEO_EXTENSIONS,
    read_definitions,
)


def suggested_label(path: Path, definitions: dict[str, str]) -> str:
    parts = {re.sub(r"[\s-]+", "_", part.lower()) for part in path.parts[:-1]}
    candidates = [label for label in definitions if label in parts]
    return candidates[0] if len(candidates) == 1 else ""


def sample_video(
    video: Path, output: Path, fps: float, max_frames: int, max_side: int, threads: int
) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        raise ValueError(
            "ffmpeg is required for --sample-videos; install it with your OS package manager"
        )
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("frame_*.jpg"))
    if existing:
        raise ValueError(
            f"Frames already exist in {output}; use a new --frames-dir for another sampling run"
        )
    scale = f"scale=w=min({max_side}\\,iw):h=min({max_side}\\,ih):force_original_aspect_ratio=decrease"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-threads",
        str(threads),
        "-i",
        str(video),
        "-an",
        "-sn",
        "-vf",
        f"fps={fps},{scale}",
        "-filter_threads",
        str(threads),
        "-threads",
        str(threads),
        "-frames:v",
        str(max_frames),
        str(output / "frame_%06d.jpg"),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=300, check=False
    )
    if result.returncode:
        raise ValueError(f"ffmpeg failed for {video}: {result.stderr.strip()}")
    return sorted(output.glob("frame_*.jpg"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data/manifest.csv"
    )
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS)
    parser.add_argument(
        "--sample-videos",
        action="store_true",
        help="Extract bounded frame samples with ffmpeg",
    )
    parser.add_argument("--frames-dir", type=Path, default=PROJECT_ROOT / "data/frames")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.2,
        help="Sample rate; 0.2 means one frame every 5 seconds",
    )
    parser.add_argument("--max-frames-per-video", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.fps)
        or args.fps <= 0
        or min(args.max_frames_per_video, args.max_side, args.threads) < 1
    ):
        parser.error(
            "fps must be positive and finite; frame, size, and thread limits must be at least 1"
        )
    if args.manifest.exists():
        parser.error(
            "Manifest already exists; choose another --manifest to preserve reviewed labels"
        )
    if not args.data_dir.is_dir():
        parser.error("data-dir does not exist; run scripts/fetch_data.py first")
    try:
        definitions = read_definitions(args.definitions)
        root = args.data_dir.resolve()
        frames_root = args.frames_dir.resolve()
        if args.sample_videos and frames_root.is_relative_to(root):
            raise ValueError(
                "frames-dir must be outside data-dir to prevent indexing frames twice"
            )
        files = sorted(path for path in root.rglob("*") if path.is_file())
        rows = []
        videos = 0
        for path in files:
            relative = path.relative_to(root)
            suggestion = suggested_label(relative, definitions)
            group = relative.as_posix()
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((path, group, suggestion))
            elif path.suffix.lower() in VIDEO_EXTENSIONS:
                videos += 1
                if args.sample_videos:
                    digest = hashlib.sha256(group.encode()).hexdigest()[:12]
                    output = frames_root / f"{path.stem}_{digest}"
                    frames = sample_video(
                        path,
                        output,
                        args.fps,
                        args.max_frames_per_video,
                        args.max_side,
                        args.threads,
                    )
                    rows.extend((frame, group, suggestion) for frame in frames)
                    print(f"Sampled {len(frames)} frames: {relative}")
        if not rows:
            raise ValueError(
                "No images found. Extract archives or use --sample-videos for video data"
            )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "split", "group_id", "suggested_label"])
            for path, group, suggestion in rows:
                writer.writerow(
                    [
                        os.path.relpath(path, args.manifest.parent.resolve()),
                        "",
                        "test",
                        group,
                        suggestion,
                    ]
                )
        print(f"Wrote {len(rows)} images to {args.manifest}; found {videos} videos.")
        print(
            "Review label and split columns before evaluating accuracy. Folder suggestions are not ground truth."
        )
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
