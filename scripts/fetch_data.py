"""Preview or resume downloads from the project's public Google Drive folder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data import PROJECT_ROOT

DRIVE_URL = "https://drive.google.com/drive/folders/1sj9yoyBsmaqMxpnueemZOoPDNrb4p54j"
CATEGORY_FOLDERS = {
    "littering": "1_5936UF3U-Gk3ww-WCLhCb0ABFwdcSFQ",
    "loitering": "1nS70WvhrzQ5iQHws40kJclBwk7jaHJmP",
    "rough_sleeping": "1xH0DsjM04QAhvG0Hi1rRyIKkNGFZ7Ter",
    "smoking": "1N3AA15chTxqpIDR9HxTzgKgXvwTi-AiE",
    "unattended_items": "1x4eQkG-Gb4kR-S7oZ7OHgCex9fUuNDh2",
}


def extract_zip(archive: Path) -> Path:
    """Extract beside the archive, rejecting links and paths outside that folder."""
    destination = archive.with_suffix("")
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError(f"Invalid extraction directory: {destination}")
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            name = PurePosixPath(member.filename.replace("\\", "/"))
            target = root.joinpath(*name.parts).resolve()
            mode = member.external_attr >> 16
            if (
                name.is_absolute()
                or ".." in name.parts
                or not target.is_relative_to(root)
            ):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if mode & 0o170000 == 0o120000:
                raise ValueError(f"Archive contains a symbolic link: {member.filename}")
        destination.mkdir(parents=True, exist_ok=True)
        handle.extractall(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=DRIVE_URL, help="Public Google Drive folder URL"
    )
    parser.add_argument(
        "--category", choices=CATEGORY_FOLDERS, help="Fetch only one project category"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--list-only", action="store_true", help="List files without downloading"
    )
    parser.add_argument(
        "--speed-mbps", type=float, help="Per-file download cap in megabits/second"
    )
    parser.add_argument(
        "--retries", type=int, default=2, help="Retries for a failed folder download"
    )
    parser.add_argument(
        "--extract-zip",
        action="store_true",
        help="Extract ZIP files into adjacent folders",
    )
    args = parser.parse_args(argv)
    if args.category and args.url != DRIVE_URL:
        parser.error("--category cannot be combined with a custom --url")
    if args.retries < 0 or (
        args.speed_mbps is not None and not 0 < args.speed_mbps < float("inf")
    ):
        parser.error(
            "retries must be nonnegative and speed-mbps must be positive and finite"
        )
    try:
        import gdown
    except ImportError:
        parser.exit(
            1, "Install download dependencies: python -m pip install gdown==6.1.1\n"
        )

    url = (
        f"https://drive.google.com/drive/folders/{CATEGORY_FOLDERS[args.category]}"
        if args.category
        else args.url
    )
    output = (
        args.output.resolve() / args.category
        if args.category
        else args.output.resolve()
    )
    # gdown preserves the Drive root folder inside the requested output directory.
    options = {
        "url": url,
        "output": str(output) + os.sep,
        "resume": True,
        "use_cookies": False,
        "speed": args.speed_mbps * 1_000_000 / 8 if args.speed_mbps else None,
    }
    try:
        if args.list_only:
            files = gdown.download_folder(**options, skip_download=True, quiet=True)
            for entry in files:
                print(entry.local_path)
            print(f"{len(files)} files; no dataset files downloaded.")
            return 0
        output.mkdir(parents=True, exist_ok=True)
        for attempt in range(args.retries + 1):
            try:
                files = gdown.download_folder(**options)
                break
            except gdown.DownloadError:
                if attempt == args.retries:
                    raise
                time.sleep(min(2**attempt, 10))
        extracted = [
            str(extract_zip(Path(file)))
            for file in files
            if args.extract_zip and Path(file).suffix.lower() == ".zip"
        ]
        receipt = {
            "source": url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "extracted_directories": extracted,
        }
        (output / "download_receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
        print(f"Downloaded/resumed {len(files)} files into {output}")
        return 0
    except (gdown.DownloadError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(
            f"Download failed: {exc}\nCheck public sharing permissions and Drive quota; rerun to resume.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
