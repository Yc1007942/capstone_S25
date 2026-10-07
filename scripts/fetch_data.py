"""Preview or resume downloads from the project's public Google Drive folder."""

from __future__ import annotations

import argparse
import fnmatch
import inspect
import json
import math
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


class DownloadPacer:
    """Space download attempts; gdown may make several HTTP requests per attempt."""

    def __init__(self, interval: float):
        self.interval = interval
        self.last_start: float | None = None

    def wait(self) -> None:
        if self.last_start is not None:
            delay = self.interval - (time.monotonic() - self.last_start)
            if delay > 0:
                time.sleep(delay)
        self.last_start = time.monotonic()


def matches_patterns(path: str, patterns: list[str]) -> bool:
    normalized = path.replace("\\", "/").casefold()
    return not patterns or any(
        fnmatch.fnmatchcase(normalized, pattern.replace("\\", "/").casefold())
        for pattern in patterns
    )


def download_entry(
    gdown,
    entry,
    *,
    use_cookies: bool,
    speed: float | None,
    retries: int,
    cookies_file: str | None = None,
    pacer: DownloadPacer | None = None,
) -> str:
    """Download one listed file, preserving completed and partial transfers."""
    destination = Path(entry.local_path)
    # gdown writes incomplete transfers to a .part file and renames on success.
    # Empty YOLO annotation files are valid completed downloads too.
    if destination.is_file():
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Google-native documents have no extension in the folder listing. Let
    # gdown resolve their exported filename from the response headers.
    download_output = (
        str(destination) if destination.suffix else str(destination.parent) + os.sep
    )
    for attempt in range(retries + 1):
        try:
            if pacer is not None:
                pacer.wait()
            cookie_options = {"cookies_file": cookies_file} if cookies_file else {}
            result = gdown.download(
                id=entry.id,
                output=download_output,
                resume=True,
                use_cookies=use_cookies,
                speed=speed,
                **cookie_options,
            )
            if result is None:
                raise gdown.DownloadError(
                    "No file returned; install gdown==6.4.1 and retry"
                )
            return str(result)
        except gdown.FileURLRetrievalError:
            # Repeated requests do not repair permissions or a quota block.
            raise
        except (gdown.DownloadError, OSError):
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 10))
    raise RuntimeError("No download attempt made")


def write_receipt(output: Path, receipt: dict) -> None:
    path = output / "download_receipt.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        "--retries",
        type=int,
        default=2,
        help="Retries per file for transient transfer failures",
    )
    parser.add_argument(
        "--use-cookies",
        action="store_true",
        help="Opt in to gdown's existing ~/.cache/gdown/cookies.txt; no browser cookies are imported",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        help="Explicit Netscape cookies file; enables cookie use (requires gdown 6.4.1)",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between file-download attempts (default: 1; 0 disables pacing)",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Stop after this many consecutive failed network downloads, even with --continue-on-error (default: 5; 0 disables)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Download only matching relative paths; repeat to include several patterns, e.g. '*.zip'",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Try remaining files after a failure; still return exit code 1 for an incomplete download",
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
    if (
        not math.isfinite(args.request_interval)
        or args.request_interval < 0
        or args.max_consecutive_errors < 0
    ):
        parser.error(
            "request-interval must be nonnegative and finite; max-consecutive-errors must be nonnegative"
        )
    if args.cookies_file:
        args.cookies_file = args.cookies_file.expanduser().resolve()
        if not args.cookies_file.is_file():
            parser.error("cookies-file must be an existing Netscape cookies file")
    try:
        import gdown
    except ImportError:
        parser.exit(
            1, "Install download dependencies: python -m pip install gdown==6.4.1\n"
        )
    if (
        args.cookies_file
        and "cookies_file" not in inspect.signature(gdown.download).parameters
    ):
        parser.exit(
            1,
            "--cookies-file requires a newer gdown: python -m pip install --upgrade gdown==6.4.1\n",
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
        "use_cookies": args.use_cookies or args.cookies_file is not None,
        "speed": args.speed_mbps * 1_000_000 / 8 if args.speed_mbps else None,
    }
    if args.cookies_file:
        options["cookies_file"] = str(args.cookies_file)
    try:
        entries = gdown.download_folder(**options, skip_download=True, quiet=True)
        available_files = len(entries)
        entries = (
            [entry for entry in entries if matches_patterns(entry.path, args.include)]
            if args.include
            else entries
        )
        if args.list_only:
            for entry in entries:
                print(entry.local_path)
            print(f"{len(entries)} files; no dataset files downloaded.")
            return 0
        if not entries:
            print(
                "No files match the selected paths; check --include patterns.",
                file=sys.stderr,
            )
            return 1
        output.mkdir(parents=True, exist_ok=True)
        receipt = {
            "source": url,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "listed_files": len(entries),
            "available_files": available_files,
            "include": args.include,
            "files": [],
            "failures": [],
            "extracted_directories": [],
            "skipped_existing": 0,
            "request_interval": args.request_interval,
            "max_consecutive_errors": args.max_consecutive_errors,
        }
        pacer = DownloadPacer(args.request_interval)
        consecutive_errors = 0
        try:
            for index, entry in enumerate(entries, start=1):
                browser_url = f"https://drive.google.com/file/d/{entry.id}/view"
                existing = Path(entry.local_path).is_file()
                print(
                    f"[{index}/{len(entries)}] {'Already downloaded' if existing else 'Downloading'}: {entry.path}"
                    + ("" if existing else f"\n  {browser_url}"),
                    flush=True,
                )
                try:
                    file = download_entry(
                        gdown,
                        entry,
                        use_cookies=options["use_cookies"],
                        speed=options["speed"],
                        retries=args.retries,
                        cookies_file=options.get("cookies_file"),
                        pacer=pacer,
                    )
                    receipt["files"].append(file)
                    if existing:
                        receipt["skipped_existing"] += 1
                    else:
                        consecutive_errors = 0
                    if args.extract_zip and Path(file).suffix.lower() == ".zip":
                        receipt["extracted_directories"].append(
                            str(extract_zip(Path(file)))
                        )
                except (
                    gdown.DownloadError,
                    ValueError,
                    OSError,
                    zipfile.BadZipFile,
                ) as exc:
                    receipt["failures"].append(
                        {
                            "id": entry.id,
                            "path": entry.path,
                            "url": browser_url,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"Failed: {entry.path}\n  Open in browser: {browser_url}\n  {exc}",
                        file=sys.stderr,
                    )
                    if not existing:
                        consecutive_errors += 1
                    if (
                        args.max_consecutive_errors
                        and consecutive_errors >= args.max_consecutive_errors
                    ):
                        receipt["stopped_reason"] = (
                            f"{consecutive_errors} consecutive downloads failed"
                        )
                        print(
                            f"Stopping after {consecutive_errors} consecutive download failures. "
                            "Check one failed file in your browser before retrying; continuing requests "
                            "does not clear a permission or quota block.",
                            file=sys.stderr,
                        )
                        break
                    if not args.continue_on_error:
                        break
            receipt["status"] = "partial" if receipt["failures"] else "complete"
        except KeyboardInterrupt:
            receipt["status"] = "interrupted"
            print(
                "Download interrupted; rerun the same command to resume.",
                file=sys.stderr,
            )
            return 130
        finally:
            write_receipt(output, receipt)
        print(
            f"Downloaded/resumed {len(receipt['files'])} of {len(entries)} files into {output}"
        )
        if receipt["failures"]:
            print(
                f"{len(receipt['failures'])} failed file(s); see {output / 'download_receipt.json'}.\n"
                "Open the failed file's browser link and try downloading it. If that also fails, "
                "check its sharing/download permissions or wait for Drive's quota to recover.\n"
                "If it works only when signed in and you already configured gdown's cookie cache, "
                "retry with --use-cookies or --cookies-file. --continue-on-error attempts other files "
                "until the consecutive-error limit; it does not bypass Drive restrictions.",
                file=sys.stderr,
            )
            return 1
        return 0
    except (gdown.DownloadError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(
            f"Download failed: {exc}\nCheck public sharing permissions and Drive quota; rerun to resume.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
