"""Separate, video-level pipeline for VadCLIP (temporal video anomaly detection).

VadCLIP is deliberately kept OUT of the frame-level CPU harness in ``models.py``:
it is a trained temporal detector whose published checkpoints expect their original
feature pipeline and label setup (UCF-Crime / XD-Violence), and it scores *sequences*
of frames, not single stills. This script is that separate pipeline.

It does three things:

1. Detect whether VadCLIP is runnable in this environment (a local checkout plus its
   pretrained weights). The checkout path comes from ``--repo`` or ``$VADCLIP_REPO``.
2. If it is, sample frames from the project's videos and run inference, writing
   per-frame anomaly scores to a JSON report.
3. If it is not (the expected case on this CPU box), write a clear ``not_configured``
   status with exact setup steps instead of fabricating a score.

The benchmark notebook calls this and folds the resulting status into the comparison
table, so VadCLIP appears alongside the frame-level models without pretending to be one.

    python scripts/vadclip.py --repo /path/to/VadCLIP \
        --video-dir data/raw/datasets/unattended\\ items --max-videos 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data import PROJECT_ROOT

# VadCLIP's public repository and the checkpoints it ships for.
VADCLIP_REPO_URL = "https://github.com/nwpu-zxr/VadCLIP"
EXPECTED_WEIGHTS = ("ucfcrime", "xdviolence")


def find_weights(repo: Path) -> list[str]:
    """Return names of any VadCLIP checkpoints present under the repo."""
    found = []
    for pattern in ("*.pth", "*.pt"):
        for path in repo.rglob(pattern):
            stem = path.stem.lower()
            if any(tag in stem for tag in EXPECTED_WEIGHTS) or "pretrain" not in stem:
                found.append(str(path.relative_to(repo)))
    return sorted(set(found))


def sample_frames(video: Path, out_dir: Path, fps: float, max_frames: int) -> list[Path]:
    """Extract a bounded set of frames with ffmpeg (mirrors scripts/init.py)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for VadCLIP frame sampling")
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = "scale=w=min(512\\,iw):h=min(512\\,ih):force_original_aspect_ratio=decrease"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-i", str(video), "-an", "-sn", "-vf", f"fps={fps},{scale}",
        "-frames:v", str(max_frames), str(out_dir / "frame_%05d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed for {video}: {result.stderr.strip()}")
    return sorted(out_dir.glob("frame_*.jpg"))


def run_vadclip(repo: Path, video_paths: list[Path], workdir: Path) -> dict:
    """Run VadCLIP inference on sampled frames.

    This is the integration point against a local VadCLIP checkout. It imports the
    repo's own model/feature code so we use its real pipeline rather than an
    approximation; if the expected entry points are missing it raises, which the
    caller reports as a configuration problem rather than a fake score.
    """
    sys.path.insert(0, str(repo))
    # VadCLIP exposes its detector through its package root; import defensively so a
    # layout change fails loudly instead of silently scoring with the wrong model.
    try:
        from main import get_model  # type: ignore  # noqa: N813 -- repo's own API
    except Exception as exc:  # pragma: no cover - depends on external checkout
        raise RuntimeError(
            f"VadCLIP checkout at {repo} does not expose the expected inference API "
            f"(import failed: {exc}). Confirm you cloned {VADCLIP_REPO_URL} and that its "
            "pretrained UCF-Crime/XD-Violence weights are present."
        ) from exc

    model = get_model(str(repo))  # loads CLIP backbone + temporal head + checkpoint
    per_video: dict[str, list[float]] = {}
    for video in video_paths:
        frames = sample_frames(video, workdir / video.stem, fps=1.0, max_frames=32)
        scores = [float(model.score([str(f) for f in frames])) for _ in (frames and [None])]
        per_video[video.name] = scores
    return {"per_video": per_video}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=None,
        help="Local VadCLIP checkout (default: $VADCLIP_REPO)",
    )
    parser.add_argument(
        "--video-dir", type=Path,
        default=PROJECT_ROOT / "data/raw/datasets/unattended items",
        help="Directory of source videos to score",
    )
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Report path (default: results/vadclip/status.json)",
    )
    args = parser.parse_args(argv)

    output = args.output or (PROJECT_ROOT / "results" / "vadclip" / "status.json")
    report: dict = {
        "model": "VadCLIP",
        "kind": "temporal video anomaly detection",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    repo = args.repo or (Path(os.environ["VADCLIP_REPO"]) if os.environ.get("VADCLIP_REPO") else None)
    if repo is None:
        report.update(
            status="not_configured",
            reason="No VadCLIP checkout provided.",
            setup=[
                f"git clone {VADCLIP_REPO_URL}",
                "Download its pretrained UCF-Crime / XD-Violence checkpoints into the repo.",
                "Re-run: python scripts/vadclip.py --repo <path-to-clone> --video-dir <videos>",
            ],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"VadCLIP not configured; wrote {output}")
        return 2

    if not repo.is_dir():
        report.update(status="not_configured", reason=f"Repo path does not exist: {repo}", setup=[])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"VadCLIP repo missing at {repo}; wrote {output}")
        return 2

    weights = find_weights(repo)
    videos = sorted(
        p for p in args.video_dir.glob("*") if p.suffix.lower() in {".mp4", ".avi", ".mpg"}
    )[: args.max_videos]
    report["repo"] = str(repo)
    report["weights_found"] = weights
    report["videos"] = [v.name for v in videos]

    try:
        workdir = output.parent / "frames"
        result = run_vadclip(repo, videos, workdir)
        report.update(status="ok", **result)
    except Exception as exc:  # noqa: BLE001 -- surface any integration failure as status
        report.update(
            status="error",
            reason=str(exc),
            note=(
                "VadCLIP needs its original feature pipeline + temporal head and a GPU for "
                "reasonable throughput; it is not part of the frame-level CPU comparison."
            ),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"VadCLIP status: {report['status']}; wrote {output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
