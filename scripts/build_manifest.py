"""Build a reviewable manifest whose labels are derived from dataset annotations.

The raw datasets ship box/keypoint annotations (COCO JSON, YOLO txt) but no
image-level class labels. This module turns those into per-frame ground truth for
the frame-level benchmark:

  * smoking-video COCO   -> "smoking" if a Smoking box is present, else "normal"
                            when only Not-Smoking boxes are present (clean negatives).
  * littering YOLO       -> "littering" when an action box exists, "normal" otherwise.
  * rough sleeping YOLO  -> "rough_sleeping" for a laying pose, "normal" for standing.

Every row is written with split="test": the five models under comparison score by
text prompt or generation and need no reference images, so no reference split (and
therefore no reference/test leakage) is required. group_id keeps frames from the
same source video/scene together so a future few-shot baseline could still be run
without leaking.

Run standalone to write data/manifest.csv:

    python scripts/build_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data import PROJECT_ROOT

DATASETS = PROJECT_ROOT / "data" / "raw" / "datasets"


def _smoking_group(file_name: str) -> str:
    """Group frames by camera + capture date so same-scene frames share a group."""
    match = re.match(r"frame_(c\d+)_(\d+)_([\d-]+)", file_name)
    if match:
        return f"smoking_{match.group(1)}_{match.group(3)}"
    return f"smoking_{file_name}"


def smoking_samples(root: Path = DATASETS) -> list[tuple[str, str, str]]:
    """Yield (relative_path, label, group_id) for the smoking-video COCO set."""
    base = root / "smoking" / "smoking-video.v1i.coco"
    rows: list[tuple[str, str, str]] = []
    for split in ("train", "valid"):
        ann_path = base / split / "_annotations.coco.json"
        if not ann_path.is_file():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        cats_by_image: dict[int, set[int]] = {}
        for annotation in data["annotations"]:
            cats_by_image.setdefault(annotation["image_id"], set()).add(
                annotation["category_id"]
            )
        for image in data["images"]:
            file_name = image.get("file_name") or ""
            if not (base / split / file_name).is_file():
                continue
            cats = cats_by_image.get(image["id"], set())
            # Category 2 is "Smoking"; category 1 is an explicit "Not Smoking".
            if 2 in cats:
                label = "smoking"
            elif 1 in cats:
                label = "normal"
            else:
                continue  # only person/super-category boxes; no clean signal
            rows.append(
                (
                    f"raw/datasets/smoking/smoking-video.v1i.coco/{split}/{file_name}",
                    label,
                    _smoking_group(file_name),
                )
            )
    return rows


def littering_samples(root: Path = DATASETS) -> list[tuple[str, str, str]]:
    """Yield (relative_path, label, group_id) for the littering YOLO set."""
    base = root / "littering" / "Littering"
    rows: list[tuple[str, str, str]] = []
    # The valid split has labels but no images; use train + test.
    for split in ("train", "test"):
        image_dir = base / split / "images"
        label_dir = base / split / "labels"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.glob("*.jpg")):
            stem = image.name[: -len(".jpg")]
            label_file = label_dir / f"{stem}.txt"
            has_action = False
            if label_file.is_file():
                has_action = any(
                    line.strip() for line in label_file.read_text().splitlines()
                )
            # Source video id is the leading "video_NN" token of the stem.
            match = re.match(r"(video_\d+)", stem)
            group_id = f"littering_{match.group(1)}" if match else f"littering_{stem}"
            rows.append(
                (
                    f"raw/datasets/littering/Littering/{split}/images/{image.name}",
                    "littering" if has_action else "normal",
                    group_id,
                )
            )
    return rows


def rough_sleeping_samples(root: Path = DATASETS) -> list[tuple[str, str, str]]:
    """Yield (relative_path, label, group_id) for the rough-sleeping pose set.

    Uses the kaggle copy (byte-identical to the hugging_face one). Class 0 is a
    laying pose (rough sleeping); class 1 is standing (normal).
    """
    base = root / "rough sleeping" / "kaggle" / "laying_dataset"
    image_dir, label_dir = base / "images", base / "labels"
    rows: list[tuple[str, str, str]] = []
    if not image_dir.is_dir():
        return rows
    for image in sorted(image_dir.glob("*.png")):
        stem = image.name[: -len(".png")]
        label_file = label_dir / f"{stem}.txt"
        classes: set[str] = set()
        if label_file.is_file():
            for line in label_file.read_text().splitlines():
                if line.strip():
                    classes.add(line.split()[0])
        if "0" in classes:
            label = "rough_sleeping"
        elif "1" in classes:
            label = "normal"
        else:
            continue
        rows.append(
            (f"raw/datasets/rough sleeping/kaggle/laying_dataset/images/{image.name}",
             label, f"roughsleep_{stem}")
        )
    return rows


def build_master_manifest(
    data_dir: Path = PROJECT_ROOT / "data",
    out_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Write the master manifest and return its (path, label, group_id) rows."""
    if out_path is None:
        out_path = data_dir / "manifest.csv"
    rows = smoking_samples() + littering_samples() + rough_sleeping_samples()
    # De-duplicate by resolved path in case a frame appears under two sources.
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for rel_path, label, group_id in rows:
        key = (data_dir / rel_path).resolve()
        if key in seen or not key.is_file():
            continue
        seen.add(key)
        unique.append((rel_path, label, group_id))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "split", "group_id", "suggested_label"])
        for rel_path, label, group_id in unique:
            writer.writerow([rel_path, label, "test", group_id, label])
    return unique


def stratified_subset(
    rows: list[tuple[str, str, str]],
    per_class: dict[str, int],
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Pick up to `per_class[label]` frames of each label, deterministically.

    Labels absent from `per_class` are excluded. This is how the notebook builds a
    balanced head-to-head set that every model sees identically.
    """
    by_label: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)
    rng = random.Random(seed)
    selected: list[tuple[str, str, str]] = []
    for label, count in per_class.items():
        pool = by_label.get(label, [])
        rng.shuffle(pool)
        selected.extend(pool[:count])
    return selected


def summarize(rows: list[tuple[str, str, str]]) -> dict[str, int]:
    """Count rows per (label) for a quick sanity check."""
    return dict(Counter(row[1] for row in rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="Output manifest (default data/manifest.csv)"
    )
    args = parser.parse_args(argv)
    rows = build_master_manifest(args.data_dir, args.manifest)
    out = args.manifest or (args.data_dir / "manifest.csv")
    print(f"Wrote {len(rows)} labeled frames to {out}")
    for label, count in sorted(summarize(rows).items()):
        print(f"  {label:16} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
