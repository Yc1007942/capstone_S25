"""Derive image-level anomaly labels from YOLO/COCO annotation files.

The benchmark deliberately does not import detection boxes; this module is the
single place that turns per-box annotation files into *image-level* labels so a
balanced multi-anomaly manifest can be built for head-to-head model evaluation.

Every function here is pure (standard library only, no torch / network) and takes
its dataset root as an argument so it can be pointed at a small fixture tree in
tests without touching the real data on disk.

Label strategy per class (see notebooks/anomaly_benchmark.ipynb for caveats):

* ``smoking``        -- >=1 "Smoking" COCO box in smoking-video.v1i.coco.
* ``normal``         -- a smoking-video frame with a "Not Smoking" box and no
                        "Smoking" box (a clean, person-present normal).
* ``littering``      -- an image whose matching YOLO label file is non-empty.
* ``rough_sleeping`` -- a kaggle laying_dataset image labelled only as class 0
                        ("laying"); scenes that also contain a "standing" person
                        are excluded to keep the positive set unambiguous.
* ``loitering``      -- every frame of the ATMGUARD-Loitering-Tracking corpus,
                        treated as a dataset-level proxy (the COCO file carries
                        only "person" boxes; loitering is inherently temporal).

``unattended_items`` has no labels in the repo and is intentionally not produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data import IMAGE_EXTENSIONS, PROJECT_ROOT  # noqa: E402

DATASETS = PROJECT_ROOT / "data" / "raw" / "datasets"

# Classes that can be labelled from the available datasets (order is stable and
# used as the manifest row order). ``normal`` is the clean-negative class.
MANIFEST_CLASSES = ("smoking", "littering", "rough_sleeping", "loitering", "normal")


@dataclass(frozen=True)
class LabeledImage:
    """One image with a derived label and a diversity group id."""

    path: Path
    label: str
    group_id: str


# --------------------------------------------------------------------------- #
# Annotation parsing helpers (pure, testable).
# --------------------------------------------------------------------------- #
def load_coco(coco_path: Path) -> tuple[dict[int, str], dict[int, str], list[dict]]:
    """Parse a COCO ``_annotations.coco.json`` into its three core tables.

    Returns ``(categories, images_by_id, annotations)`` where ``images_by_id``
    maps image id -> relative file name as stored in the file.
    """
    data = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    categories = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    images = {i["id"]: i.get("file_name", "") for i in data.get("images", [])}
    return categories, images, list(data.get("annotations", []))


def coco_positive_image_ids(
    categories: dict[int, str], annotations: list[dict], positive_names: set[str]
) -> set[int]:
    """Image ids that carry at least one box whose category name is in the set."""
    wanted = {cid for cid, name in categories.items() if name in positive_names}
    return {a["image_id"] for a in annotations if a.get("category_id") in wanted}


def yolo_has_box(label_path: Path) -> bool:
    """True when a YOLO label file exists and contains at least one box line."""
    if not label_path.is_file():
        return False
    return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())


def yolo_class_tokens(label_path: Path) -> set[str]:
    """The set of class-id tokens present in a YOLO label file (empty if absent)."""
    if not label_path.is_file():
        return set()
    return {
        line.split()[0]
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split()
    }


# --------------------------------------------------------------------------- #
# Diversity group keys (derived from file names only, so they are testable).
# --------------------------------------------------------------------------- #
_SMOKE_FRAME_RE = re.compile(r"frame_(c\d+)_(\d+)_([\d]{4}-[\d]{2}-[\d]{2})")
_LITTER_VIDEO_RE = re.compile(r"video_(\d+)")


def smoking_video_group(path: Path) -> str:
    """Group a smoking-video frame by (camera, date): one recording session."""
    match = _SMOKE_FRAME_RE.search(Path(path).name)
    if not match:
        return f"smoke:{Path(path).name}"
    camera, _, date = match.groups()
    return f"smoke:{camera}:{date}"


def littering_group(path: Path) -> str:
    """Group a littering frame by its source video id (``video_<n>`` prefix)."""
    match = _LITTER_VIDEO_RE.search(Path(path).name)
    if not match:
        return f"litter:{Path(path).name}"
    return f"litter:video_{match.group(1)}"


# --------------------------------------------------------------------------- #
# Per-category collectors. Each returns LabeledImage lists rooted at ``root``.
# --------------------------------------------------------------------------- #
def collect_smoking(root: Path = DATASETS) -> tuple[list[LabeledImage], list[LabeledImage]]:
    """Return ``(smoking_positives, normals)`` from the smoking-video COCO set.

    A frame is a *smoking* positive when it has any "Smoking" box (a mixed
    Smoking + Not-Smoking frame counts as smoking). Otherwise, if it has a
    "Not Smoking" box, it is a clean *normal*.
    """
    base = root / "smoking" / "smoking-video.v1i.coco"
    positives: list[LabeledImage] = []
    normals: list[LabeledImage] = []
    for split in ("train", "valid"):
        coco_path = base / split / "_annotations.coco.json"
        if not coco_path.is_file():
            continue
        categories, images, annotations = load_coco(coco_path)
        smoking_ids = coco_positive_image_ids(categories, annotations, {"Smoking"})
        not_smoking_ids = coco_positive_image_ids(categories, annotations, {"Not Smoking"})
        for image_id, file_name in images.items():
            path = (coco_path.parent / file_name).resolve()
            if not path.is_file():
                continue
            group = smoking_video_group(path)
            if image_id in smoking_ids:
                positives.append(LabeledImage(path, "smoking", group))
            elif image_id in not_smoking_ids:
                normals.append(LabeledImage(path, "normal", group))
    return positives, normals


def collect_littering(root: Path = DATASETS) -> list[LabeledImage]:
    """Littering positives: images with a non-empty matching YOLO label file.

    Only the ``train`` and ``test`` splits are used; the ``valid`` split ships
    labels but no images in this dataset.
    """
    base = root / "littering" / "Littering"
    positives: list[LabeledImage] = []
    for split in ("train", "test"):
        image_dir, label_dir = base / split / "images", base / split / "labels"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if yolo_has_box(label_dir / (image.stem + ".txt")):
                positives.append(
                    LabeledImage(image.resolve(), "littering", littering_group(image))
                )
    return positives


def collect_rough_sleeping(root: Path = DATASETS) -> list[LabeledImage]:
    """Rough-sleeping positives: kaggle laying_dataset images labelled only as
    class 0 ("laying"). Images that also contain a "standing" person are skipped
    so the positive set is unambiguous."""
    base = root / "rough sleeping" / "kaggle" / "laying_dataset"
    image_dir, label_dir = base / "images", base / "labels"
    positives: list[LabeledImage] = []
    if not image_dir.is_dir():
        return positives
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        tokens = yolo_class_tokens(label_dir / (image.stem + ".txt"))
        if tokens == {"0"}:
            positives.append(
                LabeledImage(image.resolve(), "rough_sleeping", f"rough:{image.stem}")
            )
    return positives


def collect_loitering(root: Path = DATASETS) -> list[LabeledImage]:
    """Loitering positives: every frame of the ATMGUARD-Loitering-Tracking corpus.

    This is a dataset-level proxy, not a box-derived label -- the COCO file only
    carries "person" boxes and loitering is a temporal behaviour that cannot be
    confirmed from a single frame. The caveat is surfaced in the notebook.
    """
    base = root / "loitering" / "Roboflow Dataset (New Workspace)"
    positives: list[LabeledImage] = []
    for split in ("train", "valid", "test"):
        split_dir = base / split
        if not split_dir.is_dir():
            continue
        for image in sorted(split_dir.iterdir()):
            if image.suffix.lower() in IMAGE_EXTENSIONS:
                positives.append(
                    LabeledImage(image.resolve(), "loitering", f"loit:{image.stem}")
                )
    return positives


# --------------------------------------------------------------------------- #
# Diversity-aware sampling.
# --------------------------------------------------------------------------- #
def diverse_sample(
    items: list[LabeledImage], k: int, max_per_group: int, rng: random.Random
) -> list[LabeledImage]:
    """Sample up to ``k`` items taking at most ``max_per_group`` per group_id.

    Groups are visited round-robin so the sample spreads across as many distinct
    sources (camera-sessions / videos) as possible instead of clustering on one.
    If capping prevents reaching ``k`` (too few groups), the cap is relaxed and
    the remainder filled so we still return ``min(k, len(items))`` items. The
    result is deterministic for a given ``rng`` seed.
    """
    if not items or k <= 0:
        return []
    by_group: dict[str, list[LabeledImage]] = {}
    for item in items:
        by_group.setdefault(item.group_id, []).append(item)

    groups = list(by_group)
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(by_group[group])

    chosen: list[LabeledImage] = []
    taken = {g: 0 for g in groups}
    while len(chosen) < k:
        progressed = False
        for group in groups:
            if len(chosen) >= k:
                break
            if taken[group] < max_per_group and taken[group] < len(by_group[group]):
                chosen.append(by_group[group][taken[group]])
                taken[group] += 1
                progressed = True
        if not progressed:
            break

    if len(chosen) < k:  # relax the cap to fill the remaining slots
        picked = {id(item) for item in chosen}
        remainder = [it for g in groups for it in by_group[g] if id(it) not in picked]
        rng.shuffle(remainder)
        chosen.extend(remainder[: k - len(chosen)])

    return chosen[:k]


# --------------------------------------------------------------------------- #
# Manifest assembly.
# --------------------------------------------------------------------------- #
def build_manifest(
    out_path: Path, per_class: int = 16, seed: int = 42, root: Path = DATASETS
) -> dict:
    """Build a balanced multi-anomaly manifest CSV and return a summary dict.

    Every row is ``split="test"`` (the four runnable models are reference-free,
    so no per-class reference images are needed). A meaningful ``group_id`` is
    written per source so the leakage check in :func:`scripts.data.read_manifest`
    stays meaningful if references are added later.
    """
    rng = random.Random(seed)

    smoking_positives, normals = collect_smoking(root)
    pools: dict[str, list[LabeledImage]] = {
        "smoking": diverse_sample(smoking_positives, per_class, 4, rng),
        "littering": diverse_sample(collect_littering(root), per_class, 4, rng),
        "rough_sleeping": diverse_sample(
            collect_rough_sleeping(root), per_class, per_class, rng
        ),
        "loitering": diverse_sample(collect_loitering(root), per_class, per_class, rng),
        "normal": diverse_sample(normals, per_class, 4, rng),
    }

    rows: list[dict] = []
    seen_paths: set[Path] = set()
    for label in MANIFEST_CLASSES:
        for item in pools[label]:
            if item.path in seen_paths:
                continue
            seen_paths.add(item.path)
            rows.append(
                {
                    "path": str(item.path),
                    "label": label,
                    "split": "test",
                    "group_id": item.group_id,
                }
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "group_id"])
        writer.writeheader()
        writer.writerows(rows)

    per_class_counts = {label: 0 for label in MANIFEST_CLASSES}
    for row in rows:
        per_class_counts[row["label"]] += 1
    return {"rows": len(rows), "per_class": per_class_counts, "out_path": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "data" / "anomaly_manifest.csv"
    )
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--root", type=Path, default=DATASETS, help="Datasets root (for tests/fixtures)"
    )
    args = parser.parse_args(argv)

    summary = build_manifest(args.out, per_class=args.per_class, seed=args.seed, root=args.root)
    print(f"Wrote {summary['rows']} rows to {summary['out_path']}")
    for label in MANIFEST_CLASSES:
        print(f"  {label:<14} {summary['per_class'][label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
