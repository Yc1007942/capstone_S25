"""Shared manifest and anomaly-definition validation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEFINITIONS = Path(__file__).with_name("anomalies.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".m4v"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str | None
    split: str
    group_id: str


def read_definitions(path: Path) -> dict[str, str]:
    definitions = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(definitions, dict)
        or "normal" not in definitions
        or len(definitions) < 2
    ):
        raise ValueError(
            "Definitions must map 'normal' and at least one anomaly to descriptions"
        )
    if any(
        not isinstance(key, str)
        or not key.strip()
        or key != key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in definitions.items()
    ):
        raise ValueError("Definition names and descriptions must be nonempty strings")
    return definitions


def read_manifest(path: Path, definitions: dict[str, str]) -> list[Sample]:
    samples = []
    seen_paths: set[Path] = set()
    groups: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "split"} <= set(reader.fieldnames or []):
            raise ValueError(
                "Manifest needs path,label,split columns (group_id is recommended)"
            )
        for line, row in enumerate(reader, start=2):
            if None in row or any(
                row[key] is None for key in ("path", "label", "split")
            ):
                raise ValueError(f"Manifest line {line}: malformed CSV row")
            if not row["path"].strip():
                raise ValueError(f"Manifest line {line}: empty image path")
            image_path = (path.parent / row["path"].strip()).resolve()
            label = row["label"].strip() or None
            split = row["split"].strip() or "test"
            group_id = (row.get("group_id") or "").strip() or str(image_path)
            if split not in {"test", "reference", "ignore"}:
                raise ValueError(
                    f"Manifest line {line}: split must be test, reference, or ignore"
                )
            if label is not None and label not in definitions:
                raise ValueError(f"Manifest line {line}: unknown label {label!r}")
            if split == "ignore":
                continue
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                raise ValueError(
                    f"Manifest line {line}: missing or unsupported image {image_path}"
                )
            if image_path in seen_paths:
                raise ValueError(f"Manifest line {line}: duplicate image {image_path}")
            if split == "reference" and label is None:
                raise ValueError(
                    f"Manifest line {line}: reference images need verified labels"
                )
            seen_paths.add(image_path)
            groups.setdefault(group_id, set()).add(split)
            samples.append(Sample(image_path, label, split, group_id))
    leaking = [
        group for group, splits in groups.items() if {"test", "reference"} <= splits
    ]
    if leaking:
        raise ValueError(
            f"Reference/test leakage: source groups appear in both splits: {leaking[:3]}"
        )
    if not any(sample.split == "test" for sample in samples):
        raise ValueError("Manifest has no test images")
    return samples


def select_test_samples(
    samples: list[Sample], limit: int | None, seed: int
) -> list[Sample]:
    selected = [sample for sample in samples if sample.split == "test"]
    random.Random(seed).shuffle(selected)
    return selected[:limit] if limit is not None else selected


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
