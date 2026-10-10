"""Unit tests for label derivation from YOLO/COCO annotation files."""

from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.derive_labels import (
    LabeledImage,
    build_manifest,
    collect_littering,
    collect_loitering,
    collect_rough_sleeping,
    collect_smoking,
    coco_positive_image_ids,
    diverse_sample,
    load_coco,
    littering_group,
    smoking_video_group,
    yolo_class_tokens,
    yolo_has_box,
)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class FixtureBase(unittest.TestCase):
    """Build a tiny dataset tree mirroring the real layout under a temp root."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._build_smoking()
        self._build_littering()
        self._build_rough_sleeping()
        self._build_loitering()

    def _build_smoking(self) -> None:
        split_dir = self.root / "smoking" / "smoking-video.v1i.coco" / "train"
        # a: normal, b/c/d: smoking (c is mixed), e/f: normal
        coco = {
            "categories": [
                {"id": 0, "name": "Not Smoking"},
                {"id": 1, "name": "Smoking"},
            ],
            "images": [
                {"id": i, "file_name": f"{letter}.jpg"}
                for i, letter in enumerate("abcdef", start=1)
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0},  # a -> normal
                {"id": 2, "image_id": 2, "category_id": 1},  # b -> smoking
                {"id": 3, "image_id": 3, "category_id": 0},  # c mixed...
                {"id": 4, "image_id": 3, "category_id": 1},  # ...-> smoking
                {"id": 5, "image_id": 4, "category_id": 1},  # d -> smoking
                {"id": 6, "image_id": 5, "category_id": 0},  # e -> normal
                {"id": 7, "image_id": 6, "category_id": 0},  # f -> normal
            ],
        }
        _write(split_dir / "_annotations.coco.json", json.dumps(coco))
        for letter in "abcdef":
            _write(split_dir / f"{letter}.jpg")

    def _build_littering(self) -> None:
        base = self.root / "littering" / "Littering"
        # train: 3 positives (non-empty labels), 1 empty, 1 missing label
        for name in ("v1_1", "v2_1", "v3_1"):
            _write(base / "train" / "images" / f"{name}.jpg")
            _write(base / "train" / "labels" / f"{name}.txt", "0 0.5 0.5 0.1 0.1\n")
        _write(base / "train" / "images" / "v4_empty.jpg")
        _write(base / "train" / "labels" / "v4_empty.txt", "")  # empty -> excluded
        _write(base / "train" / "images" / "v5_nolabel.jpg")  # no label -> excluded
        # test: 1 positive
        _write(base / "test" / "images" / "t1_1.jpg")
        _write(base / "test" / "labels" / "t1_1.txt", "0 0.5 0.5 0.2 0.2\n")
        # valid: labels but no images -> must be ignored entirely
        _write(base / "valid" / "labels" / "ghost.txt", "0 0.5 0.5 0.1 0.1\n")

    def _build_rough_sleeping(self) -> None:
        base = self.root / "rough sleeping" / "kaggle" / "laying_dataset"
        # pure-laying (class 0 only) -> positive; mixed (0+1) and standing-only excluded
        for name, tokens in (
            ("lay_a", "0\n"),
            ("lay_b", "0\n0\n"),
            ("lay_e", "0 0.5 0.5 0.1 0.1\n"),
            ("mix_c", "0\n1\n"),
            ("stand_d", "1\n"),
        ):
            _write(base / "images" / f"{name}.jpg")
            _write(base / "labels" / f"{name}.txt", tokens)

    def _build_loitering(self) -> None:
        base = self.root / "loitering" / "Roboflow Dataset (New Workspace)"
        for split, count in (("train", 2), ("valid", 1), ("test", 1)):
            for i in range(count):
                _write(base / split / f"{split}_{i}.jpg")
            _write(base / split / "_annotations.coco.json", "{}")  # non-image, skipped


class CocoParsingTests(FixtureBase):
    def test_load_coco_returns_core_tables(self) -> None:
        coco_path = (
            self.root / "smoking" / "smoking-video.v1i.coco" / "train" / "_annotations.coco.json"
        )
        categories, images, annotations = load_coco(coco_path)
        self.assertEqual(categories, {0: "Not Smoking", 1: "Smoking"})
        self.assertEqual(len(images), 6)
        self.assertEqual(annotations[0]["image_id"], 1)

    def test_positive_image_ids_by_category_name(self) -> None:
        coco_path = (
            self.root / "smoking" / "smoking-video.v1i.coco" / "train" / "_annotations.coco.json"
        )
        categories, _, annotations = load_coco(coco_path)
        smoking_ids = coco_positive_image_ids(categories, annotations, {"Smoking"})
        not_smoking_ids = coco_positive_image_ids(categories, annotations, {"Not Smoking"})
        # image ids 2 (b), 3 (c mixed), 4 (d) carry a Smoking box
        self.assertEqual(smoking_ids, {2, 3, 4})
        # image ids 1 (a), 3 (c mixed), 5 (e), 6 (f) carry a Not Smoking box
        self.assertEqual(not_smoking_ids, {1, 3, 5, 6})


class YoloHelperTests(FixtureBase):
    def test_has_box_distinguishes_empty_missing_and_present(self) -> None:
        labels = self.root / "littering" / "Littering" / "train" / "labels"
        self.assertTrue(yolo_has_box(labels / "v1_1.txt"))
        self.assertFalse(yolo_has_box(labels / "v4_empty.txt"))
        self.assertFalse(yolo_has_box(labels / "does_not_exist.txt"))

    def test_class_tokens(self) -> None:
        labels = self.root / "rough sleeping" / "kaggle" / "laying_dataset" / "labels"
        self.assertEqual(yolo_class_tokens(labels / "lay_a.txt"), {"0"})
        self.assertEqual(yolo_class_tokens(labels / "mix_c.txt"), {"0", "1"})
        self.assertEqual(yolo_class_tokens(labels / "missing.txt"), set())


class GroupKeyTests(unittest.TestCase):
    def test_smoking_video_group_parses_camera_and_date(self) -> None:
        name = "frame_c2_002888_2025-10-29_10-00-27-059_AM_jpg.rf.abc.jpg"
        self.assertEqual(smoking_video_group(Path(name)), "smoke:c2:2025-10-29")

    def test_smoking_video_group_falls_back_to_name(self) -> None:
        name = "something_else.jpg"
        self.assertEqual(smoking_video_group(Path(name)), f"smoke:{name}")

    def test_littering_group_parses_video_id(self) -> None:
        name = "video_10_29_jpg.rf.f294141f80f8bf41e96006742fb9ef29.jpg"
        self.assertEqual(littering_group(Path(name)), "litter:video_10")


class CollectorTests(FixtureBase):
    def test_collect_smoking_splits_positives_and_normals(self) -> None:
        positives, normals = collect_smoking(self.root)
        pos_names = sorted(p.path.name for p in positives)
        norm_names = sorted(n.path.name for n in normals)
        self.assertEqual(pos_names, ["b.jpg", "c.jpg", "d.jpg"])  # c mixed -> positive
        self.assertEqual(norm_names, ["a.jpg", "e.jpg", "f.jpg"])

    def test_collect_littering_uses_only_nonempty_labels_and_valid_splits(self) -> None:
        positives = collect_littering(self.root)
        names = sorted(p.path.name for p in positives)
        self.assertEqual(names, ["t1_1.jpg", "v1_1.jpg", "v2_1.jpg", "v3_1.jpg"])

    def test_collect_rough_sleeping_keeps_only_pure_laying(self) -> None:
        positives = collect_rough_sleeping(self.root)
        names = sorted(p.path.name for p in positives)
        self.assertEqual(names, ["lay_a.jpg", "lay_b.jpg", "lay_e.jpg"])

    def test_collect_loitering_includes_all_frames_across_splits(self) -> None:
        positives = collect_loitering(self.root)
        names = sorted(p.path.name for p in positives)
        self.assertEqual(
            names, ["test_0.jpg", "train_0.jpg", "train_1.jpg", "valid_0.jpg"]
        )


class DiverseSampleTests(unittest.TestCase):
    def _items(self, groups: list[str]) -> list[LabeledImage]:
        return [
            LabeledImage(Path(f"/x/{g}_{i}.jpg"), "smoking", g)
            for i, g in enumerate(groups)
        ]

    def test_respects_max_per_group_cap(self) -> None:
        items = self._items(["a"] * 10)  # one group only
        chosen = diverse_sample(items, k=5, max_per_group=2, rng=random.Random(0))
        self.assertEqual(len(chosen), 5)  # cap relaxed to still reach k

    def test_spreads_across_groups_round_robin(self) -> None:
        items = self._items(["a", "b", "c"] * 4)  # 12 items, 3 groups of 4
        chosen = diverse_sample(items, k=6, max_per_group=2, rng=random.Random(0))
        per_group: dict[str, int] = {}
        for item in chosen:
            per_group[item.group_id] = per_group.get(item.group_id, 0) + 1
        self.assertEqual(len(chosen), 6)
        self.assertTrue(all(count <= 2 for count in per_group.values()))
        self.assertEqual(set(per_group), {"a", "b", "c"})  # spread across all groups

    def test_returns_min_k_len(self) -> None:
        items = self._items(["a", "b"])
        self.assertEqual(len(diverse_sample(items, k=10, max_per_group=5, rng=random.Random(0))), 2)
        self.assertEqual(len(diverse_sample([], k=3, max_per_group=1, rng=random.Random(0))), 0)

    def test_deterministic_for_seed(self) -> None:
        items = self._items([f"g{i % 5}" for i in range(40)])
        first = [i.path.name for i in diverse_sample(items, k=12, max_per_group=3, rng=random.Random(7))]
        second = [i.path.name for i in diverse_sample(items, k=12, max_per_group=3, rng=random.Random(7))]
        self.assertEqual(first, second)


class BuildManifestTests(FixtureBase):
    def test_writes_balanced_test_manifest(self) -> None:
        out_path = self.root / "out" / "manifest.csv"
        summary = build_manifest(out_path, per_class=3, seed=42, root=self.root)

        # every class that has >=3 candidates yields exactly 3 rows
        self.assertEqual(summary["rows"], 15)
        for label in ("smoking", "littering", "rough_sleeping", "loitering", "normal"):
            self.assertEqual(summary["per_class"][label], 3, label)

        with out_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 15)
        self.assertEqual(set(rows[0]), {"path", "label", "split", "group_id"})
        self.assertTrue(all(row["split"] == "test" for row in rows))

        # no duplicate image paths, and every referenced file exists on disk
        paths = [Path(row["path"]) for row in rows]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(p.is_file() for p in paths))


if __name__ == "__main__":
    unittest.main()
