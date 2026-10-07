"""Behavioral checks for throttling, data preparation, evaluation, and isolation."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import benchmark, fetch_data
from scripts.data import Sample, read_manifest, select_test_samples
from scripts.init import main as prepare_main
from scripts.init import sample_video
from scripts.metrics import detection_metrics, roc_auc
from scripts.models import Prediction, parse_vlm_response
from scripts.throttle import CPUSettings, Throttler

DEFINITIONS = {"normal": "Ordinary activity", "smoking": "A person smoking"}


class ThrottleTests(unittest.TestCase):
    def test_paces_active_time(self):
        throttle = Throttler(CPUSettings(duty_cycle=0.5))
        with (
            patch("scripts.throttle.time.perf_counter", side_effect=[10.0, 12.0]),
            patch("scripts.throttle.time.sleep") as sleep,
            throttle.operation(),
        ):
            pass
        sleep.assert_called_once_with(2.0)
        self.assertEqual(throttle.total_sleep_seconds, 2.0)

    def test_rate_cap_and_full_duty(self):
        throttle = Throttler(CPUSettings(duty_cycle=1.0, max_samples_per_second=0.2))
        with (
            patch("scripts.throttle.time.perf_counter", side_effect=[10.0, 12.0]),
            patch("scripts.throttle.time.sleep") as sleep,
            throttle.operation(),
        ):
            pass
        sleep.assert_called_once_with(3.0)

    def test_failure_does_not_delay_cancellation(self):
        with (
            patch("scripts.throttle.time.sleep") as sleep,
            self.assertRaises(KeyboardInterrupt),
            Throttler(CPUSettings()).operation(),
        ):
            raise KeyboardInterrupt
        sleep.assert_not_called()

    def test_invalid_settings(self):
        for options in (
            {"threads": 0},
            {"duty_cycle": 0},
            {"duty_cycle": 1.1},
            {"duty_cycle": float("nan")},
            {"max_samples_per_second": float("inf")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CPUSettings(**options)

    def test_environment_overrides_thread_pools(self):
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "99"}):
            CPUSettings(threads=2).configure_environment()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "one.jpg").touch()
        (self.root / "two.jpg").touch()

    def manifest(self, rows):
        path = self.root / "manifest.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "split", "group_id"])
            writer.writerows(rows)
        return path

    def test_unlabeled_timing_and_relative_paths(self):
        path = self.manifest([["one.jpg", "", "test", "one"]])
        samples = read_manifest(path, DEFINITIONS)
        self.assertEqual(samples[0].path, self.root / "one.jpg")
        self.assertIsNone(samples[0].label)

    def test_reference_test_group_leakage_rejected(self):
        path = self.manifest(
            [
                ["one.jpg", "normal", "test", "same-video"],
                ["two.jpg", "normal", "reference", "same-video"],
            ]
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_manifest(path, DEFINITIONS)

    def test_duplicate_images_rejected(self):
        path = self.manifest(
            [
                ["one.jpg", "normal", "test", "one"],
                ["./one.jpg", "normal", "test", "two"],
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_manifest(path, DEFINITIONS)

    def test_unknown_label_and_unlabeled_reference_rejected(self):
        for label, split in (("invented", "test"), ("", "reference")):
            path = self.manifest([["one.jpg", label, split, "one"]])
            with self.subTest(label=label, split=split), self.assertRaises(ValueError):
                read_manifest(path, DEFINITIONS)

    def test_malformed_csv_rejected(self):
        path = self.root / "manifest.csv"
        path.write_text("path,label,split\none.jpg\n")
        with self.assertRaisesRegex(ValueError, "malformed"):
            read_manifest(path, DEFINITIONS)

    def test_selection_is_repeatable_and_excludes_references(self):
        samples = [
            Sample(Path(str(index)), None, "test", str(index)) for index in range(20)
        ]
        samples.append(Sample(Path("ref"), "normal", "reference", "ref"))
        selected = select_test_samples(samples, 4, 42)
        self.assertEqual(selected, select_test_samples(samples, 4, 42))
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(sample.split == "test" for sample in selected))

    def test_manifest_keeps_folder_hint_separate_and_preserves_labels(self):
        image = self.root / "raw/smoking/image.jpg"
        image.parent.mkdir(parents=True)
        image.touch()
        manifest = self.root / "output/manifest.csv"
        arguments = ["--data-dir", str(self.root / "raw"), "--manifest", str(manifest)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(prepare_main(arguments), 0)
        with manifest.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["suggested_label"], "smoking")
        self.assertEqual(row["label"], "")
        self.assertEqual((manifest.parent / row["path"]).resolve(), image)
        original = manifest.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_main(arguments)
        self.assertEqual(manifest.read_bytes(), original)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_video_sampling_is_bounded_and_keeps_resolution_limit(self):
        from PIL import Image

        video = self.root / "a video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=160x120:r=10:d=3",
                "-c:v",
                "mpeg4",
                str(video),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        frames = sample_video(video, self.root / "frames", 1, 2, 64, 1)
        self.assertEqual(len(frames), 2)
        with Image.open(frames[0]) as image:
            self.assertLessEqual(max(image.size), 64)
        with self.assertRaisesRegex(ValueError, "already exist"):
            sample_video(video, self.root / "frames", 1, 2, 64, 1)


class FetchTests(unittest.TestCase):
    def test_empty_cookie_listing_retries_anonymously_and_keeps_download_cookies(self):
        with tempfile.TemporaryDirectory() as temp:
            cookie_file = Path(temp) / "cookies.txt"
            cookie_file.write_text("# Netscape HTTP Cookie File\n")
            entry = types.SimpleNamespace(
                id="id", path="image.jpg", local_path=str(Path(temp) / "image.jpg")
            )
            calls = []

            def download(*, cookies_file=None, **options):
                calls.append({"cookies_file": cookies_file, **options})
                return entry.local_path

            fake = types.SimpleNamespace(
                download_folder=Mock(side_effect=[[], [entry]]),
                download=download,
                DownloadError=RuntimeError,
                FileURLRetrievalError=RuntimeError,
            )
            stderr = io.StringIO()
            with (
                patch.dict("sys.modules", {"gdown": fake}),
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
            ):
                self.assertEqual(
                    fetch_data.main(
                        ["--output", temp, "--cookies-file", str(cookie_file)]
                    ),
                    0,
                )
            self.assertIn("Retrying the public folder listing", stderr.getvalue())
            first, second = fake.download_folder.call_args_list
            self.assertTrue(first.kwargs["use_cookies"])
            self.assertFalse(second.kwargs["use_cookies"])
            self.assertNotIn("cookies_file", second.kwargs)
            self.assertTrue(calls[0]["use_cookies"])
            self.assertEqual(calls[0]["cookies_file"], str(cookie_file))
            receipt = json.loads((Path(temp) / "download_receipt.json").read_text())
            self.assertFalse(receipt["listing_use_cookies"])

    def test_forced_anonymous_listing_preserves_authenticated_downloads(self):
        with tempfile.TemporaryDirectory() as temp:
            entry = types.SimpleNamespace(
                id="id", path="image.jpg", local_path=str(Path(temp) / "image.jpg")
            )
            fake = types.SimpleNamespace(
                download_folder=Mock(return_value=[entry]),
                download=Mock(return_value=entry.local_path),
                DownloadError=RuntimeError,
                FileURLRetrievalError=RuntimeError,
            )
            with (
                patch.dict("sys.modules", {"gdown": fake}),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    fetch_data.main(
                        ["--output", temp, "--use-cookies", "--anonymous-listing"]
                    ),
                    0,
                )
            fake.download_folder.assert_called_once()
            self.assertFalse(fake.download_folder.call_args.kwargs["use_cookies"])
            self.assertTrue(fake.download.call_args.kwargs["use_cookies"])

    def test_empty_listing_reports_discovery_failure_before_filtering(self):
        for flags, expected_calls in (([], 1), (["--use-cookies"], 2)):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "absent"
                fake = types.SimpleNamespace(
                    download_folder=Mock(return_value=[]),
                    download=Mock(),
                    DownloadError=RuntimeError,
                )
                stderr = io.StringIO()
                with (
                    patch.dict("sys.modules", {"gdown": fake}),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(stderr),
                ):
                    self.assertEqual(
                        fetch_data.main(
                            ["--output", str(output), "--list-only", *flags]
                        ),
                        1,
                    )
                self.assertIn("empty folder listing", stderr.getvalue())
                self.assertNotIn("none match --include", stderr.getvalue())
                self.assertEqual(fake.download_folder.call_count, expected_calls)
                fake.download.assert_not_called()
                self.assertFalse(output.exists())

    def test_unmatched_filter_does_not_retry_a_nonempty_listing(self):
        fake = types.SimpleNamespace(
            download_folder=Mock(
                return_value=[types.SimpleNamespace(path="image.jpg")]
            ),
            DownloadError=RuntimeError,
        )
        stderr = io.StringIO()
        with (
            patch.dict("sys.modules", {"gdown": fake}),
            redirect_stderr(stderr),
        ):
            self.assertEqual(
                fetch_data.main(["--include", "*.zip", "--use-cookies", "--list-only"]),
                1,
            )
        self.assertIn(
            "Drive listed 1 files, but none match --include", stderr.getvalue()
        )
        fake.download_folder.assert_called_once()

    def test_list_only_is_read_only_and_category_is_resolved(self):
        fake = types.SimpleNamespace(
            download_folder=Mock(
                return_value=[types.SimpleNamespace(local_path="video.avi")]
            ),
            DownloadError=RuntimeError,
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "absent"
            with (
                patch.dict("sys.modules", {"gdown": fake}),
                redirect_stdout(io.StringIO()),
            ):
                result = fetch_data.main(
                    [
                        "--category",
                        "unattended_items",
                        "--output",
                        str(output),
                        "--list-only",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            options = fake.download_folder.call_args.kwargs
            self.assertTrue(options["skip_download"])
            self.assertTrue(options["resume"])
            self.assertIn(
                fetch_data.CATEGORY_FOLDERS["unattended_items"], options["url"]
            )

    def test_retry_reuses_partial_downloads_per_file(self):
        class LinkError(RuntimeError):
            pass

        fake = types.SimpleNamespace(
            download_folder=Mock(),
            download=Mock(),
            DownloadError=RuntimeError,
            FileURLRetrievalError=LinkError,
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict("sys.modules", {"gdown": fake}),
            patch("scripts.fetch_data.time.sleep"),
            redirect_stdout(io.StringIO()),
        ):
            destination = Path(temp) / "dataset/image.jpg"
            fake.download_folder.return_value = [
                types.SimpleNamespace(
                    id="file-id", path="image.jpg", local_path=str(destination)
                )
            ]
            fake.download.side_effect = [
                RuntimeError("network failed"),
                str(destination),
            ]
            self.assertEqual(fetch_data.main(["--output", temp, "--retries", "1"]), 0)
            self.assertEqual(fake.download_folder.call_count, 1)
            self.assertEqual(fake.download.call_count, 2)
            self.assertTrue(fake.download.call_args.kwargs["resume"])
            self.assertFalse(fake.download.call_args.kwargs["use_cookies"])
            receipt = json.loads((Path(temp) / "download_receipt.json").read_text())
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["files"], [str(destination)])

    def test_public_link_failure_is_not_retried_and_identifies_file(self):
        class LinkError(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp:
            entries = [
                types.SimpleNamespace(
                    id="blocked-id",
                    path="01.avi",
                    local_path=str(Path(temp) / "01.avi"),
                ),
                types.SimpleNamespace(
                    id="next-id", path="02.avi", local_path=str(Path(temp) / "02.avi")
                ),
            ]
            fake = types.SimpleNamespace(
                download_folder=Mock(return_value=entries),
                download=Mock(side_effect=LinkError("Cannot retrieve public link")),
                DownloadError=RuntimeError,
                FileURLRetrievalError=LinkError,
            )
            error_output = io.StringIO()
            with (
                patch.dict("sys.modules", {"gdown": fake}),
                patch("scripts.fetch_data.time.sleep") as sleep,
                redirect_stdout(io.StringIO()),
                redirect_stderr(error_output),
            ):
                self.assertEqual(fetch_data.main(["--output", temp]), 1)
            fake.download.assert_called_once()
            sleep.assert_not_called()
            self.assertIn("01.avi", error_output.getvalue())
            self.assertIn(
                "https://drive.google.com/file/d/blocked-id/view",
                error_output.getvalue(),
            )
            receipt = json.loads((Path(temp) / "download_receipt.json").read_text())
            self.assertEqual(receipt["status"], "partial")
            self.assertEqual(receipt["failures"][0]["id"], "blocked-id")
            self.assertEqual(receipt["listed_files"], 2)

    def test_continue_on_error_downloads_and_extracts_remaining_files(self):
        class LinkError(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("train/one.jpg", "image contents")
            entries = [
                types.SimpleNamespace(
                    id="blocked-id",
                    path="01.avi",
                    local_path=str(Path(temp) / "01.avi"),
                ),
                types.SimpleNamespace(
                    id="archive-id", path="dataset.zip", local_path=str(archive)
                ),
            ]
            fake = types.SimpleNamespace(
                download_folder=Mock(return_value=entries),
                download=Mock(
                    side_effect=[LinkError("Cannot retrieve public link"), str(archive)]
                ),
                DownloadError=RuntimeError,
                FileURLRetrievalError=LinkError,
            )
            with (
                patch.dict("sys.modules", {"gdown": fake}),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    fetch_data.main(
                        [
                            "--output",
                            temp,
                            "--continue-on-error",
                            "--extract-zip",
                            "--use-cookies",
                        ]
                    ),
                    1,
                )
            # The ZIP already exists locally; it must be extracted without a
            # second request to Drive, even after a preceding access failure.
            self.assertEqual(fake.download.call_count, 1)
            self.assertTrue(fake.download.call_args.kwargs["use_cookies"])
            self.assertTrue(fake.download_folder.call_args.kwargs["use_cookies"])
            self.assertEqual(
                (Path(temp) / "dataset/train/one.jpg").read_text(), "image contents"
            )
            receipt = json.loads((Path(temp) / "download_receipt.json").read_text())
            self.assertEqual(receipt["files"], [str(archive)])
            self.assertEqual(len(receipt["failures"]), 1)
            self.assertEqual(
                receipt["extracted_directories"], [str(Path(temp) / "dataset")]
            )

    def test_interruption_preserves_receipt(self):
        class LinkError(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp:
            first = str(Path(temp) / "01.jpg")
            entries = [
                types.SimpleNamespace(
                    id=str(index),
                    path=f"{index}.jpg",
                    local_path=str(Path(temp) / f"{index}.jpg"),
                )
                for index in range(2)
            ]
            fake = types.SimpleNamespace(
                download_folder=Mock(return_value=entries),
                download=Mock(side_effect=[first, KeyboardInterrupt]),
                DownloadError=RuntimeError,
                FileURLRetrievalError=LinkError,
            )
            with (
                patch.dict("sys.modules", {"gdown": fake}),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(fetch_data.main(["--output", temp]), 130)
            receipt = json.loads((Path(temp) / "download_receipt.json").read_text())
            self.assertEqual(receipt["status"], "interrupted")
            self.assertEqual(receipt["files"], [first])

    def test_existing_empty_annotation_skips_drive_entirely(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "label.txt"
            destination.touch()
            entry = types.SimpleNamespace(
                id="id", path="label.txt", local_path=str(destination)
            )
            fake = types.SimpleNamespace(
                download=Mock(side_effect=AssertionError("No network request expected"))
            )
            pacer = Mock()
            result = fetch_data.download_entry(
                fake, entry, use_cookies=False, speed=None, retries=2, pacer=pacer
            )
            self.assertEqual(result, str(destination))
            fake.download.assert_not_called()
            pacer.wait.assert_not_called()

    def test_partial_file_does_not_count_as_completed(self):
        class LinkError(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "image.jpg"
            part = Path(temp) / "image.jpgrandom.part"
            part.write_bytes(b"partial")
            entry = types.SimpleNamespace(
                id="id", path="image.jpg", local_path=str(destination)
            )
            fake = types.SimpleNamespace(
                download=Mock(return_value=str(destination)),
                DownloadError=RuntimeError,
                FileURLRetrievalError=LinkError,
            )
            self.assertEqual(
                fetch_data.download_entry(
                    fake, entry, use_cookies=False, speed=None, retries=0
                ),
                str(destination),
            )
            fake.download.assert_called_once()
            self.assertTrue(fake.download.call_args.kwargs["resume"])
            self.assertEqual(part.read_bytes(), b"partial")

    def test_pacer_spaces_network_attempts(self):
        pacer = fetch_data.DownloadPacer(1)
        with (
            patch("scripts.fetch_data.time.monotonic", side_effect=[0.0, 0.25, 1.0]),
            patch("scripts.fetch_data.time.sleep") as sleep,
        ):
            pacer.wait()
            pacer.wait()
        sleep.assert_called_once_with(0.75)

    def test_include_filter_handles_windows_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            entries = [
                types.SimpleNamespace(
                    path="smoking\\images\\image.jpg", local_path="image.jpg"
                ),
                types.SimpleNamespace(
                    path="smoking\\archive.ZIP", local_path="archive.ZIP"
                ),
            ]
            fake = types.SimpleNamespace(
                download_folder=Mock(return_value=entries), DownloadError=RuntimeError
            )
            stdout = io.StringIO()
            with patch.dict("sys.modules", {"gdown": fake}), redirect_stdout(stdout):
                self.assertEqual(
                    fetch_data.main(
                        ["--output", temp, "--include", "smoking/*.zip", "--list-only"]
                    ),
                    0,
                )
            self.assertIn("archive.ZIP", stdout.getvalue())
            self.assertNotIn("image.jpg", stdout.getvalue())
            self.assertFalse((Path(temp) / "download_receipt.json").exists())

    def test_repeated_failures_stop_even_with_continue_on_error(self):
        class LinkError(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp:
            entries = [
                types.SimpleNamespace(
                    id=str(index),
                    path=f"{index}.txt",
                    local_path=str(Path(temp) / f"{index}.txt"),
                )
                for index in range(4)
            ]
            # A local skip between two failed requests must not reset the
            # consecutive network failure counter.
            Path(entries[1].local_path).touch()
            fake = types.SimpleNamespace(
                download_folder=Mock(return_value=entries),
                download=Mock(side_effect=LinkError("access blocked")),
                DownloadError=RuntimeError,
                FileURLRetrievalError=LinkError,
            )
            with (
                patch.dict("sys.modules", {"gdown": fake}),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    fetch_data.main(
                        [
                            "--output",
                            temp,
                            "--continue-on-error",
                            "--max-consecutive-errors",
                            "2",
                            "--request-interval",
                            "0",
                        ]
                    ),
                    1,
                )
            self.assertEqual(fake.download.call_count, 2)
            receipt = json.loads((Path(temp) / "download_receipt.json").read_text())
            self.assertEqual(receipt["skipped_existing"], 1)
            self.assertEqual(
                receipt["stopped_reason"], "2 consecutive downloads failed"
            )
            self.assertEqual(len(receipt["failures"]), 2)

    def test_explicit_cookie_file_is_forwarded(self):
        class LinkError(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp:
            cookie_file = Path(temp) / "cookies.txt"
            cookie_file.write_text("# Netscape HTTP Cookie File\n")
            destination = Path(temp) / "image.jpg"
            entry = types.SimpleNamespace(
                id="id", path="image.jpg", local_path=str(destination)
            )
            calls = []

            def download(*, cookies_file=None, **options):
                calls.append({"cookies_file": cookies_file, **options})
                return str(destination)

            fake = types.SimpleNamespace(
                download_folder=Mock(return_value=[entry]),
                download=download,
                DownloadError=RuntimeError,
                FileURLRetrievalError=LinkError,
            )
            with (
                patch.dict("sys.modules", {"gdown": fake}),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    fetch_data.main(
                        ["--output", temp, "--cookies-file", str(cookie_file)]
                    ),
                    0,
                )
            self.assertEqual(calls[0]["cookies_file"], str(cookie_file))
            self.assertTrue(calls[0]["use_cookies"])
            self.assertEqual(
                fake.download_folder.call_args.kwargs["cookies_file"], str(cookie_file)
            )

    def test_invalid_pacing_settings_fail_before_network(self):
        for arguments in (
            ["--request-interval", "nan"],
            ["--request-interval", "-1"],
            ["--max-consecutive-errors", "-1"],
        ):
            with (
                self.subTest(arguments=arguments),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                fetch_data.main(arguments)

    def test_zip_traversal_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                fetch_data.extract_zip(archive)
            self.assertFalse((Path(temp) / "escape.txt").exists())
            with zipfile.ZipFile(archive, "w") as handle:
                info = zipfile.ZipInfo("link")
                info.external_attr = 0o120777 << 16
                handle.writestr(info, "../escape.txt")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                fetch_data.extract_zip(archive)

    def test_zip_nested_files_extracted(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "images.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("train/one.txt", "contents")
            destination = fetch_data.extract_zip(archive)
            self.assertEqual((destination / "train/one.txt").read_text(), "contents")


class EvaluationTests(unittest.TestCase):
    def test_vlm_accepts_an_unambiguous_exact_label(self):
        for response in ("normal", '"normal".', "'normal'.", '"normal"', "smoking."):
            with self.subTest(response=response):
                expected = "smoking" if response.startswith("smoking") else "normal"
                self.assertEqual(
                    parse_vlm_response(response, DEFINITIONS).label, expected
                )
        self.assertIsNone(parse_vlm_response("normal or smoking", DEFINITIONS).label)

    def test_vlm_abstains_on_ambiguous_or_invalid_output(self):
        for text in (
            '{"label":"unknown"}',
            "The normal scene might contain smoking.",
            '{"label":"invented"}',
            '{"label":[]}',
            "{}",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_vlm_response(text, DEFINITIONS).label)
        self.assertEqual(
            parse_vlm_response('```json\n{"label":"smoking"}\n```', DEFINITIONS).label,
            "smoking",
        )

    def test_coverage_failures_and_repeat_deduplication(self):
        rows = [
            {
                "repeat": 0,
                "label": "normal",
                "prediction": "normal",
                "status": "ok",
                "anomaly_score": 0.1,
            },
            {
                "repeat": 0,
                "label": "smoking",
                "prediction": None,
                "status": "abstain",
                "anomaly_score": None,
            },
            {
                "repeat": 0,
                "label": "smoking",
                "prediction": None,
                "status": "error",
                "anomaly_score": None,
            },
            {
                "repeat": 1,
                "label": "normal",
                "prediction": "smoking",
                "status": "ok",
                "anomaly_score": 0.9,
            },
        ]
        metrics = detection_metrics(rows, list(DEFINITIONS))
        self.assertEqual(metrics["prediction_coverage"], 1 / 3)
        self.assertEqual(metrics["accuracy_including_failures"], 1 / 3)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertIsNone(metrics["binary"]["roc_auc"])

    def test_binary_detection_counts_wrong_anomaly_type_as_detected(self):
        labels = ["normal", "smoking", "littering"]
        rows = [
            {
                "repeat": 0,
                "label": "smoking",
                "prediction": "littering",
                "status": "ok",
                "anomaly_score": 0.8,
            },
            {
                "repeat": 0,
                "label": "normal",
                "prediction": "normal",
                "status": "ok",
                "anomaly_score": 0.1,
            },
        ]
        metrics = detection_metrics(rows, labels)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["binary"]["f1"], 1)
        self.assertEqual(metrics["binary"]["roc_auc"], 1)

    def test_auc_handles_ties_and_missing_classes(self):
        self.assertEqual(roc_auc([(False, 0.5), (True, 0.5)]), 0.5)
        self.assertEqual(roc_auc([(False, 0.2), (True, 0.9)]), 1)
        self.assertEqual(roc_auc([(True, 0.2), (False, 0.9)]), 0)
        self.assertIsNone(roc_auc([(True, 0.2)]))


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("psutil"),
    "ML dependencies not installed",
)
class WorkerTests(unittest.TestCase):
    def test_worker_streams_errors_abstentions_and_metrics(self):
        from PIL import Image

        predictions = iter(
            [
                Prediction("normal"),
                Prediction(None, raw_response="unknown"),
                ValueError("bad inference"),
            ]
        )

        def predict(image):
            value = next(predictions)
            if isinstance(value, Exception):
                raise value
            return value

        adapter = types.SimpleNamespace(predict=predict, prepare=Mock())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.png"
            Image.new("RGB", (8, 8)).save(image)
            options = {
                "threads": 1,
                "duty_cycle": 1,
                "max_samples_per_second": None,
                "cpu_cores": None,
                "max_rss_mb": None,
                "offline": True,
                "seed": 42,
                "cache_dir": None,
                "max_new_tokens": 32,
                "max_side": 512,
                "warmup": 0,
                "repeats": 1,
            }
            samples = [
                {
                    "path": str(image),
                    "label": "normal",
                    "split": "test",
                    "group_id": str(index),
                }
                for index in range(3)
            ]
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "model": "clip-vit-b32",
                        "output": temp,
                        "options": options,
                        "definitions": DEFINITIONS,
                        "samples": samples,
                    }
                )
            )
            with (
                patch("scripts.benchmark.create_adapter", return_value=adapter),
                patch("scripts.throttle.CPUSettings.configure_worker"),
                patch.dict(os.environ),
            ):
                self.assertEqual(benchmark.worker(config), 1)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["errors"], 1)
            self.assertEqual(summary["abstentions"], 1)
            self.assertEqual(summary["metrics"]["prediction_coverage"], 1 / 3)
            rows = [
                json.loads(line)
                for line in (root / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["status"] for row in rows], ["ok", "abstain", "error"]
            )

    def test_model_timeout_preserves_failed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "worker_config.json"
            config.write_text(
                json.dumps(
                    {
                        "model": "clip-vit-b32",
                        "output": temp,
                        "options": {},
                        "samples": [],
                    }
                )
            )
            # Start a real sleeper process to test termination, without a model download.
            original = subprocess.Popen
            with patch(
                "scripts.benchmark.subprocess.Popen",
                side_effect=lambda *args, **kwargs: original(
                    [os.sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
                ),
            ):
                summary = benchmark.run_model(config, timeout=0.1, max_rss_mb=None)
            self.assertEqual(summary["status"], "error")
            self.assertIn("timeout", summary["error"])
            self.assertTrue((root / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
