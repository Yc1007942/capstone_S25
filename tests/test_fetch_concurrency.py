"""Concurrent fetching: overlap, pacing, cookie isolation, stopping, and resume."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import types
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import fetch_data


class LinkError(RuntimeError):
    pass


class DownloadCancelled(RuntimeError):
    pass


class ConcurrentFetchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def entries(self, count):
        return [
            types.SimpleNamespace(
                id=str(index),
                path=f"{index}.txt",
                local_path=str(self.root / f"{index}.txt"),
            )
            for index in range(count)
        ]

    def run_fetch(self, entries, download, *arguments):
        fake = types.SimpleNamespace(
            download_folder=Mock(return_value=entries),
            download=download,
            DownloadError=RuntimeError,
            FileURLRetrievalError=LinkError,
            DownloadCancelled=DownloadCancelled,
        )
        with (
            patch.dict("sys.modules", {"gdown": fake}),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            code = fetch_data.main(
                [
                    "--output",
                    str(self.root),
                    "--workers",
                    "2",
                    "--request-interval",
                    "0",
                    *arguments,
                ]
            )
        receipt_path = self.root / "download_receipt.json"
        receipt = (
            json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        )
        return code, receipt

    def test_transfers_overlap_with_bounded_workers_and_skip_existing_empty_files(self):
        entries = self.entries(5)
        Path(entries[1].local_path).touch()
        gate = threading.Barrier(2)
        lock = threading.Lock()
        calls = []
        active = peak = 0

        def download(*, id, output, **options):
            nonlocal active, peak
            with lock:
                calls.append(id)
                first_pair = len(calls) <= 2
                active += 1
                peak = max(peak, active)
            try:
                if first_pair:
                    gate.wait(timeout=2)
                self.assertTrue(options["resume"])
                Path(output).write_text(id)
                return output
            finally:
                with lock:
                    active -= 1

        code, receipt = self.run_fetch(entries, download)
        self.assertEqual(code, 0)
        self.assertEqual(peak, 2)
        self.assertEqual(set(calls), {"0", "2", "3", "4"})
        self.assertEqual(receipt["skipped_existing"], 1)
        self.assertEqual(len(receipt["files"]), 5)
        self.assertEqual(receipt["workers"], 2)

    def test_interval_is_shared_across_workers(self):
        starts = []
        lock = threading.Lock()

        def download(*, id, output, **options):
            with lock:
                starts.append(time.monotonic())
            Path(output).write_text(id)
            return output

        code, _ = self.run_fetch(
            self.entries(8), download, "--workers", "4", "--request-interval", "0.04"
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(starts), 8)
        self.assertGreaterEqual(max(starts) - min(starts), 0.26)

    def test_cookie_files_are_isolated_and_cleaned_without_changing_source(self):
        source = self.root / "cookies.txt"
        original = "# Netscape HTTP Cookie File\n# test session\n"
        source.write_text(original)
        gate = threading.Barrier(3)
        cookie_paths = []
        lock = threading.Lock()

        def download(*, id, output, cookies_file, **options):
            self.assertTrue(options["use_cookies"])
            cookie_path = Path(cookies_file)
            self.assertEqual(cookie_path.read_text(), original)
            with lock:
                cookie_paths.append(cookie_path)
            cookie_path.write_text(f"worker {id}")
            gate.wait(timeout=2)
            self.assertEqual(cookie_path.read_text(), f"worker {id}")
            Path(output).write_text(id)
            return output

        code, receipt = self.run_fetch(
            self.entries(3), download, "--workers", "3", "--cookies-file", str(source)
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(set(cookie_paths)), 3)
        self.assertNotIn(source, cookie_paths)
        self.assertEqual(source.read_text(), original)
        self.assertTrue(all(not path.exists() for path in cookie_paths))
        self.assertEqual(receipt["status"], "complete")

    def test_failure_limit_stops_scheduling_and_records_inflight_success(self):
        gate = threading.Barrier(2)
        calls = []

        def download(*, id, output, cancel, **options):
            calls.append(id)
            gate.wait(timeout=2)
            if id == "0":
                raise LinkError("access blocked")
            self.assertTrue(cancel.wait(2))
            # Publication can finish just as cancellation is requested.
            Path(output).write_text(id)
            return output

        code, receipt = self.run_fetch(
            self.entries(8),
            download,
            "--continue-on-error",
            "--max-consecutive-errors",
            "1",
        )
        self.assertEqual(code, 1)
        self.assertEqual(set(calls), {"0", "1"})
        self.assertEqual(receipt["status"], "partial")
        self.assertEqual(receipt["stopped_reason"], "1 consecutive downloads failed")
        self.assertEqual(receipt["files"], [str(self.root / "1.txt")])
        self.assertEqual(len(receipt["failures"]), 1)

    def test_cancelled_transfers_keep_partial_files_and_resume_on_rerun(self):
        gate = threading.Barrier(2)
        partial = self.root / "1.txt-test.part"
        calls = []

        def blocked_download(*, id, output, cancel, **options):
            calls.append(id)
            if id == "1":
                partial.write_text("partial")
            gate.wait(timeout=2)
            if id == "0":
                raise LinkError("access blocked")
            self.assertTrue(cancel.wait(2))
            raise DownloadCancelled()

        entries = self.entries(4)
        code, receipt = self.run_fetch(entries, blocked_download)
        self.assertEqual(code, 1)
        self.assertEqual(set(calls), {"0", "1"})
        self.assertEqual(receipt["cancelled_files"], 1)
        self.assertEqual(len(receipt["failures"]), 1)
        self.assertEqual(partial.read_text(), "partial")

        def resumed_download(*, id, output, **options):
            self.assertTrue(options["resume"])
            if id == "1":
                self.assertEqual(partial.read_text(), "partial")
                partial.replace(output)
            else:
                Path(output).write_text(id)
            return output

        code, receipt = self.run_fetch(entries, resumed_download)
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(len(receipt["files"]), 4)
        self.assertEqual((self.root / "1.txt").read_text(), "partial")

    def test_interrupt_cancels_workers_and_preserves_completed_receipt(self):
        gate = threading.Barrier(2)
        partial = self.root / "2.txt-test.part"

        def download(*, id, output, cancel, **options):
            if id == "0":
                Path(output).write_text(id)
                return output
            if id == "2":
                partial.write_text("partial")
            gate.wait(timeout=2)
            if id == "1":
                raise KeyboardInterrupt()
            self.assertTrue(cancel.wait(2))
            raise DownloadCancelled()

        code, receipt = self.run_fetch(self.entries(5), download)
        self.assertEqual(code, 130)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["files"], [str(self.root / "0.txt")])
        self.assertEqual(partial.read_text(), "partial")

    def test_zip_extraction_waits_until_transfers_finish(self):
        contents = io.BytesIO()
        with zipfile.ZipFile(contents, "w") as archive:
            archive.writestr("one.txt", "archive member")
        entries = self.entries(2)
        entries[0].path = "dataset.zip"
        entries[0].local_path = str(self.root / "dataset.zip")
        loose_finished = threading.Event()

        def download(*, id, output, **options):
            if id == "0":
                Path(output).write_bytes(contents.getvalue())
            else:
                time.sleep(0.02)
                Path(output).write_text(id)
                loose_finished.set()
            return output

        original_extract = fetch_data.extract_zip

        def extract(archive):
            self.assertTrue(loose_finished.is_set())
            return original_extract(archive)

        with patch("scripts.fetch_data.extract_zip", side_effect=extract):
            code, receipt = self.run_fetch(entries, download, "--extract-zip")
        self.assertEqual(code, 0)
        self.assertEqual((self.root / "dataset/one.txt").read_text(), "archive member")
        self.assertEqual(receipt["extracted_directories"], [str(self.root / "dataset")])

    def test_duplicate_destinations_fail_before_downloads(self):
        entries = self.entries(2)
        entries[1].local_path = entries[0].local_path
        download = Mock()
        code, receipt = self.run_fetch(entries, download)
        self.assertEqual(code, 1)
        self.assertIsNone(receipt)
        download.assert_not_called()

    def test_invalid_workers_fail_before_network(self):
        for workers in ("0", "-1"):
            with (
                self.subTest(workers=workers),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                fetch_data.main(["--workers", workers])


if __name__ == "__main__":
    unittest.main()
