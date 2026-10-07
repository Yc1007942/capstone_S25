"""Benchmark anomaly models sequentially in isolated, CPU-only workers."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data import (
    DEFAULT_DEFINITIONS,
    PROJECT_ROOT,
    Sample,
    file_sha256,
    read_definitions,
    read_manifest,
    select_test_samples,
)
from scripts.metrics import detection_metrics, percentile
from scripts.models import DEFAULT_MODELS, MODELS, create_adapter, load_image
from scripts.throttle import Throttler, add_cpu_arguments, settings_from_args


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


class ResourceMonitor:
    """Sample worker RSS during loading, reference preparation, and inference."""

    def __init__(self, max_rss_mb: float | None):
        import psutil

        self.process = psutil.Process()
        self.max_rss_mb = max_rss_mb
        self.peak_rss_mb = 0.0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stopped.is_set():
            self.peak_rss_mb = max(
                self.peak_rss_mb, self.process.memory_info().rss / 1024**2
            )
            self.stopped.wait(0.05)

    def check(self) -> None:
        rss = self.process.memory_info().rss / 1024**2
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        if self.max_rss_mb is not None and self.peak_rss_mb > self.max_rss_mb:
            raise MemoryError(f"Worker exceeded {self.max_rss_mb:g} MiB RSS")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.stopped.set()
        self.thread.join()


def worker(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = argparse.Namespace(**config["options"])
    output = Path(config["output"])
    name = config["model"]
    settings = settings_from_args(args)
    settings.configure_environment()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    definitions = config["definitions"]
    samples = [
        Sample(Path(row["path"]), row["label"], row["split"], row["group_id"])
        for row in config["samples"]
    ]
    references = [sample for sample in samples if sample.split == "reference"]
    tests = [sample for sample in samples if sample.split == "test"]
    summary = {
        "model": name,
        "checkpoint": MODELS[name].checkpoint,
        "model_revision": MODELS[name].revision,
        "status": "error",
    }
    rows = []
    monitor = None
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        # Reject missing reference classes before downloading any vision weights.
        if MODELS[name].kind in {"vision", "timm"}:
            missing = set(definitions) - {sample.label for sample in references}
            if missing:
                raise ValueError(
                    "Vision baselines need labeled reference images for every class; missing "
                    + ", ".join(sorted(missing))
                )
        with ResourceMonitor(args.max_rss_mb) as monitor:
            settings.configure_worker()
            import torch

            torch.manual_seed(args.seed)
            throttle = Throttler(settings)
            load_start = time.perf_counter()
            adapter = create_adapter(
                name,
                definitions,
                args.cache_dir,
                args.offline,
                args.max_new_tokens,
                args.max_side,
            )
            summary["load_seconds"] = time.perf_counter() - load_start
            monitor.check()
            preparation_start = time.perf_counter()
            adapter.prepare(references, throttle, monitor.check, args.max_side)
            summary["reference_seconds"] = time.perf_counter() - preparation_start
            warmup_start = time.perf_counter()
            for _ in range(args.warmup):
                monitor.check()
                with throttle.operation():
                    adapter.predict(load_image(tests[0].path, args.max_side))
            summary["warmup_seconds"] = time.perf_counter() - warmup_start
            sleep_before = throttle.total_sleep_seconds
            measured_start = time.perf_counter()
            measured_cpu_start = time.process_time()
            with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for repeat in range(args.repeats):
                    for sample in tests:
                        monitor.check()
                        row = {
                            "path": str(sample.path),
                            "group_id": sample.group_id,
                            "label": sample.label,
                            "repeat": repeat,
                            "prediction": None,
                            "anomaly_score": None,
                            "scores": {},
                            "status": "error",
                        }
                        with throttle.operation():
                            step_start = time.perf_counter()
                            try:
                                prediction = adapter.predict(
                                    load_image(sample.path, args.max_side)
                                )
                                row.update(
                                    prediction=prediction.label,
                                    anomaly_score=prediction.anomaly_score,
                                    scores=prediction.scores,
                                    raw_response=prediction.raw_response,
                                    generated_tokens=prediction.generated_tokens,
                                    status="ok" if prediction.label else "abstain",
                                )
                            except (MemoryError, KeyboardInterrupt):
                                raise
                            except Exception as exc:  # noqa: BLE001 -- record adapter failures per image
                                row["error"] = f"{type(exc).__name__}: {exc}"
                            row["latency_seconds"] = time.perf_counter() - step_start
                        rows.append(row)
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
            measured_seconds = time.perf_counter() - measured_start
            measured_cpu = time.process_time() - measured_cpu_start
            latencies = [
                row["latency_seconds"] for row in rows if row["status"] != "error"
            ]
            completed = len(latencies)
            summary.update(
                status="ok"
                if all(row["status"] != "error" for row in rows)
                else "partial",
                samples=len(tests),
                repeats=args.repeats,
                attempts=len(rows),
                completed_inferences=completed,
                errors=sum(row["status"] == "error" for row in rows),
                abstentions=sum(row["status"] == "abstain" for row in rows),
                latency_mean_seconds=statistics.mean(latencies) if latencies else None,
                latency_p50_seconds=percentile(latencies, 0.5),
                latency_p95_seconds=percentile(latencies, 0.95),
                active_samples_per_second=completed / sum(latencies)
                if sum(latencies)
                else None,
                wall_samples_per_second=completed / measured_seconds
                if measured_seconds
                else None,
                measured_wall_seconds=measured_seconds,
                measured_cpu_seconds=measured_cpu,
                throttle_sleep_seconds=throttle.total_sleep_seconds - sleep_before,
                average_cpu_cores=measured_cpu / measured_seconds
                if measured_seconds
                else None,
                peak_rss_mb=monitor.peak_rss_mb,
                metrics=detection_metrics(rows, list(definitions)),
                package_versions={
                    package: importlib.metadata.version(package)
                    for package in (
                        "torch",
                        "transformers",
                        "timm",
                        "open_clip_torch",
                        "Pillow",
                        "psutil",
                    )
                },
            )
    except Exception as exc:  # noqa: BLE001 -- isolate model failures and preserve diagnostics
        summary["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        summary["total_wall_seconds"] = time.perf_counter() - started
        summary["total_cpu_seconds"] = time.process_time() - cpu_started
        if monitor:
            summary["peak_rss_mb"] = monitor.peak_rss_mb
        write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "ok" else 1


def stop_worker(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_model(config_path: Path, timeout: float, max_rss_mb: float | None) -> dict:
    import psutil

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"])
    reason = None
    peak_rss_mb = 0.0
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-config",
                str(config_path),
            ],
            stdout=log,
            stderr=log,
        )
        started = time.perf_counter()
        watched = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:
                    peak_rss_mb = max(peak_rss_mb, watched.memory_info().rss / 1024**2)
                except psutil.NoSuchProcess:
                    pass
                if max_rss_mb is not None and peak_rss_mb > max_rss_mb:
                    reason = f"Worker exceeded {max_rss_mb:g} MiB RSS"
                    break
                if time.perf_counter() - started > timeout:
                    reason = f"Worker exceeded {timeout:g} second model timeout (including loading and pacing)"
                    break
                time.sleep(0.05)
        finally:
            stop_worker(process)
    summary_path = output / "summary.json"
    if reason or not summary_path.is_file():
        summary = {
            "model": config["model"],
            "checkpoint": MODELS[config["model"]].checkpoint,
            "status": "error",
            "error": reason or f"Worker exited {process.returncode} without a summary",
            "total_wall_seconds": time.perf_counter() - started,
            "peak_rss_mb": peak_rss_mb,
        }
        write_json(summary_path, summary)
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary


def write_comparison(output: Path, summaries: list[dict]) -> None:
    columns = [
        "model",
        "status",
        "samples",
        "latency_mean_seconds",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "active_samples_per_second",
        "wall_samples_per_second",
        "peak_rss_mb",
        "load_seconds",
        "errors",
        "abstentions",
        "prediction_coverage",
        "accuracy_including_failures",
        "macro_f1",
        "binary_f1",
        "roc_auc",
        "error",
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            row = {column: summary.get(column) for column in columns}
            metrics = summary.get("metrics", {})
            for column in (
                "prediction_coverage",
                "accuracy_including_failures",
                "macro_f1",
            ):
                row[column] = metrics.get(column)
            row["binary_f1"] = (metrics.get("binary") or {}).get("f1")
            row["roc_auc"] = (metrics.get("binary") or {}).get("roc_auc")
            writer.writerow(row)
    write_json(output / "results.json", {"models": summaries})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV of test/reference image paths and reviewed labels",
    )
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names, or all",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="New results directory; defaults to results/<UTC timestamp>",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum test images, randomly selected (0 means all)",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup inferences excluded from timings"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-side", type=int, default=512, help="Maximum input image edge"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Generative VLM token budget"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900,
        help="Timeout per model, including load and sleeps",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        help="Optional sampled per-worker memory limit in MiB",
    )
    parser.add_argument(
        "--cache-dir", type=str, help="Optional Hugging Face weight cache"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only cached models; no network access",
    )
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    add_cpu_arguments(parser)
    args = parser.parse_args(argv)
    if args.worker_config:
        return worker(args.worker_config)
    if args.list_models:
        for name, spec in MODELS.items():
            print(f"{name:22} {spec.kind:6} {spec.checkpoint}")
        return 0
    if not args.manifest:
        parser.error("--manifest is required unless using --list-models")
    names = list(MODELS) if args.models == ["all"] else list(dict.fromkeys(args.models))
    unknown = set(names) - MODELS.keys()
    if unknown:
        parser.error("Unknown models: " + ", ".join(sorted(unknown)))
    if (
        min(args.repeats, args.max_side, args.max_new_tokens) < 1
        or min(args.warmup, args.limit) < 0
    ):
        parser.error(
            "repeats, max-side, and max-new-tokens must be positive; warmup and limit nonnegative"
        )
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive and finite")
    if args.max_rss_mb is not None and (
        not math.isfinite(args.max_rss_mb) or args.max_rss_mb <= 0
    ):
        parser.error("max-rss-mb must be positive and finite")
    try:
        import psutil

        settings = settings_from_args(args)
        settings.configure_environment()
        definitions = read_definitions(args.definitions)
        samples = read_manifest(args.manifest.resolve(), definitions)
        tests = select_test_samples(samples, args.limit or None, args.seed)
        references = [sample for sample in samples if sample.split == "reference"]
        output = (
            args.output
            or PROJECT_ROOT
            / "results"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        ).resolve()
        output.mkdir(parents=True, exist_ok=False)
        options = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        serialized_samples = [
            {**asdict(sample), "path": str(sample.path)}
            for sample in references + tests
        ]
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_model": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "total_ram_mb": psutil.virtual_memory().total / 1024**2,
            "available_cpu_cores": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "options": options,
            "definitions": definitions,
            "manifest_sha256": file_sha256(args.manifest),
            "definitions_sha256": file_sha256(args.definitions),
            "samples": serialized_samples,
            "models": names,
        }
        write_json(output / "run.json", metadata)
        summaries = []
        for name in names:
            model_output = output / name
            model_output.mkdir()
            config_path = model_output / "worker_config.json"
            write_json(
                config_path,
                {
                    "model": name,
                    "output": str(model_output),
                    "options": options,
                    "definitions": definitions,
                    "samples": serialized_samples,
                },
            )
            print(
                f"Running {name} on {len(tests)} test images ({settings.threads} CPU threads)...",
                flush=True,
            )
            summary = run_model(config_path, args.timeout_seconds, args.max_rss_mb)
            summaries.append(summary)
            # Write after every model, so completed work survives interruption.
            write_comparison(output, summaries)
            print(
                f"  {summary['status']}: {summary.get('error', str(summary.get('completed_inferences', 0)) + ' completed inferences')}",
                flush=True,
            )
        print(f"Results: {output / 'comparison.csv'}")
        return 0 if all(summary["status"] == "ok" for summary in summaries) else 1
    except (ValueError, OSError, ImportError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; completed model results and streamed predictions are preserved.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
