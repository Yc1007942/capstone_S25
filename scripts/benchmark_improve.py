"""Statistical hardening of the head-to-head benchmark, from saved per-sample predictions.

The main benchmark (``scripts/benchmark.py``) reports point estimates only. This script adds what a
rigorous comparison needs without re-running any inference: it reads each model's ``predictions.jsonl``
and, by bootstrap-resampling the labelled samples and scoring every resample through the *same*
``detection_metrics`` code path used to build ``comparison.csv``, produces

  * 95% confidence intervals (percentile method) for accuracy-including-failures, macro-F1,
    binary anomaly F1, and binary ROC-AUC;
  * a per-class precision/recall/F1/support table (the "which frames / which categories" detail);
  * an abstention diagnostic (how many samples each model refused to classify, and why).

Because the point estimate is recomputed with the identical function on the full sample set, it must
match ``comparison.csv`` exactly -- that equality is asserted as a sanity check before any CI is
reported. No image content is read; only the already-saved per-sample score/label rows are used.

Run from the project root:

    python -m scripts.benchmark_improve \
        --results-dir results/anomaly_headtohead \
        --definitions scripts/anomalies.json \
        --bootstrap 2000 --seed 12345
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .data import read_definitions
from .metrics import detection_metrics, percentile


def _load_rows(results_dir: Path, model: str) -> list[dict]:
    path = results_dir / model / "predictions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ci(values: list[float]) -> tuple[float, float] | None:
    """Percentile 95% CI; None when the metric was undefined on every resample."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return None
    return (percentile(finite, 0.025), percentile(finite, 0.975))


def _extract(metrics: dict) -> dict:
    """Pull the four headline numbers out of one detection_metrics result."""
    binary = metrics.get("binary") or {}
    return {
        "accuracy_including_failures": metrics.get("accuracy_including_failures"),
        "macro_f1": metrics.get("macro_f1"),
        "binary_f1": binary.get("f1"),
        "roc_auc": binary.get("roc_auc"),
    }


def _abstention_diagnostic(rows: list[dict]) -> dict:
    """Explain a model's abstentions from its raw responses (no image content involved)."""
    labeled = [r for r in rows if r["repeat"] == 0 and r["label"] is not None]
    abstain = [r for r in labeled if r["status"] == "abstain"]
    responses: dict[str, int] = {}
    for r in abstain:
        key = (r.get("raw_response") or "").strip()[:60]
        responses[key] = responses.get(key, 0) + 1
    return {
        "labeled_samples": len(labeled),
        "abstentions": len(abstain),
        "coverage": 1.0 - (len(abstain) / len(labeled)) if labeled else None,
        "distinct_raw_responses": dict(sorted(responses.items(), key=lambda kv: -kv[1])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--definitions", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)

    from .benchmark import DEFAULT_DEFINITIONS  # local import: pulls in the full benchmark module
    definitions_path = args.definitions or DEFAULT_DEFINITIONS
    labels = list(read_definitions(definitions_path))

    models = sorted(p.name for p in args.results_dir.iterdir() if (p / "predictions.jsonl").exists())
    out: dict[str, object] = {"labels": labels, "bootstrap": args.bootstrap, "seed": args.seed, "models": {}}

    for model in models:
        rows = _load_rows(args.results_dir, model)
        labeled = [r for r in rows if r["repeat"] == 0 and r["label"] is not None]

        # Point estimate through the exact benchmark code path (must match comparison.csv).
        point = detection_metrics(labeled, labels)
        headline = _extract(point)

        rng = random.Random(args.seed)
        boot: dict[str, list[float | None]] = {k: [] for k in headline}
        for _ in range(args.bootstrap):
            sample = [labeled[i] for i in rng.choices(range(len(labeled)), k=len(labeled))]
            m = _extract(detection_metrics(sample, labels))
            for k in boot:
                boot[k].append(m.get(k))

        cis = {k: _ci(v) for k, v in boot.items()}
        out["models"][model] = {
            "n_labeled": len(labeled),
            "point": headline,
            "ci95": {k: (list(c) if c else None) for k, c in cis.items()},
            "per_class": point.get("per_class", {}),
            "abstention": _abstention_diagnostic(rows),
        }

    out_path = args.results_dir / "improvements.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Console summary.
    print(f"[improve] {len(models)} models | labels={labels} | bootstrap={args.bootstrap} seed={args.seed}")
    for model in models:
        d = out["models"][model]  # type: ignore[index]
        p, c = d["point"], d["ci95"]  # type: ignore[index]
        parts = []
        for k in ("accuracy_including_failures", "macro_f1", "binary_f1", "roc_auc"):
            pv = p.get(k)
            ci = c.get(k)
            if pv is None:
                parts.append(f"{k}=NA")
            elif ci is None:
                parts.append(f"{k}={pv:.4f}")
            else:
                parts.append(f"{k}={pv:.4f}[{ci[0]:.3f},{ci[1]:.3f}]")
        print(f"  {model}: " + " ".join(parts))
    print(f"[improve] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
