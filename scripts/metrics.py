"""Performance and classification metrics without an extra scientific dependency."""

from __future__ import annotations

import math
import statistics


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def roc_auc(pairs: list[tuple[bool, float]]) -> float | None:
    """Rank-sum AUC with average ranks for tied scores."""
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(pairs, key=lambda pair: pair[1])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        rank_sum += rank * sum(label for label, _ in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def detection_metrics(rows: list[dict], labels: list[str]) -> dict:
    """Score repeat zero, report abstention coverage, keep error denominators visible."""
    labeled = [row for row in rows if row["repeat"] == 0 and row["label"] is not None]
    valid = [
        row for row in labeled if row["status"] == "ok" and row["prediction"] in labels
    ]
    count = len(labeled)
    result = {
        "labeled_samples": count,
        "valid_predictions": len(valid),
        "prediction_coverage": len(valid) / count if count else None,
        "accuracy_including_failures": sum(
            row["label"] == row["prediction"] for row in valid
        )
        / count
        if count
        else None,
    }
    if not valid:
        result.update(
            accuracy=None, macro_f1=None, binary=None, per_class={}, confusion_matrix={}
        )
        return result
    per_class = {}
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for row in valid:
        matrix[row["label"]][row["prediction"]] += 1
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][predicted] for predicted in labels if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        support = sum(matrix[label].values())
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "support": support,
        }
    tp = sum(
        row["label"] != "normal" and row["prediction"] != "normal" for row in valid
    )
    fp = sum(
        row["label"] == "normal" and row["prediction"] != "normal" for row in valid
    )
    fn = sum(
        row["label"] != "normal" and row["prediction"] == "normal" for row in valid
    )
    tn = sum(
        row["label"] == "normal" and row["prediction"] == "normal" for row in valid
    )
    score_pairs = [
        (row["label"] != "normal", row["anomaly_score"])
        for row in valid
        if row["anomaly_score"] is not None and math.isfinite(row["anomaly_score"])
    ]
    result.update(
        accuracy=sum(row["label"] == row["prediction"] for row in valid) / len(valid),
        macro_f1=None,
        per_class=per_class,
        confusion_matrix=matrix,
        binary={
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "roc_auc": roc_auc(score_pairs),
            "auc_samples": len(score_pairs),
        },
    )
    active_labels = [
        label
        for label in labels
        if sum(matrix[label].values())
        or sum(matrix[actual][label] for actual in labels)
    ]
    result["macro_f1"] = statistics.mean(
        per_class[label]["f1"] for label in active_labels
    )
    return result
