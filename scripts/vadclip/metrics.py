"""Frame-level detection metrics for the VadCLIP benchmark (sklearn-backed).

All scores are computed on *valid* frames only (padded positions masked out by length),
so a short session never contributes spurious zero-scored negatives. The headline numbers
are the canonical WSVAD frame/segment-detection metrics: AUPR (average precision) and
AUROC, plus P/R/F1 at the max-F1 operating point for an interpretable accuracy figure.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def frame_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """y_true/y_score are 1-D arrays over the same set of valid frames.

    Returns a dict with auroc, aupr (average precision), and P/R/F1 at the max-F1 threshold.
    Degenerate single-class inputs yield NaN for the rank-based scores (reported as such).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    out = {"n": int(len(y_true)), "n_pos": int(y_true.sum()), "n_neg": int((1 - y_true).sum())}
    if len(np.unique(y_true)) < 2:
        out.update({"auroc": float("nan"), "aupr": float("nan")})

    # max-F1 operating point from the precision-recall curve
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    i = int(np.argmax(f1s))
    t_opt = float(thr[i - 1]) if i >= 1 else float("inf")   # thr has len(prec)-1 entries

    pred = (y_score >= t_opt).astype(int) if np.isfinite(t_opt) else np.zeros_like(y_true)
    out.update({
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "aupr": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "f1_opt": float(f1s[i]),
        "precision_opt": float(prec[i]),
        "recall_opt": float(rec[i]),
        "threshold_opt": t_opt,
        "acc_at_f1opt": float((pred == y_true).mean()),
    })
    return out


def session_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Clip-level (any-positive-in-clip) metrics -- reported for completeness only.

    Because every clip in this corpus contains at least one positive frame, the clip label is
    all-ones and these are degenerate; we surface them to document *why* supervision is done
    at frame level rather than with the paper's video-MIL weak label.
    """
    return frame_metrics(y_true, y_score)
