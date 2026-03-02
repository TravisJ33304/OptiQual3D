"""Evaluation metrics for 3D anomaly detection.

Implements the standard metrics used for benchmarking:
    - **AUROC** (Area Under ROC Curve) for image/sample-level detection.
    - **AU-PRO** (Area Under Per-Region Overlap) for point-level
      localisation, integrated up to FPR = 0.3.
    - **F1 Score** for binary classification at a given threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class DetectionMetrics:
    """Sample-level (image-level) anomaly detection metrics.

    Attributes:
        auroc: Area under the ROC curve.
        f1: F1 score at the optimal threshold.
        precision: Precision at the optimal threshold.
        recall: Recall at the optimal threshold.
        threshold: Optimal threshold (maximising F1).
    """

    auroc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    threshold: float = 0.5


@dataclass
class LocalisationMetrics:
    """Point-level anomaly localisation metrics.

    Attributes:
        au_pro: Area under the per-region overlap curve
            (integrated up to ``fpr_limit``).
        point_auroc: Per-point AUROC.
        fpr_limit: FPR integration limit for AU-PRO.
    """

    au_pro: float = 0.0
    point_auroc: float = 0.0
    fpr_limit: float = 0.3


@dataclass
class CategoryMetrics:
    """Combined metrics for a single object category.

    Attributes:
        category: Category name.
        detection: Sample-level detection metrics.
        localisation: Point-level localisation metrics.
        num_samples: Number of evaluated samples.
    """

    category: str = ""
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    localisation: LocalisationMetrics = field(default_factory=LocalisationMetrics)
    num_samples: int = 0


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Compute Area Under the ROC Curve.

    Args:
        labels: ``(N,)`` binary ground-truth labels.
        scores: ``(N,)`` predicted anomaly scores.

    Returns:
        AUROC value in ``[0, 1]``.
    """
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present; AUROC undefined. Returning 0.")
        return 0.0
    return float(roc_auc_score(labels, scores))


def compute_f1_optimal(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Compute F1 score at the threshold that maximises it.

    Args:
        labels: ``(N,)`` binary ground-truth labels.
        scores: ``(N,)`` predicted anomaly scores.

    Returns:
        Tuple of ``(best_f1, optimal_threshold)``.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_f1, best_threshold


def compute_detection_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> DetectionMetrics:
    """Compute all sample-level detection metrics.

    Args:
        labels: ``(N,)`` binary sample labels (0 = normal, 1 = anomalous).
        scores: ``(N,)`` predicted global anomaly scores.

    Returns:
        :class:`DetectionMetrics` dataclass.
    """
    auroc = compute_auroc(labels, scores)
    f1, threshold = compute_f1_optimal(labels, scores)

    binary_pred = (scores >= threshold).astype(int)
    tp = int(((binary_pred == 1) & (labels == 1)).sum())
    fp = int(((binary_pred == 1) & (labels == 0)).sum())
    fn = int(((binary_pred == 0) & (labels == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return DetectionMetrics(
        auroc=auroc,
        f1=f1,
        precision=precision,
        recall=recall,
        threshold=threshold,
    )


def compute_au_pro(
    point_masks: list[np.ndarray],
    point_scores: list[np.ndarray],
    fpr_limit: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """Compute the Area Under Per-Region Overlap (AU-PRO) curve.

    The AU-PRO metric evaluates localisation quality by measuring
    the overlap between predicted and ground-truth anomaly regions
    across FPR values, integrated from 0 to ``fpr_limit``.

    Args:
        point_masks: List of ``(N_i,)`` binary ground-truth masks per sample.
        point_scores: List of ``(N_i,)`` predicted per-point scores per sample.
        fpr_limit: Maximum false positive rate for integration.
        num_thresholds: Number of score thresholds to evaluate.

    Returns:
        AU-PRO value normalised to ``[0, 1]``.
    """
    # Aggregate all scores to determine threshold range
    all_scores = np.concatenate(point_scores)
    thresholds = np.linspace(
        all_scores.min(), all_scores.max(), num_thresholds
    )

    fprs: list[float] = []
    pros: list[float] = []

    for threshold in thresholds:
        total_fp = 0
        total_tn = 0
        region_overlaps: list[float] = []

        for mask, scores in zip(point_masks, point_scores):
            binary_pred = (scores >= threshold).astype(int)

            # Per-region overlap for anomalous regions in this sample
            if mask.sum() > 0:
                overlap = float((binary_pred[mask == 1] == 1).mean())
                region_overlaps.append(overlap)

            # FPR contribution from normal points
            normal_points = mask == 0
            if normal_points.sum() > 0:
                total_fp += int(binary_pred[normal_points].sum())
                total_tn += int((~binary_pred.astype(bool))[normal_points].sum())

        fpr = total_fp / max(total_fp + total_tn, 1)
        pro = float(np.mean(region_overlaps)) if region_overlaps else 0.0

        fprs.append(fpr)
        pros.append(pro)

    # Sort by FPR
    sorted_indices = np.argsort(fprs)
    fprs_sorted = np.array(fprs)[sorted_indices]
    pros_sorted = np.array(pros)[sorted_indices]

    # Integrate up to fpr_limit
    valid = fprs_sorted <= fpr_limit
    if valid.sum() < 2:
        return 0.0

    au_pro = float(auc(fprs_sorted[valid], pros_sorted[valid]))
    # Normalise by fpr_limit
    au_pro /= fpr_limit

    return au_pro


def compute_localisation_metrics(
    point_masks: list[np.ndarray],
    point_scores: list[np.ndarray],
    fpr_limit: float = 0.3,
) -> LocalisationMetrics:
    """Compute all point-level localisation metrics.

    Args:
        point_masks: List of ``(N_i,)`` binary ground-truth masks.
        point_scores: List of ``(N_i,)`` predicted per-point scores.
        fpr_limit: FPR integration limit for AU-PRO.

    Returns:
        :class:`LocalisationMetrics` dataclass.
    """
    # Point-level AUROC
    all_labels = np.concatenate(point_masks)
    all_scores = np.concatenate(point_scores)
    point_auroc = compute_auroc(all_labels, all_scores)

    # AU-PRO
    au_pro = compute_au_pro(point_masks, point_scores, fpr_limit)

    return LocalisationMetrics(
        au_pro=au_pro,
        point_auroc=point_auroc,
        fpr_limit=fpr_limit,
    )
