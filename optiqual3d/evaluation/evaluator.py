"""Evaluation pipeline for zero-shot 3D anomaly detection.

Orchestrates model inference on benchmark datasets and computes
detection and localisation metrics per category and overall.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from optiqual3d.config.settings import Config, EvaluationConfig
from optiqual3d.data.preprocessing import extract_patches
from optiqual3d.evaluation.metrics import (
    CategoryMetrics,
    compute_detection_metrics,
    compute_localisation_metrics,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Zero-shot evaluation pipeline.

    Runs inference on evaluation benchmarks without any fine-tuning
    and computes standard metrics.

    Attributes:
        model: Trained OptiQual3D model.
        eval_cfg: Evaluation configuration.
        device: Target device.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        device: torch.device | None = None,
    ) -> None:
        """Initialise the evaluator.

        Args:
            model: Trained (and frozen) OptiQual3D model.
            cfg: Full configuration.
            device: Target device.
        """
        self.model = model
        self.cfg = cfg
        self.eval_cfg: EvaluationConfig = cfg.evaluation
        self.device = device or torch.device("cuda")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        dataloader: DataLoader[Any],
        dataset_name: str,
    ) -> dict[str, CategoryMetrics]:
        """Evaluate on a full benchmark dataset.

        Iterates over the dataloader, collects predictions per category,
        and computes metrics.

        Args:
            dataloader: DataLoader for the evaluation dataset.
            dataset_name: Name of the dataset (for logging).

        Returns:
            Dictionary mapping category name to :class:`CategoryMetrics`.
        """
        logger.info("Evaluating on %s (%d batches)", dataset_name, len(dataloader))
        self.model.eval()

        # Accumulate predictions per category
        predictions: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: {
                "labels": [],
                "global_scores": [],
                "point_masks": [],
                "point_scores": [],
            }
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval {dataset_name}"):
                batch_preds = self._predict_batch(batch)

                for i in range(len(batch_preds["labels"])):
                    cat = batch_preds["categories"][i]
                    predictions[cat]["labels"].append(batch_preds["labels"][i])
                    predictions[cat]["global_scores"].append(
                        batch_preds["global_scores"][i]
                    )
                    predictions[cat]["point_masks"].append(
                        batch_preds["point_masks"][i]
                    )
                    predictions[cat]["point_scores"].append(
                        batch_preds["point_scores"][i]
                    )

        # Compute metrics per category
        results: dict[str, CategoryMetrics] = {}
        for category, preds in predictions.items():
            metrics = self._compute_category_metrics(category, preds)
            results[category] = metrics
            logger.info(
                "%s | %s | I-AUROC: %.2f%% | AU-PRO: %.2f%%",
                dataset_name,
                category,
                metrics.detection.auroc * 100,
                metrics.localisation.au_pro * 100,
            )

        # Log overall averages
        self._log_summary(dataset_name, results)

        return results

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def _predict_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """Run inference on a single batch.

        Args:
            batch: Batch dictionary from the evaluation dataset.

        Returns:
            Predictions dictionary with per-sample results.
        """
        points = batch["points"].to(self.device)
        labels = batch["label"]
        masks = batch["mask"]
        categories = batch["category"]

        b, n, _ = points.shape
        pc_cfg = self.cfg.data.point_cloud

        # Vectorised patch extraction across the batch
        points_np = points.cpu().numpy()
        patch_centroid_pairs = [
            extract_patches(points_np[i], pc_cfg.num_patches, pc_cfg.patch_size)
            for i in range(b)
        ]
        patches_tensor = torch.stack(
            [torch.from_numpy(p).float() for p, _ in patch_centroid_pairs]
        ).to(self.device)
        centroids_tensor = torch.stack(
            [torch.from_numpy(c).float() for _, c in patch_centroid_pairs]
        ).to(self.device)

        # Model prediction
        raw_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        pred = raw_model.predict(patches_tensor, centroids_tensor, points)

        return {
            "labels": labels.cpu().numpy().tolist(),
            "global_scores": pred["global_score"].cpu().numpy().tolist(),
            "point_masks": [m.cpu().numpy() for m in masks],
            "point_scores": [
                pred["point_scores"][i].cpu().numpy() for i in range(b)
            ],
            "categories": (
                list(categories) if isinstance(categories, (list, tuple))
                else [categories] * b
            ),
        }

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_category_metrics(
        self,
        category: str,
        preds: dict[str, list[Any]],
    ) -> CategoryMetrics:
        """Compute metrics for a single category.

        Args:
            category: Category name.
            preds: Accumulated predictions for this category.

        Returns:
            :class:`CategoryMetrics` instance.
        """
        labels = np.array(preds["labels"])
        global_scores = np.array(preds["global_scores"])

        detection = compute_detection_metrics(labels, global_scores)
        localisation = compute_localisation_metrics(
            preds["point_masks"],
            preds["point_scores"],
            fpr_limit=self.eval_cfg.au_pro_fpr_limit,
        )

        return CategoryMetrics(
            category=category,
            detection=detection,
            localisation=localisation,
            num_samples=len(labels),
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log_summary(
        self,
        dataset_name: str,
        results: dict[str, CategoryMetrics],
    ) -> None:
        """Log overall summary metrics.

        Args:
            dataset_name: Name of the dataset.
            results: Per-category metrics.
        """
        if not results:
            return

        avg_auroc = np.mean([m.detection.auroc for m in results.values()])
        avg_au_pro = np.mean([m.localisation.au_pro for m in results.values()])
        avg_point_auroc = np.mean(
            [m.localisation.point_auroc for m in results.values()]
        )
        total_samples = sum(m.num_samples for m in results.values())

        logger.info(
            "%s OVERALL | I-AUROC: %.2f%% | P-AUROC: %.2f%% | "
            "AU-PRO: %.2f%% | Samples: %d",
            dataset_name,
            avg_auroc * 100,
            avg_point_auroc * 100,
            avg_au_pro * 100,
            total_samples,
        )

    def save_results(
        self,
        results: dict[str, CategoryMetrics],
        dataset_name: str,
    ) -> Path:
        """Save evaluation results to JSON.

        Args:
            results: Per-category metrics.
            dataset_name: Name for the results file.

        Returns:
            Path to the saved JSON file.
        """
        output_dir = Path(self.eval_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_name}_results.json"

        serialisable: dict[str, Any] = {}
        for category, metrics in results.items():
            serialisable[category] = {
                "detection": {
                    "auroc": metrics.detection.auroc,
                    "f1": metrics.detection.f1,
                    "precision": metrics.detection.precision,
                    "recall": metrics.detection.recall,
                    "threshold": metrics.detection.threshold,
                },
                "localisation": {
                    "au_pro": metrics.localisation.au_pro,
                    "point_auroc": metrics.localisation.point_auroc,
                    "fpr_limit": metrics.localisation.fpr_limit,
                },
                "num_samples": metrics.num_samples,
            }

        output_path.write_text(
            json.dumps(serialisable, indent=2), encoding="utf-8"
        )
        logger.info("Results saved to %s", output_path)
        return output_path
