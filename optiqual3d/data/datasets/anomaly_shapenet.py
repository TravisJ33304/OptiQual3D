"""Anomaly-ShapeNet dataset loader.

Loads the Anomaly-ShapeNet synthetic benchmark with 1,600 point cloud
samples across 40 categories for controlled zero-shot evaluation.

Reference:
    Li et al., "Towards Scalable 3D Anomaly Detection and Localization:
    A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning
    Network", CVPR 2024.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from optiqual3d.config.settings import PointCloudConfig
from optiqual3d.data.preprocessing import normalize_point_cloud, sample_points

logger = logging.getLogger(__name__)


class AnomalyShapeNetDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for Anomaly-ShapeNet benchmark.

    Provides controlled synthetic anomalies with ground-truth per-point
    labels for evaluating anomaly detection and localisation.

    Attributes:
        root: Root directory of Anomaly-ShapeNet.
        categories: Categories to load.
        split: ``"train"`` or ``"test"``.
        samples: Indexed sample metadata.
    """

    def __init__(
        self,
        root: str | Path,
        categories: list[str] | None = None,
        split: str = "test",
        point_cloud_cfg: PointCloudConfig | None = None,
        transform: (
            Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None
        ) = None,
    ) -> None:
        """Initialise the Anomaly-ShapeNet dataset.

        Args:
            root: Path to the Anomaly-ShapeNet root directory.
            categories: Category names to include (``None`` = all).
            split: ``"train"`` or ``"test"``.
            point_cloud_cfg: Point cloud processing parameters.
            transform: Optional per-sample transform.
        """
        self.root = Path(root)
        self.split = split
        self.pc_cfg = point_cloud_cfg or PointCloudConfig()
        self.transform = transform
        self.samples: list[dict[str, torch.Tensor | str | int]] = []

        self._discover_samples(categories)
        logger.info(
            "AnomalyShapeNet %s: %d samples",
            split,
            len(self.samples),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load and return a single sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with keys:
                - ``points``: ``(N, 3)`` point cloud tensor.
                - ``label``: 0 for normal, 1 for anomalous.
                - ``mask``: ``(N,)`` per-point anomaly mask.
                - ``category``: Category name.
        """
        meta = self.samples[index]
        points, mask = self._load_sample(meta)
        points = sample_points(points, self.pc_cfg.num_points)

        if self.pc_cfg.normalize:
            points = normalize_point_cloud(points)

        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
        else:
            mask_tensor = torch.zeros(points.shape[0], dtype=torch.float32)

        sample: dict[str, Any] = {
            "points": torch.from_numpy(points).float(),
            "label": meta.get("label", 0),
            "mask": mask_tensor,
            "category": meta.get("category", "unknown"),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_samples(self, categories: list[str] | None = None) -> None:
        """Scan dataset directory for point cloud files.

        Args:
            categories: Optional filter list of category names.
        """
        split_dir = self.root / self.split
        if not split_dir.exists():
            split_dir = self.root  # Flat layout fallback

        for cat_dir in sorted(split_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            if categories and cat_dir.name not in categories:
                continue

            for pcd_file in sorted(cat_dir.rglob("*.npz")):
                label = 0 if "good" in pcd_file.stem.lower() else 1
                self.samples.append(
                    {
                        "path": str(pcd_file),
                        "category": cat_dir.name,
                        "label": label,
                    }
                )

    def _load_sample(
        self, meta: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Load a point cloud and optional mask from a NPZ file.

        Args:
            meta: Sample metadata dict.

        Returns:
            Tuple of ``(points, mask)``.
        """
        data = np.load(meta["path"])
        points = data["points"].astype(np.float32)
        mask = data["mask"].astype(np.float32) if "mask" in data else None
        return points, mask
