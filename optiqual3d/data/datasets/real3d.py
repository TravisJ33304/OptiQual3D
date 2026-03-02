"""Real3D-AD dataset loader.

Loads the Real3D-AD benchmark - the largest high-precision 3D anomaly
dataset with 1,254 objects across 12 categories scanned at
0.001-0.0015 mm resolution with 360° coverage.

Reference:
    Liu et al., "Real3D-AD: A Dataset of Point Cloud Anomaly Detection", NeurIPS 2023.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from optiqual3d.config.settings import PointCloudConfig
from optiqual3d.data.preprocessing import normalize_point_cloud, sample_points

logger = logging.getLogger(__name__)


class Real3DDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for the Real3D-AD benchmark.

    Supports the standard few-shot evaluation protocol with 4 prototype
    (reference) scans per category.

    Attributes:
        root: Root directory of the Real3D-AD dataset.
        category: Category to load (``None`` = all).
        split: ``"train"`` (prototypes) or ``"test"``.
        samples: Indexed sample metadata.
    """

    CATEGORIES: list[str] = [
        "airplane",
        "candybar",
        "car",
        "chicken",
        "diamond",
        "duck",
        "fish",
        "gemstone",
        "seahorse",
        "shell",
        "starfish",
        "toffees",
    ]

    def __init__(
        self,
        root: str | Path,
        category: str | None = None,
        split: str = "test",
        point_cloud_cfg: PointCloudConfig | None = None,
        transform: Any | None = None,
    ) -> None:
        """Initialise the Real3D-AD dataset.

        Args:
            root: Path to the Real3D-AD root directory.
            category: Category name to load (``None`` = all categories).
            split: ``"train"`` for prototypes, ``"test"`` for evaluation.
            point_cloud_cfg: Point cloud processing parameters.
            transform: Optional per-sample transform callable.
        """
        self.root = Path(root)
        self.split = split
        self.pc_cfg = point_cloud_cfg or PointCloudConfig()
        self.transform = transform
        self.categories_to_load = (
            [category] if category else self.CATEGORIES
        )
        self.samples: list[dict[str, Any]] = []

        self._discover_samples()
        logger.info(
            "Real3D-AD %s: %d samples, categories=%s",
            split,
            len(self.samples),
            self.categories_to_load,
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
                - ``category``: Category name string.
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
            "category": meta["category"],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_samples(self) -> None:
        """Walk dataset directories and build sample index.

        Real3D-AD stores point clouds as ``.pcd`` files.
        The directory layout is typically::

            root/
              airplane/
                train/ (4 prototype scans)
                test/  (normal + anomalous scans)
        """
        for category in self.categories_to_load:
            cat_dir = self.root / category
            if not cat_dir.exists():
                logger.warning("Missing category dir: %s", cat_dir)
                continue

            split_dir = cat_dir / self.split
            if not split_dir.exists():
                # Try flat structure
                split_dir = cat_dir

            pcd_files = sorted(split_dir.rglob("*.pcd")) + sorted(
                split_dir.rglob("*.ply")
            )

            for pcd_file in pcd_files:
                # Determine label from filename or parent directory
                is_anomalous = self._infer_label(pcd_file)

                self.samples.append(
                    {
                        "path": pcd_file,
                        "category": category,
                        "label": is_anomalous,
                    }
                )

    @staticmethod
    def _infer_label(path: Path) -> int:
        """Infer anomaly label from file path conventions.

        Args:
            path: Path to the point cloud file.

        Returns:
            0 for normal, 1 for anomalous.
        """
        name_lower = path.stem.lower()
        parent_lower = path.parent.name.lower()
        if "good" in name_lower or "normal" in parent_lower or "train" in parent_lower:
            return 0
        return 1

    def _load_sample(
        self, meta: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Load a PCD point cloud and optional per-point GT mask.

        Args:
            meta: Sample metadata dict from :attr:`samples`.

        Returns:
            Tuple of ``(points, mask)`` where *mask* may be ``None``.
        """
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(meta["path"]))
        points = np.asarray(pcd.points, dtype=np.float32)

        if points.shape[0] == 0:
            raise RuntimeError(f"Empty point cloud: {meta['path']}")

        # Look for a matching GT text file: root/category/gt/<stem>.txt
        mask: np.ndarray | None = None
        gt_path = (
            self.root
            / meta["category"]
            / "gt"
            / f"{Path(str(meta['path'])).stem}.txt"
        )
        if gt_path.exists():
            try:
                gt_data = np.loadtxt(str(gt_path), dtype=np.float32)
                if gt_data.ndim == 2 and gt_data.shape[1] >= 4:
                    mask = gt_data[:, 3].astype(np.float32)
            except Exception:  # noqa: BLE001
                logger.warning("Could not parse GT file: %s", gt_path)

        return points, mask
