"""MVTec 3D-AD dataset loader.

Loads the MVTec 3D Anomaly Detection benchmark dataset for evaluation.
The dataset contains 10 industrial object categories with high-resolution
3D scans, including both normal and anomalous samples with pixel-level
ground truth masks.

Reference:
    Bergmann et al., "The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly
    Detection and Localization", VISAPP 2022.
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


class MVTec3DDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for MVTec 3D-AD benchmark.

    The dataset directory structure is expected to be::

        root/
          bagel/
            train/good/  *.tiff
            test/good/   *.tiff
            test/crack/  *.tiff
            ...
          cable_gland/
            ...

    Each sample includes an organized point cloud derived from the
    depth image plus an optional ground-truth anomaly mask.

    Attributes:
        root: Root directory of the MVTec 3D-AD dataset.
        category: Object category to load (e.g. ``"bagel"``).
        split: ``"train"`` or ``"test"``.
        samples: Indexed sample metadata.
    """

    CATEGORIES: list[str] = [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

    def __init__(
        self,
        root: str | Path,
        category: str | None = None,
        split: str = "test",
        point_cloud_cfg: PointCloudConfig | None = None,
        transform: Any | None = None,
    ) -> None:
        """Initialise the MVTec 3D-AD dataset.

        Args:
            root: Path to the MVTec 3D-AD root directory.
            category: Single category name to load.  ``None`` loads all.
            split: ``"train"`` or ``"test"``.
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
            "MVTec3D %s: %d samples, categories=%s",
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
        """Load and return a single sample with ground-truth labels.

        Args:
            index: Sample index.

        Returns:
            Dictionary with keys:
                - ``points``: ``(N, 3)`` point cloud tensor.
                - ``label``: 0 for normal, 1 for anomalous.
                - ``mask``: ``(N,)`` per-point binary anomaly mask.
                - ``category``: Category name string.
                - ``defect_type``: Defect type string (``"good"`` if normal).
        """
        meta = self.samples[index]
        points, mask = self._load_sample(meta)
        points = sample_points(points, self.pc_cfg.num_points)

        if self.pc_cfg.normalize:
            points = normalize_point_cloud(points)

        # If mask exists, downsample in tandem with points
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
        else:
            mask_tensor = torch.zeros(points.shape[0], dtype=torch.float32)

        is_anomalous = 0 if meta["defect_type"] == "good" else 1

        sample: dict[str, Any] = {
            "points": torch.from_numpy(points).float(),
            "label": is_anomalous,
            "mask": mask_tensor,
            "category": meta["category"],
            "defect_type": meta["defect_type"],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_samples(self) -> None:
        """Walk dataset directories and build sample index.

        Populates :attr:`samples` with per-file metadata dicts.
        """
        for category in self.categories_to_load:
            cat_dir = self.root / category / self.split
            if not cat_dir.exists():
                logger.warning("Missing category dir: %s", cat_dir)
                continue

            for defect_dir in sorted(cat_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_type = defect_dir.name

                # Look for 3D data files (TIFF depth maps or pre-extracted PLY)
                data_files = sorted(defect_dir.rglob("xyz/*.tiff")) or sorted(
                    defect_dir.rglob("*.ply")
                ) or sorted(defect_dir.rglob("*.npy"))

                # Ground-truth mask directory
                gt_dir = self.root / category / self.split / defect_type / "gt"

                for data_file in data_files:
                    stem = data_file.stem
                    gt_path = (gt_dir / f"{stem}.png") if gt_dir.exists() else None

                    self.samples.append(
                        {
                            "path": data_file,
                            "gt_path": gt_path,
                            "category": category,
                            "defect_type": defect_type,
                        }
                    )

    def _load_sample(
        self, meta: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Load a point cloud and optional ground-truth mask.

        Args:
            meta: Sample metadata dict from :attr:`samples`.

        Returns:
            Tuple of ``(points, mask)`` where *mask* may be ``None``.
        """
        import cv2

        path = str(meta["path"])
        suffix = meta["path"].suffix.lower()

        if suffix in (".tiff", ".tif"):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Could not read TIFF: {path}")

            # Filter out background pixels (all-zero or non-finite)
            valid = np.any(img != 0, axis=-1) & np.all(np.isfinite(img), axis=-1)
            points = img[valid].astype(np.float32)

            # Load per-pixel ground-truth mask and align to valid points
            per_point_mask: np.ndarray | None = None
            gt_path = meta.get("gt_path")
            if gt_path is not None and Path(str(gt_path)).exists():
                gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if gt_img is not None:
                    flat_gt = (gt_img > 0).flatten()
                    flat_valid = valid.flatten()
                    per_point_mask = flat_gt[flat_valid].astype(np.float32)

            return points, per_point_mask

        elif suffix in (".ply",):
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points, dtype=np.float32)
            return points, None

        elif suffix == ".npy":
            points = np.load(path).astype(np.float32)
            if points.ndim == 2 and points.shape[1] > 3:
                points = points[:, :3]
            return points, None

        else:
            raise RuntimeError(f"Unsupported MVTec3D file format: {path}")
